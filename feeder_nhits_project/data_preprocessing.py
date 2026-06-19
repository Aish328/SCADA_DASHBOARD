"""
data_preprocessing.py
=====================
SCADA feeder data preprocessing pipeline for the GLOBAL NHITS model.
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

import config as cfg

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("preprocess")

_NUM_UNIT_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*([A-Za-z]*)")


def _parse_value_unit(raw: object) -> tuple[float, str]:
    if pd.isna(raw):
        return np.nan, ""
    m = _NUM_UNIT_RE.search(str(raw))
    if not m:
        return np.nan, ""
    return float(m.group(1)), m.group(2).upper()


def parse_load_mw(raw: object) -> float:
    v, u = _parse_value_unit(raw)
    if np.isnan(v):
        return np.nan
    if u.startswith("KW"):
        return v / 1000.0
    if u == "W":
        return v / 1e6
    return v


def parse_current_a(raw: object) -> float:
    v, _ = _parse_value_unit(raw)
    return v


def parse_voltage_kv(raw: object) -> float:
    v, u = _parse_value_unit(raw)
    if np.isnan(v):
        return np.nan
    return v / 1000.0 if u == "V" else v


def parse_time_column(series: pd.Series) -> pd.Series:
    expected = pd.Timedelta(cfg.EXPECTED_FREQ)
    best, best_score, best_fmt = None, None, None

    for fmt in cfg.TIME_FORMATS:
        try:
            parsed = pd.to_datetime(series, format=fmt, errors="coerce")
        except ValueError:
            continue
        n_bad = int(parsed.isna().sum())
        valid = parsed.dropna()
        if len(valid) < 2:
            continue
        med = valid.sort_values().diff().median()
        interval_err = abs((med - expected).total_seconds()) if pd.notna(med) else np.inf
        score = (n_bad, interval_err)
        if best_score is None or score < best_score:
            best, best_score, best_fmt = parsed, score, fmt

    if best is None:
        best = pd.to_datetime(series, errors="coerce")
        best_fmt = "inferred"

    log.info("Timestamp format selected: %s  (unparsed rows: %d)",
             best_fmt, best_score[0] if best_score else int(best.isna().sum()))
    return best


def remove_outliers(s: pd.Series,
                    window: int = cfg.OUTLIER_ROLLING_WINDOW,
                    k: float = cfg.OUTLIER_MAD_THRESHOLD,
                    non_negative: bool = True) -> pd.Series:
    s = s.copy()
    if non_negative:
        s[s < 0] = np.nan
    med = s.rolling(window, center=True, min_periods=3).median()
    mad = (s - med).abs().rolling(window, center=True, min_periods=3).median()
    mad = mad.replace(0, np.nan).ffill().bfill()
    mask = (s - med).abs() > k * 1.4826 * mad
    if mask.sum():
        s[mask] = np.nan
    return s


def add_electrical_features(df: pd.DataFrame) -> pd.DataFrame:
    ir, iy, ib  = (df[c] for c in cfg.CURRENT_COLS)
    vry, vyb, vbr = (df[c] for c in cfg.VOLTAGE_COLS)

    avg_i = (ir + iy + ib) / 3.0
    df["avg_current"] = avg_i
    i_dev = pd.concat([(ir - avg_i).abs(),
                       (iy - avg_i).abs(),
                       (ib - avg_i).abs()], axis=1).max(axis=1)
    df["current_imbalance_pct"] = 100.0 * i_dev / avg_i.replace(0, np.nan)

    avg_v = (vry + vyb + vbr) / 3.0
    df["avg_voltage"] = avg_v
    v_dev = pd.concat([(vry - avg_v).abs(),
                       (vyb - avg_v).abs(),
                       (vbr - avg_v).abs()], axis=1).max(axis=1)
    df["voltage_imbalance_pct"] = 100.0 * v_dev / avg_v.replace(0, np.nan)
    return df


def add_calendar_features(df: pd.DataFrame, ts_col: str = "ds") -> pd.DataFrame:
    ts = df[ts_col]
    df["hour"]        = ts.dt.hour.astype(float)
    df["minute"]      = ts.dt.minute.astype(float)
    df["day_of_week"] = ts.dt.dayofweek.astype(float)
    df["is_weekend"]  = (ts.dt.dayofweek >= 5).astype(float)
    tod = (ts.dt.hour + ts.dt.minute / 60.0) / 24.0
    df["tod_sin"] = np.sin(2 * np.pi * tod)
    df["tod_cos"] = np.cos(2 * np.pi * tod)
    return df


def _build_feeder_frame(raw: pd.DataFrame, feeder: str) -> pd.DataFrame:
    uid = cfg.FEEDER_ID_MAP[feeder]
    g = raw[raw[cfg.FEEDER_COL] == feeder].copy()
    if g.empty:
        raise ValueError(f"No rows found for feeder {feeder!r}")

    g = (g.sort_values("ds")
          .drop_duplicates(subset="ds", keep="last")
          .set_index("ds"))

    numeric_cols = ["y"] + cfg.CURRENT_COLS + cfg.VOLTAGE_COLS
    g = g[numeric_cols]

    g = g.resample(cfg.EXPECTED_FREQ).mean()

    for c in numeric_cols:
        g[c] = remove_outliers(g[c])
    n_repaired = int(g["y"].isna().sum())
    g = g.interpolate(method="time", limit_direction="both")

    g = g.reset_index()
    g = add_electrical_features(g)
    g = add_calendar_features(g)

    g["unique_id"] = uid

    log.info("Feeder %-26s (uid=%s)  ->  %4d rows  (%d repaired)",
             feeder, uid, len(g), n_repaired)

    all_feats = list(dict.fromkeys(
        cfg.EDA_FEATURES + cfg.HIST_EXOG + cfg.FUTR_EXOG + cfg.CALENDAR_FEATURES))
    ordered = ["unique_id", "ds", "y"] + all_feats
    return g[ordered]


def _coerce_numerics(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Coerce active_load, current, and voltage columns to float.
    """
    # Debug: show what columns we have
    log.debug("Coerce numerics: available columns = %s", raw.columns.tolist())

    # active_load → MW
    sample = raw[cfg.TARGET_COL].dropna().iloc[:1]
    if sample.empty or pd.to_numeric(sample, errors="coerce").notna().all():
        raw["y"] = pd.to_numeric(raw[cfg.TARGET_COL], errors="coerce")
    else:
        raw["y"] = raw[cfg.TARGET_COL].map(parse_load_mw)

    # currents → A
    for c in cfg.CURRENT_COLS:
        sample_c = raw[c].dropna().iloc[:1]
        if sample_c.empty or pd.to_numeric(sample_c, errors="coerce").notna().all():
            raw[c] = pd.to_numeric(raw[c], errors="coerce")
        else:
            raw[c] = raw[c].map(parse_current_a)

    # voltages → kV
    for c in cfg.VOLTAGE_COLS:
        sample_v = raw[c].dropna().iloc[:1]
        if sample_v.empty or pd.to_numeric(sample_v, errors="coerce").notna().all():
            raw[c] = pd.to_numeric(raw[c], errors="coerce")
        else:
            raw[c] = raw[c].map(parse_voltage_kv)

    return raw


def load_and_preprocess(
    feeders:  list[str] | None = None,
    start_dt: str | None       = None,
    end_dt:   str | None       = None,
    limit:    int | None       = None,
) -> pd.DataFrame:
    # 1. Fetch raw data
    raw = cfg.load_raw_data(
        feeders=feeders,
        start_dt=start_dt,
        end_dt=end_dt,
        limit=limit,
    )

    # 2. Numeric coercion
    raw = _coerce_numerics(raw)

    # 3. Timestamps
    if pd.api.types.is_datetime64_any_dtype(raw[cfg.TIME_COL]):
        raw["ds"] = pd.to_datetime(raw[cfg.TIME_COL], utc=False)
    else:
        raw["ds"] = parse_time_column(raw[cfg.TIME_COL])

    raw = raw.dropna(subset=["ds", "y"])

    # 4. Per-feeder processing
    target_feeders = feeders or list(cfg.FEEDER_ID_MAP.keys())
    parts: list[pd.DataFrame] = []
    for feeder in target_feeders:
        if feeder not in raw[cfg.FEEDER_COL].values:
            log.warning("Feeder %r not found in data — skipping.", feeder)
            continue
        try:
            parts.append(_build_feeder_frame(raw, feeder))
        except ValueError as e:
            log.warning(str(e))
            continue

    if not parts:
        raise RuntimeError("No feeder data could be processed.")

    combined = (pd.concat(parts, ignore_index=True)
                  .sort_values(["unique_id", "ds"])
                  .reset_index(drop=True))
    log.info("Combined frame: %d rows across %d feeders",
             len(combined), combined["unique_id"].nunique())
    return combined


def chronological_split_sizes(n: int) -> tuple[int, int, int]:
    n_test  = int(round(n * cfg.TEST_FRAC))
    n_val   = int(round(n * cfg.VAL_FRAC))
    n_train = n - n_val - n_test
    return n_train, n_val, n_test


def save_processed(df: pd.DataFrame) -> None:
    df.to_csv(cfg.PROCESSED_DIR / "all_feeders_processed.csv", index=False)
    for uid, grp in df.groupby("unique_id"):
        grp.to_csv(cfg.PROCESSED_DIR / f"{uid}_processed.csv", index=False)
    log.info("Processed data written to %s", cfg.PROCESSED_DIR)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Preprocess SCADA feeder data")
    ap.add_argument("--start", default=None, help="Start datetime ISO")
    ap.add_argument("--end", default=None, help="End datetime ISO")
    ap.add_argument("--limit", default=None, type=int, help="Row cap")
    args = ap.parse_args()

    df = load_and_preprocess(start_dt=args.start, end_dt=args.end, limit=args.limit)
    save_processed(df)

    print("\nPer-feeder summary:")
    for uid, grp in df.groupby("unique_id"):
        name = cfg.ID_FEEDER_MAP.get(uid, uid)
        tr, va, te = chronological_split_sizes(len(grp))
        log.info("  %-8s  %-26s  n=%4d  train=%d val=%d test=%d",
                 uid, name, len(grp), tr, va, te)