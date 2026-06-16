"""
data_preprocessing.py
=====================
SCADA feeder data preprocessing pipeline for the GLOBAL NHITS model.

Key difference from the per-feeder pipeline:
  - `load_and_preprocess()` returns a SINGLE concatenated DataFrame
    (all feeders, identified by unique_id) instead of a dict of frames.
  - NeuralForecast natively handles multiple series via `unique_id`.

Steps
-----
1.  Load raw CSV; parse unit-laden strings → numerics.
2.  Auto-detect timestamp format (MM/DD vs DD/MM by interval heuristic).
3.  Per feeder: de-duplicate, resample to 3-min grid, repair gaps.
4.  Outlier handling: rolling-median/MAD filter + physical-bound clipping.
5.  Feature engineering: calendar (cyclical ToD) + electrical (avg current, imbalance).
6.  Concatenate all feeder frames → single NeuralForecast-ready DataFrame.
7.  Assign `unique_id` from FEEDER_ID_MAP (short alphanumeric, stable across runs).

Run directly to materialise processed CSVs:
    python data_preprocessing.py [--data path/to/data.csv]
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


# --------------------------------------------------------------------------- #
# Unit parsers
# --------------------------------------------------------------------------- #
def _parse_value_unit(raw: object) -> tuple[float, str]:
    """Extract (value, unit-string) from '1.53 MW', '74.9 A', '10.8 kV', etc."""
    if pd.isna(raw):
        return np.nan, ""
    m = _NUM_UNIT_RE.search(str(raw))
    if not m:
        return np.nan, ""
    return float(m.group(1)), m.group(2).upper()


def parse_load_mw(raw: object) -> float:
    """'1.53 MW' → 1.53 | '850 kW' → 0.85 | bare number assumed MW."""
    v, u = _parse_value_unit(raw)
    if np.isnan(v):
        return np.nan
    if u.startswith("KW"):
        return v / 1000.0
    if u == "W":                     # plain watts (defensive)
        return v / 1e6
    return v                         # MW or unit-less


def parse_current_a(raw: object) -> float:
    """'74.9 A' (or typo 'AM') → 74.9."""
    v, _ = _parse_value_unit(raw)
    return v


def parse_voltage_kv(raw: object) -> float:
    """'10.8 kV' → 10.8 | bare volts → kV."""
    v, u = _parse_value_unit(raw)
    if np.isnan(v):
        return np.nan
    return v / 1000.0 if u == "V" else v


# --------------------------------------------------------------------------- #
# Timestamp parsing — auto-selects MM/DD vs DD/MM
# --------------------------------------------------------------------------- #
def parse_time_column(series: pd.Series) -> pd.Series:
    """
    Try each candidate format in cfg.TIME_FORMATS.
    Score = (n_unparsed, |median_interval − EXPECTED_FREQ|).
    The format with the lowest score wins (fewest bad rows and closest
    to the expected 3-minute SCADA polling cadence).
    """
    expected = pd.Timedelta(cfg.EXPECTED_FREQ)
    best, best_score, best_fmt = None, None, None
    for fmt in cfg.TIME_FORMATS:
        parsed = pd.to_datetime(series, format=fmt, errors="coerce")
        n_bad  = int(parsed.isna().sum())
        med    = parsed.sort_values().diff().median()
        interval_err = abs((med - expected).total_seconds()) if pd.notna(med) else np.inf
        score  = (n_bad, interval_err)
        if best_score is None or score < best_score:
            best, best_score, best_fmt = parsed, score, fmt
    log.info("Timestamp format selected: %s  (unparsed rows: %d)",
             best_fmt, best_score[0])
    return best


# --------------------------------------------------------------------------- #
# Outlier handling
# --------------------------------------------------------------------------- #
def remove_outliers(s: pd.Series,
                    window: int = cfg.OUTLIER_ROLLING_WINDOW,
                    k: float = cfg.OUTLIER_MAD_THRESHOLD,
                    non_negative: bool = True) -> pd.Series:
    """
    Mark outliers as NaN (interpolated later):
      • negative values (physical impossibility)
      • points > k rolling-MADs from the rolling median
    Uses a centred window with min_periods=3 for robustness at series edges.
    """
    s = s.copy()
    if non_negative:
        s[s < 0] = np.nan
    med = s.rolling(window, center=True, min_periods=3).median()
    mad = (s - med).abs().rolling(window, center=True, min_periods=3).median()
    mad = mad.replace(0, np.nan).ffill().bfill()
    mask = (s - med).abs() > k * 1.4826 * mad   # 1.4826 makes MAD consistent with σ
    if mask.sum():
        s[mask] = np.nan
    return s


# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #
def add_electrical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    avg_current           – mean of three-phase currents (informative for load)
    current_imbalance_pct – NEMA-style: max phase deviation / mean (%) — EDA only
    avg_voltage           – mean of three-phase voltages (EDA only; near-constant)
    voltage_imbalance_pct – max phase deviation / mean (%) — EDA only
    """
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
    """
    Cyclical time-of-day (tod_sin / tod_cos):
      • Bounded and continuous across midnight — safe for robust scaler.
      • Varies within every 4.8 h history window — gives the model diurnal signal.
    Raw hour/minute/day_of_week kept for EDA but excluded from FUTR_EXOG
    because they are near-constant over a single window (zero IQR breaks
    window-level robust normalisation and destabilises gradient flow).
    """
    ts = df[ts_col]
    df["hour"]        = ts.dt.hour.astype(float)
    df["minute"]      = ts.dt.minute.astype(float)
    df["day_of_week"] = ts.dt.dayofweek.astype(float)
    df["is_weekend"]  = (ts.dt.dayofweek >= 5).astype(float)
    tod = (ts.dt.hour + ts.dt.minute / 60.0) / 24.0
    df["tod_sin"] = np.sin(2 * np.pi * tod)
    df["tod_cos"] = np.cos(2 * np.pi * tod)
    return df


# --------------------------------------------------------------------------- #
# Per-feeder pipeline  (called internally; builds one clean series frame)
# --------------------------------------------------------------------------- #
def _build_feeder_frame(raw: pd.DataFrame, feeder: str) -> pd.DataFrame:
    """
    Filter rows for one feeder → clean → regularise → feature-engineer.
    Returns a NeuralForecast-ready frame with unique_id set to the feeder's
    short alphanumeric ID from FEEDER_ID_MAP.
    """
    uid = cfg.FEEDER_ID_MAP[feeder]
    g   = raw[raw[cfg.FEEDER_COL] == feeder].copy()
    if g.empty:
        raise ValueError(f"No rows found for feeder {feeder!r}")

    g = (g.sort_values("ds")
          .drop_duplicates(subset="ds", keep="last")
          .set_index("ds"))

    numeric_cols = ["y"] + cfg.CURRENT_COLS + cfg.VOLTAGE_COLS
    g = g[numeric_cols]

    # Resample to regular 3-min grid (NeuralForecast requires fixed frequency)
    g = g.resample(cfg.EXPECTED_FREQ).mean()

    # Remove outliers then repair gaps by time-aware linear interpolation
    for c in numeric_cols:
        g[c] = remove_outliers(g[c])
    n_repaired = int(g["y"].isna().sum())
    g = g.interpolate(method="time", limit_direction="both")

    g = g.reset_index()
    g = add_electrical_features(g)
    g = add_calendar_features(g)

    # unique_id is the stable short code — NeuralForecast groups series by this
    g["unique_id"] = uid

    log.info("Feeder %-26s (uid=%s)  ->  %4d rows  (%d repaired/outlier points)",
             feeder, uid, len(g), n_repaired)

    # Keep ALL features for EDA; training code subsets to model inputs later
    all_feats = list(dict.fromkeys(
        cfg.EDA_FEATURES + cfg.HIST_EXOG + cfg.FUTR_EXOG + cfg.CALENDAR_FEATURES))
    ordered   = ["unique_id", "ds", "y"] + all_feats
    return g[ordered]


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def _coerce_numerics(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Coerce active_load, current, and voltage columns to float.

    PostgreSQL FLOAT/NUMERIC columns arrive as Python floats already — the
    pd.to_numeric() call is a no-op for those.  If the DB stores unit-laden
    strings ("1.53 MW", "74.9 A") the unit parsers are applied as fallback.
    Both paths produce a float column; no data is lost either way.
    """
    # --- active_load → MW ----------------------------------------------------
    sample = raw[cfg.TARGET_COL].dropna().iloc[:1]
    if sample.empty or pd.to_numeric(sample, errors="coerce").notna().all():
        # Already numeric (DB stores FLOAT) — direct cast, MW assumed
        raw["y"] = pd.to_numeric(raw[cfg.TARGET_COL], errors="coerce")
    else:
        # String values like "1.53 MW" — use unit parser
        raw["y"] = raw[cfg.TARGET_COL].map(parse_load_mw)

    # --- currents → A --------------------------------------------------------
    for c in cfg.CURRENT_COLS:
        sample_c = raw[c].dropna().iloc[:1]
        if sample_c.empty or pd.to_numeric(sample_c, errors="coerce").notna().all():
            raw[c] = pd.to_numeric(raw[c], errors="coerce")
        else:
            raw[c] = raw[c].map(parse_current_a)

    # --- voltages → kV -------------------------------------------------------
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
    """
    Full preprocessing pipeline — reads from PostgreSQL.

    Parameters
    ----------
    feeders  : Feeder names to include (default: all in FEEDER_ID_MAP).
    start_dt : Earliest timestamp, ISO string e.g. '2026-01-01 00:00:00'.
    end_dt   : Latest  timestamp, ISO string.
    limit    : Row cap (useful for quick smoke-tests).

    Returns
    -------
    pd.DataFrame
        Single concatenated NeuralForecast-ready frame, columns:
            unique_id | ds | y | ir | iy | ib | avg_current |
            tod_sin | tod_cos | [EDA features …]
        Sorted by (unique_id, ds).

    Data source
    -----------
    Calls cfg.load_raw_data() which issues a parameterised SELECT against
    the dashboard PostgreSQL table.  Connection is configured via env vars —
    see config.py for the full list (DB_HOST, DB_PORT, DB_NAME, DB_USER,
    DB_PASSWORD, DB_TABLE, DB_SCHEMA).
    """
    # ---- 1. Fetch from PostgreSQL ------------------------------------------
    raw = cfg.load_raw_data(
        feeders  = feeders,
        start_dt = start_dt,
        end_dt   = end_dt,
        limit    = limit,
    )

    # ---- 2. Numeric coercion (handles both float and unit-string columns) ---
    raw = _coerce_numerics(raw)

    # ---- 3. Timestamps ------------------------------------------------------
    # psycopg2 / pandas read_sql returns datetime objects for TIMESTAMP columns;
    # pd.to_datetime() is a safe no-op in that case.  If the column is a string
    # (e.g. TEXT column), parse_time_column() auto-selects MM/DD vs DD/MM.
    if pd.api.types.is_datetime64_any_dtype(raw[cfg.TIME_COL]):
        raw["ds"] = pd.to_datetime(raw[cfg.TIME_COL], utc=False)
    else:
        raw["ds"] = parse_time_column(raw[cfg.TIME_COL])

    raw = raw.dropna(subset=["ds", "y"])

    # ---- 4. Per-feeder processing → concatenate ----------------------------
    target_feeders = feeders or list(cfg.FEEDER_ID_MAP.keys())
    parts: list[pd.DataFrame] = []
    for feeder in target_feeders:
        if feeder not in raw[cfg.FEEDER_COL].values:
            log.warning("Feeder %r not found in DB results — skipping.", feeder)
            continue
        parts.append(_build_feeder_frame(raw, feeder))

    if not parts:
        raise RuntimeError(
            "No feeder data could be processed. "
            "Check DB_TABLE contents and FEEDER_ID_MAP keys."
        )

    combined = (pd.concat(parts, ignore_index=True)
                  .sort_values(["unique_id", "ds"])
                  .reset_index(drop=True))
    log.info("Combined frame: %d rows across %d feeders",
             len(combined), combined["unique_id"].nunique())
    return combined


def chronological_split_sizes(n: int) -> tuple[int, int, int]:
    """Return (n_train, n_val, n_test) for a 70 / 15 / 15 chronological split."""
    n_test  = int(round(n * cfg.TEST_FRAC))
    n_val   = int(round(n * cfg.VAL_FRAC))
    n_train = n - n_val - n_test
    return n_train, n_val, n_test


def save_processed(df: pd.DataFrame) -> None:
    """Save the combined frame and per-feeder CSVs for audit / EDA."""
    df.to_csv(cfg.PROCESSED_DIR / "all_feeders_processed.csv", index=False)
    for uid, grp in df.groupby("unique_id"):
        grp.to_csv(cfg.PROCESSED_DIR / f"{uid}_processed.csv", index=False)
    log.info("Processed data written to %s", cfg.PROCESSED_DIR)


# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Preprocess SCADA feeder data from PostgreSQL (global model)"
    )
    ap.add_argument("--start",  default=None, help="Start datetime ISO e.g. 2026-01-01 00:00:00")
    ap.add_argument("--end",    default=None, help="End datetime ISO")
    ap.add_argument("--limit",  default=None, type=int, help="Row cap for testing")
    args = ap.parse_args()

    df = load_and_preprocess(start_dt=args.start, end_dt=args.end, limit=args.limit)
    save_processed(df)

    print("\nPer-feeder summary:")
    for uid, grp in df.groupby("unique_id"):
        name = cfg.ID_FEEDER_MAP.get(uid, uid)
        tr, va, te = chronological_split_sizes(len(grp))
        log.info("  %-8s  %-26s  n=%4d  train=%d val=%d test=%d  %s → %s",
                 uid, name, len(grp), tr, va, te,
                 grp["ds"].min(), grp["ds"].max())