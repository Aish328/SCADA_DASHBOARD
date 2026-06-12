"""
data_preprocessing.py
=====================
SCADA feeder data preprocessing pipeline.

Steps
-----
1.  Load the raw CSV.
2.  Parse unit-laden strings to numerics:
        "1.53 MW" / "850 kW"  -> MW   (kW values divided by 1000)
        "74.9 A"  / "74.9 AM" -> A    (unit typos tolerated)
        "10.8 kV"             -> kV
3.  Parse `Time` (auto-detects MM/DD vs DD/MM by checking which format
    yields the expected ~3-minute sampling interval) and sort.
4.  Per feeder: de-duplicate timestamps, resample to a regular 3-minute
    grid, repair gaps by time-interpolation.
5.  Outlier handling: physical-bound clipping + rolling-median/MAD filter.
6.  Feature engineering: calendar features + electrical features
    (average current/voltage, current/voltage imbalance).
7.  Emit one NeuralForecast-ready frame per feeder
    (columns: unique_id, ds, y, <exogenous features>).

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

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("preprocess")

_NUM_UNIT_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*([A-Za-z]*)")

# --------------------------------------------------------------------------- #
# Unit parsing
# --------------------------------------------------------------------------- #
def parse_value_unit(raw: object) -> tuple[float, str]:
    """Extract (value, unit) from strings like '1.53 MW'. Returns (nan, '') on failure."""
    if pd.isna(raw):
        return np.nan, ""
    m = _NUM_UNIT_RE.search(str(raw))
    if not m:
        return np.nan, ""
    return float(m.group(1)), m.group(2).upper()


def parse_load_mw(raw: object) -> float:
    """'1.53 MW' -> 1.53 ;  '850 kW' -> 0.85 ;  bare numbers assumed MW."""
    value, unit = parse_value_unit(raw)
    if np.isnan(value):
        return np.nan
    if unit.startswith("KW"):
        return value / 1000.0
    if unit.startswith("W") and unit != "":          # plain watts (defensive)
        return value / 1e6
    return value                                      # MW or unit-less


def parse_current_a(raw: object) -> float:
    """'74.9 A' (or typo 'AM') -> 74.9."""
    value, _ = parse_value_unit(raw)
    return value


def parse_voltage_kv(raw: object) -> float:
    """'10.8 kV' -> 10.8 ;  bare volts converted to kV."""
    value, unit = parse_value_unit(raw)
    if np.isnan(value):
        return np.nan
    if unit == "V":
        return value / 1000.0
    return value


# --------------------------------------------------------------------------- #
# Timestamp parsing
# --------------------------------------------------------------------------- #
def parse_time_column(series: pd.Series) -> pd.Series:
    """
    Try each candidate format; keep the one that (a) parses the most rows and
    (b) yields a median sampling interval closest to EXPECTED_FREQ.
    Guards against the classic MM/DD vs DD/MM ambiguity.
    """
    expected = pd.Timedelta(cfg.EXPECTED_FREQ)
    best, best_score = None, None
    for fmt in cfg.TIME_FORMATS:
        parsed = pd.to_datetime(series, format=fmt, errors="coerce")
        n_bad = int(parsed.isna().sum())
        med = parsed.sort_values().diff().median()
        interval_err = abs((med - expected).total_seconds()) if pd.notna(med) else np.inf
        score = (n_bad, interval_err)
        if best_score is None or score < best_score:
            best, best_score, best_fmt = parsed, score, fmt
    log.info("Timestamp format selected: %s (unparsed rows: %d)", best_fmt, best_score[0])
    return best


# --------------------------------------------------------------------------- #
# Outlier handling
# --------------------------------------------------------------------------- #
def remove_outliers(s: pd.Series,
                    window: int = cfg.OUTLIER_ROLLING_WINDOW,
                    k: float = cfg.OUTLIER_MAD_THRESHOLD,
                    non_negative: bool = True) -> pd.Series:
    """
    Replace outliers with NaN (later interpolated):
      * physically impossible values (negative load/current/voltage),
      * points deviating > k rolling-MADs from the rolling median.
    """
    s = s.copy()
    if non_negative:
        s[s < 0] = np.nan
    med = s.rolling(window, center=True, min_periods=3).median()
    mad = (s - med).abs().rolling(window, center=True, min_periods=3).median()
    mad = mad.replace(0, np.nan).ffill().bfill()
    mask = (s - med).abs() > k * 1.4826 * mad
    n = int(mask.sum())
    if n:
        s[mask] = np.nan
    return s


# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #
def add_electrical_features(df: pd.DataFrame) -> pd.DataFrame:
    ir, iy, ib = (df[c] for c in cfg.CURRENT_COLS)
    vry, vyb, vbr = (df[c] for c in cfg.VOLTAGE_COLS)

    df["avg_current"] = (ir + iy + ib) / 3.0
    # NEMA-style imbalance: max deviation from mean / mean (%)
    i_dev = pd.concat([(ir - df["avg_current"]).abs(),
                       (iy - df["avg_current"]).abs(),
                       (ib - df["avg_current"]).abs()], axis=1).max(axis=1)
    df["current_imbalance_pct"] = 100.0 * i_dev / df["avg_current"].replace(0, np.nan)

    df["avg_voltage"] = (vry + vyb + vbr) / 3.0
    v_dev = pd.concat([(vry - df["avg_voltage"]).abs(),
                       (vyb - df["avg_voltage"]).abs(),
                       (vbr - df["avg_voltage"]).abs()], axis=1).max(axis=1)
    df["voltage_imbalance_pct"] = 100.0 * v_dev / df["avg_voltage"].replace(0, np.nan)
    return df


def add_calendar_features(df: pd.DataFrame, ts_col: str = "ds") -> pd.DataFrame:
    ts = df[ts_col]
    df["hour"] = ts.dt.hour.astype(float)
    df["minute"] = ts.dt.minute.astype(float)
    df["day_of_week"] = ts.dt.dayofweek.astype(float)
    df["is_weekend"] = (ts.dt.dayofweek >= 5).astype(float)
    # Cyclical time-of-day encoding: bounded, smooth, and varies within every
    # training window — this is what lets the model condition on the daily
    # pattern. Raw hour/minute/day_of_week are kept for EDA but excluded from
    # FUTR_EXOG because they are (near-)constant inside a 4.8 h window, which
    # breaks per-window robust scaling (zero IQR) and destabilises training.
    tod = (ts.dt.hour + ts.dt.minute / 60.0) / 24.0
    df["tod_sin"] = np.sin(2 * np.pi * tod)
    df["tod_cos"] = np.cos(2 * np.pi * tod)
    return df


# --------------------------------------------------------------------------- #
# Per-feeder pipeline
# --------------------------------------------------------------------------- #
def build_feeder_frame(raw: pd.DataFrame, feeder: str) -> pd.DataFrame:
    """Filter, clean, regularise and feature-engineer a single feeder series."""
    g = raw[raw[cfg.FEEDER_COL] == feeder].copy()
    if g.empty:
        raise ValueError(f"No rows found for feeder {feeder!r}")

    g = g.sort_values("ds").drop_duplicates(subset="ds", keep="last").set_index("ds")

    numeric_cols = ["y"] + cfg.CURRENT_COLS + cfg.VOLTAGE_COLS
    g = g[numeric_cols]

    # Regular 3-minute grid (NeuralForecast requires a fixed frequency)
    g = g.resample(cfg.EXPECTED_FREQ).mean()

    # Outlier removal then gap repair
    for c in numeric_cols:
        g[c] = remove_outliers(g[c])
    n_missing = int(g["y"].isna().sum())
    g = g.interpolate(method="time", limit_direction="both")

    g = g.reset_index()
    g = add_electrical_features(g)
    g = add_calendar_features(g)
    g["unique_id"] = cfg.FEEDERS[feeder]

    log.info("Feeder %-22s -> %4d rows on regular grid (%d repaired/outlier points)",
             feeder, len(g), n_missing)

    # Keep ALL engineered + calendar features for EDA; training subsets later.
    all_feats = list(dict.fromkeys(cfg.EDA_FEATURES + cfg.HIST_EXOG
                                   + cfg.FUTR_EXOG + cfg.CALENDAR_FEATURES))
    ordered = ["unique_id", "ds", "y"] + all_feats
    return g[ordered]


def load_and_preprocess(data_path: Path | str = cfg.RAW_DATA_PATH) -> dict[str, pd.DataFrame]:
    """Full pipeline. Returns {feeder_name: NeuralForecast-ready DataFrame}."""
    log.info("Loading raw data: %s", data_path)
    raw = pd.read_csv(data_path)

    # ---- unit parsing -----------------------------------------------------
    raw["y"] = raw[cfg.TARGET_COL].map(parse_load_mw)
    for c in cfg.CURRENT_COLS:
        raw[c] = raw[c].map(parse_current_a)
    for c in cfg.VOLTAGE_COLS:
        raw[c] = raw[c].map(parse_voltage_kv)

    # ---- timestamps -------------------------------------------------------
    raw["ds"] = parse_time_column(raw[cfg.TIME_COL])
    raw = raw.dropna(subset=["ds", "y"])

    # ---- per feeder -------------------------------------------------------
    frames: dict[str, pd.DataFrame] = {}
    for feeder in cfg.FEEDERS:
        frames[feeder] = build_feeder_frame(raw, feeder)

    return frames


def chronological_split_sizes(n: int) -> tuple[int, int, int]:
    """Return (n_train, n_val, n_test) for a 70/15/15 chronological split."""
    n_test = int(round(n * cfg.TEST_FRAC))
    n_val = int(round(n * cfg.VAL_FRAC))
    n_train = n - n_val - n_test
    return n_train, n_val, n_test


def save_processed(frames: dict[str, pd.DataFrame]) -> None:
    combined = pd.concat(frames.values(), ignore_index=True)
    combined.to_csv(cfg.PROCESSED_DIR / "all_feeders_processed.csv", index=False)
    for feeder, df in frames.items():
        fn = cfg.PROCESSED_DIR / f"{cfg.FEEDERS[feeder]}_processed.csv"
        df.to_csv(fn, index=False)
    log.info("Processed data written to %s", cfg.PROCESSED_DIR)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Preprocess SCADA feeder data")
    ap.add_argument("--data", default=str(cfg.RAW_DATA_PATH), help="Path to raw data.csv")
    args = ap.parse_args()

    frames = load_and_preprocess(args.data)
    save_processed(frames)
    for feeder, df in frames.items():
        tr, va, te = chronological_split_sizes(len(df))
        log.info("%-22s n=%4d  train=%d val=%d test=%d  span %s -> %s",
                 feeder, len(df), tr, va, te, df["ds"].min(), df["ds"].max())
