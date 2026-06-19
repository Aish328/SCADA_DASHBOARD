"""
services/nhits_forecast.py
==========================
Global NHITS inference service for the SCADA dashboard.

Replaces the old per-feeder bundle approach with a single
NeuralForecast bundle (NHITS_GLOBAL) trained on all feeders.

Drop-in: the public function run_nhits_forecast() has an identical
signature and response shape to the old version — ml_inference.py
and ml.html need no changes to consume it.

Model location
--------------
Reads from the directory pointed to by NHITS_MODELS_DIR env var.
Default: <repo>/feeder_nhits_project/models/NHITS_GLOBAL

Input schema (DataLoader rows)
------------------------------
    datetime | substation | feeder | ir | iy | ib | vry | vyb | vbr | active_load
"""

from __future__ import annotations

import logging
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
_DEFAULT_MODEL_DIR = (
    Path(__file__).resolve().parent.parent
    / "feeder_nhits_project" / "models" / "NHITS_GLOBAL"
)
GLOBAL_MODEL_DIR = Path(os.getenv("NHITS_MODELS_DIR", str(_DEFAULT_MODEL_DIR)))

FREQ       = "3min"
INPUT_SIZE = 48     # must match training config.py INPUT_SIZE
HORIZON    = 20     # must match training config.py HORIZON

HIST_EXOG = ["ir", "iy", "ib", "avg_current"]
FUTR_EXOG = ["tod_sin", "tod_cos"]

# Maps feeder display name (as stored in DB) → unique_id used at training time.
# Must match FEEDER_ID_MAP in feeder_nhits_project/config.py exactly.
FEEDER_ID_MAP: dict[str, str] = {
    "11KV BHOSALE NAGAR":    "EP_FD01",
    "11KV KUBERA":           "EP_FD02",
    "RMU1":                  "ML_FD01",
    "11KV LUMAX":            "ML_FD02",
    "11KV MALWADI HADAPSAR": "ML_FD03",
}
ID_FEEDER_MAP = {v: k for k, v in FEEDER_ID_MAP.items()}

# ── Singleton model cache ─────────────────────────────────────────────────────
_nf   = None
_lock = threading.Lock()


def _get_model():
    """Load the global NeuralForecast bundle once; share across all requests."""
    global _nf
    if _nf is None:
        with _lock:
            if _nf is None:
                from neuralforecast import NeuralForecast
                path = GLOBAL_MODEL_DIR
                if not path.exists():
                    raise FileNotFoundError(
                        f"Global NHITS bundle not found at {path}. "
                        "Run feeder_nhits_project/train_global_nhits.py first."
                    )
                logger.info("Loading global NHITS bundle from %s …", path)
                _nf = NeuralForecast.load(path=str(path))
                logger.info("Global NHITS bundle ready.")
    return _nf


# ── Feeder resolution ─────────────────────────────────────────────────────────
def _norm(name: str) -> str:
    return re.sub(r"\s+", " ", str(name).strip().upper())

_NORM_LOOKUP = {_norm(k): v for k, v in FEEDER_ID_MAP.items()}


def resolve_feeder(name: str) -> Optional[str]:
    """Feeder display name → unique_id, or None if not trained."""
    return _NORM_LOOKUP.get(_norm(name))


# ── Preprocessing ─────────────────────────────────────────────────────────────
def _prepare_history(df_feeder: pd.DataFrame, uid: str) -> pd.DataFrame:
    """
    Convert DataLoader rows for one feeder into a NeuralForecast input frame.
    Mirrors feeder_nhits_project/data_preprocessing.py exactly so the model
    sees the same feature distribution it was trained on.
    """
    g = (df_feeder[["datetime", "active_load", "ir", "iy", "ib"]]
         .rename(columns={"datetime": "ds", "active_load": "y"})
         .copy())

    g["ds"] = pd.to_datetime(g["ds"])
    g = (g.dropna(subset=["ds"])
          .sort_values("ds")
          .drop_duplicates(subset="ds", keep="last")
          .set_index("ds")
          .resample(FREQ).mean()
          .interpolate(method="time", limit_direction="both")
          .reset_index())

    # Unit repair: DB rows occasionally stored in kW (value > 100 means kW)
    kw_mask = g["y"] > 100
    if kw_mask.any():
        logger.info("%s: converting %d kW-scale rows → MW", uid, int(kw_mask.sum()))
        g.loc[kw_mask, "y"] /= 1000.0

    g["avg_current"] = (g["ir"] + g["iy"] + g["ib"]) / 3.0
    tod = (g["ds"].dt.hour + g["ds"].dt.minute / 60.0) / 24.0
    g["tod_sin"]   = np.sin(2 * np.pi * tod)
    g["tod_cos"]   = np.cos(2 * np.pi * tod)
    g["unique_id"] = uid

    cols = ["unique_id", "ds", "y"] + HIST_EXOG + FUTR_EXOG
    return g[cols].dropna()


def _build_future_exog(uid: str, last_ds: pd.Timestamp) -> pd.DataFrame:
    """Build HORIZON-step future exog frame (tod_sin/cos only — known in advance)."""
    future_ds = pd.date_range(last_ds, periods=HORIZON + 1, freq=FREQ)[1:]
    f   = pd.DataFrame({"unique_id": uid, "ds": future_ds})
    tod = (f["ds"].dt.hour + f["ds"].dt.minute / 60.0) / 24.0
    f["tod_sin"] = np.sin(2 * np.pi * tod)
    f["tod_cos"] = np.cos(2 * np.pi * tod)
    return f


# ── Single-feeder inference ───────────────────────────────────────────────────
def _forecast_one(df_feeder: pd.DataFrame, uid: str) -> pd.DataFrame:
    """
    Run global NHITS for one feeder. Returns DataFrame[ds, mw].
    The global model uses the stored per-feeder scaler automatically.
    """
    nf   = _get_model()
    hist = _prepare_history(df_feeder, uid)

    if len(hist) < INPUT_SIZE:
        raise ValueError(
            f"{uid}: need ≥ {INPUT_SIZE} rows on the 3-min grid "
            f"({INPUT_SIZE * 3 // 60:.1f} h of history), have {len(hist)}."
        )

    futr = _build_future_exog(uid, hist["ds"].max())
    fcst = nf.predict(df=hist, futr_df=futr)
    fcst = fcst.reset_index() if "unique_id" not in fcst.columns else fcst

    yhat = [c for c in fcst.columns if c not in ("unique_id", "ds")][0]
    return pd.DataFrame({
        "ds": pd.to_datetime(fcst["ds"]),
        "mw": fcst[yhat].astype(float).clip(lower=0.0),  # load cannot be negative
    })


# ── Public entry point ────────────────────────────────────────────────────────
def run_nhits_forecast(
    df:             pd.DataFrame,
    substation:     Optional[str] = None,
    feeder:         Optional[str] = None,
    history_points: int           = INPUT_SIZE,
) -> dict:
    """
    Main entry point called by ml_inference.run_load_forecast().

    Parameters
    ----------
    df              : DataLoader rows already filtered by the caller.
    substation      : Selected substation (informational).
    feeder          : Selected feeder display name, or None / "All".
    history_points  : Recent history steps to include in the response
                      for the frontend chart.

    Returns
    -------
    dict with keys: method, horizon, values, history, per_feeder,
                    horizon_minutes, unit, last_run
    Response shape is identical to the old per-feeder version.
    """
    # ── Single feeder ─────────────────────────────────────────────────────────
    if feeder and feeder.strip().lower() not in ("", "all"):
        uid = resolve_feeder(feeder)
        if uid is None:
            raise ValueError(
                f"No trained NHITS model for feeder {feeder!r}. "
                f"Known feeders: {list(FEEDER_ID_MAP)}"
            )
        fc   = _forecast_one(df, uid)
        hist = _prepare_history(df, uid).tail(history_points)

        return {
            "method":          f"NHITS Global (feeder: {feeder})",
            "horizon":         [pd.Timestamp(t).isoformat() for t in fc["ds"]],
            "values":          [round(float(v), 4) for v in fc["mw"]],
            "history": {
                "ds": [pd.Timestamp(t).isoformat() for t in hist["ds"]],
                "mw": [round(float(v), 4)          for v in hist["y"]],
            },
            "per_feeder":      {uid: [round(float(v), 4) for v in fc["mw"]]},
            "horizon_minutes": HORIZON * 3,
            "unit":            "MW",
            "last_run":        datetime.now(timezone.utc).isoformat(),
        }

    # ── All feeders — step-wise sum ───────────────────────────────────────────
    feeders_present = [
        (f, resolve_feeder(f))
        for f in df["feeder"].dropna().unique()
        if resolve_feeder(f) is not None
    ]
    if not feeders_present:
        raise ValueError(
            "No feeders with trained NHITS models found in the current selection."
        )

    per_feeder: dict  = {}
    fc_parts:   list  = []
    hist_parts: list  = []

    for f_name, uid in feeders_present:
        sub_df = df[df["feeder"] == f_name]
        try:
            fc = _forecast_one(sub_df, uid)
        except ValueError as exc:
            logger.warning("Skipping feeder %s (%s): %s", f_name, uid, exc)
            continue
        per_feeder[uid] = [round(float(v), 4) for v in fc["mw"]]
        fc_parts.append(fc.reset_index(drop=True))
        hist_parts.append(
            _prepare_history(sub_df, uid).tail(history_points)
            .set_index("ds")["y"].rename(uid)
        )

    if not fc_parts:
        raise ValueError(
            f"Insufficient history for all feeders. "
            f"Need ≥ {INPUT_SIZE} rows (≈ {INPUT_SIZE * 3 // 60} h) per feeder "
            "on the 3-min grid."
        )

    values  = pd.concat([p["mw"] for p in fc_parts], axis=1).sum(axis=1)
    horizon = max(fc_parts, key=len)["ds"]
    hist    = (
        pd.concat(hist_parts, axis=1)
          .interpolate(method="time")
          .sum(axis=1)
          .tail(history_points)
          .rename("y")
          .reset_index()
    )
    hist_tail  = df.tail(60).copy()   # df is the full substation df passed in
    hist_ds    = [dt.isoformat() for dt in hist_tail["datetime"]]
    hist_mw    = [round(float(v) / 1000.0, 4)          # same MW_DIVISOR as your values
              for v in hist_tail["active_load"].fillna(0).values]

    return {
        "method":          f"NHITS Global (sum of {len(fc_parts)} feeders)",
        "horizon":         [pd.Timestamp(t).isoformat() for t in horizon],
        "values":          [round(float(v), 4) for v in values],
        "history": {
            "ds": [pd.Timestamp(t).isoformat() for t in hist["ds"]],
            "mw": [round(float(v), 4)          for v in hist["y"]],
        },
        "per_feeder":      per_feeder,
        "horizon_minutes": HORIZON * 3,
        "unit":            "MW",
        "last_run":        datetime.now(timezone.utc).isoformat(),
    }