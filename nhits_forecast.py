"""
services/nhits_forecast.py
---------------------------------------------------------
NHITS per-feeder Active Load forecasting for the SCADA dashboard.

Serves the pre-trained NeuralForecast NHITS bundles produced by
feeder_nhits_project/train_nhits.py — NO refitting at request time
(unlike the old pickled-PatchTST flow). Each bundle forecasts the
next 20 x 3-min steps (1 hour) of Active Load in MW.

Input data comes from services.data_loader.DataLoader rows with the
dashboard's lowercase schema:
    datetime, substation, feeder, ir, iy, ib, vry, vyb, vbr, active_load
This module mirrors the training-time preprocessing (3-min grid,
interpolation, avg_current, cyclical time-of-day) on those columns.

Configuration
-------------
NHITS_MODELS_DIR   env var pointing at the trained `models/` folder.
                   Default: <repo>/feeder_nhits_project/models
"""

from __future__ import annotations

import logging
import os
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Configuration (kept aligned with feeder_nhits_project/config.py)
# --------------------------------------------------------------------------- #
MODELS_DIR = Path(os.getenv(
    "NHITS_MODELS_DIR",
    Path(__file__).resolve().parent.parent / "feeder_nhits_project" / "models",
))

FREQ = "3min"
INPUT_SIZE = 96            # 4.8 h of history required
HORIZON = 20               # 1 hour ahead

HIST_EXOG = ["IR", "IY", "IB", "avg_current"]
FUTR_EXOG = ["tod_sin", "tod_cos"]

# Trained feeder -> model directory name (same sanitisation as training).
FEEDER_MODELS = {
    "11KV BHOSALE NAGAR":    "NHITS_11KV_BHOSALENAGAR",
    "11KV KUBERA":           "NHITS_11KV_KUBERA",
    "11KV MALWADI HADAPSAR": "NHITS_11KV_MALWADI_HADAPSAR",
    "RMU1":                  "NHITS_RMU1",
    "11KV LUMAX":            "NHITS_11KV_LUMAX",
}

_models: dict[str, "object"] = {}
_lock = threading.Lock()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _norm(name: str) -> str:
    """Normalise feeder names for matching: uppercase, collapse whitespace."""
    return re.sub(r"\s+", " ", str(name).strip().upper())


_NORM_LOOKUP = {_norm(k): k for k in FEEDER_MODELS}


def resolve_feeder(name: str) -> Optional[str]:
    """Map a dashboard feeder string onto a trained feeder key (or None)."""
    return _NORM_LOOKUP.get(_norm(name))


def _get_model(feeder_key: str):
    """Lazy, thread-safe load of one NeuralForecast bundle."""
    model_id = FEEDER_MODELS[feeder_key]
    with _lock:
        if model_id not in _models:
            from neuralforecast import NeuralForecast
            path = MODELS_DIR / model_id
            if not path.exists():
                raise FileNotFoundError(f"NHITS bundle missing: {path}")
            logger.info("Loading %s ...", model_id)
            _models[model_id] = NeuralForecast.load(path=str(path))
        return _models[model_id], model_id


def _prepare_frame(df: pd.DataFrame, model_id: str) -> pd.DataFrame:
    """
    Dashboard rows (one feeder) -> NeuralForecast input frame.
    Mirrors training preprocessing: 3-min grid, time interpolation,
    avg_current, cyclical time-of-day.
    """
    g = (df[["datetime", "active_load", "ir", "iy", "ib"]]
         .rename(columns={"datetime": "ds", "active_load": "y",
                          "ir": "IR", "iy": "IY", "ib": "IB"})
         .copy())
    g["ds"] = pd.to_datetime(g["ds"])
    g = (g.dropna(subset=["ds"])
          .sort_values("ds")
          .drop_duplicates(subset="ds", keep="last")
          .set_index("ds")
          .resample(FREQ).mean()
          .interpolate(method="time", limit_direction="both")
          .reset_index())

    g["avg_current"] = (g["IR"] + g["IY"] + g["IB"]) / 3.0
    tod = (g["ds"].dt.hour + g["ds"].dt.minute / 60.0) / 24.0
    g["tod_sin"] = np.sin(2 * np.pi * tod)
    g["tod_cos"] = np.cos(2 * np.pi * tod)
    g["unique_id"] = model_id
    return g[["unique_id", "ds", "y"] + HIST_EXOG + FUTR_EXOG].dropna()


def _future_exog(model_id: str, last_ds: pd.Timestamp) -> pd.DataFrame:
    future_ds = pd.date_range(last_ds, periods=HORIZON + 1, freq=FREQ)[1:]
    f = pd.DataFrame({"unique_id": model_id, "ds": future_ds})
    tod = (f["ds"].dt.hour + f["ds"].dt.minute / 60.0) / 24.0
    f["tod_sin"] = np.sin(2 * np.pi * tod)
    f["tod_cos"] = np.cos(2 * np.pi * tod)
    return f


def forecast_one_feeder(df_feeder: pd.DataFrame, feeder_key: str) -> pd.DataFrame:
    """Forecast next HORIZON steps for one feeder. Returns columns [ds, mw]."""
    nf, model_id = _get_model(feeder_key)
    frame = _prepare_frame(df_feeder, model_id)
    if len(frame) < INPUT_SIZE:
        raise ValueError(f"{feeder_key}: need >= {INPUT_SIZE} samples on the "
                         f"3-min grid, have {len(frame)}")
    futr = _future_exog(model_id, frame["ds"].max())
    fcst = nf.predict(df=frame, futr_df=futr)
    fcst = fcst.reset_index() if "unique_id" not in fcst.columns else fcst
    yhat = [c for c in fcst.columns if c not in ("unique_id", "ds")][0]
    return pd.DataFrame({"ds": pd.to_datetime(fcst["ds"]),
                         "mw": fcst[yhat].astype(float)})


# --------------------------------------------------------------------------- #
# Public entry point — payload matches ml.html's /ml/forecast contract
# --------------------------------------------------------------------------- #
def run_nhits_forecast(df: pd.DataFrame,
                       substation: Optional[str] = None,
                       feeder: Optional[str] = None,
                       history_points: int = 96) -> dict:
    """
    df: pre-filtered DataLoader rows (the caller applies substation/feeder
        filters; `feeder` here only tells us whether one feeder is selected).

    Single feeder  -> that feeder's NHITS forecast.
    No feeder (All)-> per-feeder forecasts, summed step-wise into a total.
    """
    if feeder:
        key = resolve_feeder(feeder)
        if key is None:
            raise ValueError(f"No trained NHITS model for feeder {feeder!r}. "
                             f"Trained: {list(FEEDER_MODELS)}")
        fc = forecast_one_feeder(df, key)
        hist = _prepare_frame(df, FEEDER_MODELS[key]).tail(history_points)
        per_feeder = {key: [round(v, 4) for v in fc["mw"]]}
        horizon, values = fc["ds"], fc["mw"]
        method = "NHITS"
    else:
        feeders_present = [f for f in df["feeder"].dropna().unique()
                           if resolve_feeder(f)]
        if not feeders_present:
            raise ValueError("No feeders with trained NHITS models in selection.")
        per_feeder, parts, hist_parts = {}, [], []
        for f in feeders_present:
            key = resolve_feeder(f)
            sub_df = df[df["feeder"] == f]
            try:
                fc = forecast_one_feeder(sub_df, key)
            except ValueError as e:           # too little history for this feeder
                logger.warning("Skipping %s: %s", f, e)
                continue
            per_feeder[key] = [round(v, 4) for v in fc["mw"]]
            parts.append(fc.reset_index(drop=True))
            hist_parts.append(
                _prepare_frame(sub_df, key).tail(history_points)
                .set_index("ds")["y"])
        if not parts:
            raise ValueError("Insufficient history for every feeder in selection.")
        # Step-wise sum (timestamps can differ by a step or two across feeders).
        values = pd.concat([p["mw"] for p in parts], axis=1).sum(axis=1)
        horizon = max(parts, key=len)["ds"]
        hist = (pd.concat(hist_parts, axis=1).interpolate(method="time")
                  .sum(axis=1).tail(history_points)
                  .rename("y").reset_index())
        method = f"NHITS (sum of {len(parts)} feeders)"

    return {
        "method":   method,
        "horizon":  [pd.Timestamp(t).isoformat() for t in horizon],
        "values":   [round(float(v), 4) for v in values],
        # extras (frontend uses these when present; harmless otherwise)
        "history": {
            "ds": [pd.Timestamp(t).isoformat() for t in hist["ds"]],
            "mw": [round(float(v), 4) for v in hist["y"]],
        },
        "per_feeder": per_feeder,
        "horizon_minutes": HORIZON * 3,
        "unit": "MW",
        "last_run": datetime.utcnow().isoformat(),
    }