"""
per_feeder_forecast.py
=======================
Per-feeder NHITS inference using individual trained models.
Loads each feeder's dedicated model bundle from models/<MODEL_ID>/
Falls back to global model or linear extrapolation if per-feeder model unavailable.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
_DEFAULT_MODEL_DIR = (
    Path(__file__).resolve().parent / "models"
)
MODELS_DIR = Path(_DEFAULT_MODEL_DIR)

FREQ       = "3min"
INPUT_SIZE = 48     # must match training config.py
HORIZON    = 20     # must match training config.py

# Maps feeder display name → model_id (trained model folder name)
FEEDER_MODEL_MAP: dict[str, str] = {
    "11KV BHOSALE NAGAR":    "EP_FD01",
    "11KV KUBERA":           "EP_FD02",
    "RMU1":                  "ML_FD01",
    "11KV LUMAX":            "ML_FD02",
    "11KV MALWADI HADAPSAR": "ML_FD03",
}

# Extended mapping for data that may have these feeders
_ALL_KNOWN_MODELS = {
    "EP_FD01", "EP_FD02", "EP_FD03",
    "ML_FD01", "ML_FD02", "ML_FD03", "ML_FD04", "ML_FD05",
}

# ── Model cache (singleton per model ID) ───────────────────────────────────────
_model_cache: dict[str, object] = {}
_cache_lock = threading.Lock()


def _get_model(model_id: str) -> object:
    """Load and cache a NeuralForecast bundle by model ID."""
    with _cache_lock:
        if model_id in _model_cache:
            return _model_cache[model_id]
    
    try:
        from neuralforecast import NeuralForecast
        path = MODELS_DIR / model_id
        if not path.exists():
            logger.warning("Model not found at %s", path)
            return None
        logger.info("Loading per-feeder model %s from %s", model_id, path)
        nf = NeuralForecast.load(path=str(path))
        with _cache_lock:
            _model_cache[model_id] = nf
        return nf
    except Exception as e:
        logger.warning("Failed to load model %s: %s", model_id, e)
        return None


def _prepare_history(df_feeder: pd.DataFrame, model_id: str) -> pd.DataFrame:
    """
    Prepare history frame for per-feeder model inference.
    Mirrors data_preprocessing.py from the training pipeline.
    Includes historical and future exogenous features.
    """
    g = (df_feeder[["datetime", "active_load", "ir", "iy", "ib"]]
         .rename(columns={"datetime": "ds", "active_load": "y", "ir": "IR_A", "iy": "IY_A", "ib": "IB_A"})
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
        logger.info("%s: converting %d kW-scale rows → MW", model_id, int(kw_mask.sum()))
        g.loc[kw_mask, "y"] /= 1000.0

    # Compute average current (historical exogenous)
    g["avg_current"] = (g["IR_A"] + g["IY_A"] + g["IB_A"]) / 3.0
    
    # Add time-of-day features (future exogenous)
    tod = (g["ds"].dt.hour + g["ds"].dt.minute / 60.0) / 24.0
    g["tod_sin"] = np.sin(2 * np.pi * tod)
    g["tod_cos"] = np.cos(2 * np.pi * tod)

    g["unique_id"] = model_id
    
    # Return with all required columns
    cols = ["unique_id", "ds", "y", "IR_A", "IY_A", "IB_A", "avg_current", "tod_sin", "tod_cos"]
    return g[cols].dropna()


def _forecast_one_model(df_feeder: pd.DataFrame, model_id: str) -> Optional[pd.DataFrame]:
    """
    Run a single per-feeder model. Returns DataFrame[ds, mw] or None on failure.
    """
    nf = _get_model(model_id)
    if nf is None:
        logger.warning("No model available for %s", model_id)
        return None

    try:
        hist = _prepare_history(df_feeder, model_id)
        if len(hist) < INPUT_SIZE:
            logger.warning(
                "%s: need ≥ %d rows on the 3-min grid, have %d.",
                model_id, INPUT_SIZE, len(hist)
            )
            return None

        # Build future exogenous features (time-of-day)
        last_ds = hist["ds"].max()
        future_ds = pd.date_range(last_ds, periods=HORIZON + 1, freq=FREQ)[1:]
        futr = pd.DataFrame({"unique_id": model_id, "ds": future_ds})
        tod = (futr["ds"].dt.hour + futr["ds"].dt.minute / 60.0) / 24.0
        futr["tod_sin"] = np.sin(2 * np.pi * tod)
        futr["tod_cos"] = np.cos(2 * np.pi * tod)

        # Predict with future exogenous features
        fcst = nf.predict(df=hist, futr_df=futr)
        fcst = fcst.reset_index() if "unique_id" not in fcst.columns else fcst

        yhat = [c for c in fcst.columns if c not in ("unique_id", "ds")][0]
        return pd.DataFrame({
            "ds": pd.to_datetime(fcst["ds"]),
            "mw": fcst[yhat].astype(float).clip(lower=0.0),
        })
    except Exception as e:
        logger.warning("Forecast failed for %s: %s", model_id, e)
        return None


def run_per_feeder_forecast(
    df: pd.DataFrame,
    substation: Optional[str] = None,
    feeder: Optional[str] = None,
    history_points: int = INPUT_SIZE,
) -> dict:
    """
    Per-feeder model inference.
    
    If a single feeder is selected, use its dedicated model.
    If "all feeders" (feeder=None), sum forecasts from all available per-feeder models.
    
    Parameters
    ----------
    df              : Full DataLoader df (already filtered by substation if provided)
    substation      : Selected substation (informational)
    feeder          : Feeder display name, or None for all
    history_points  : Recent history steps to include
    
    Returns
    -------
    dict compatible with ml.html frontend
    """
    
    # ── Single feeder ──────────────────────────────────────────────────────────
    if feeder and feeder.strip().lower() not in ("", "all"):
        model_id = FEEDER_MODEL_MAP.get(feeder)
        if not model_id:
            logger.warning("No model mapping for feeder %r", feeder)
            return {"error": f"No trained model for feeder {feeder}"}
        
        df_feeder = df[df["feeder"] == feeder].copy() if "feeder" in df.columns else df
        fcst = _forecast_one_model(df_feeder, model_id)
        if fcst is None:
            return {"error": f"Could not forecast for {feeder} (model_id={model_id})"}
        
        hist = _prepare_history(df_feeder, model_id).tail(history_points)
        
        return {
            "method":          f"NHITS Per-Feeder ({model_id})",
            "model_id":        model_id,
            "horizon":         [pd.Timestamp(t).isoformat() for t in fcst["ds"]],
            "values":          [round(float(v), 4) for v in fcst["mw"]],
            "history": {
                "ds": [pd.Timestamp(t).isoformat() for t in hist["ds"]],
                "mw": [round(float(v), 4)          for v in hist["y"]],
            },
            "per_feeder":      {model_id: [round(float(v), 4) for v in fcst["mw"]]},
            "horizon_minutes": HORIZON * 3,
            "unit":            "MW",
            "last_run":        datetime.now(timezone.utc).isoformat(),
        }
    
    # ── All feeders — sum per-feeder forecasts ─────────────────────────────────
    feeders_present = df["feeder"].dropna().unique() if "feeder" in df.columns else []
    per_feeder: dict = {}
    fc_parts: list = []
    hist_parts: list = []
    
    for f_name in feeders_present:
        model_id = FEEDER_MODEL_MAP.get(f_name)
        if not model_id:
            logger.debug("No model for feeder %r, skipping", f_name)
            continue
        
        sub_df = df[df["feeder"] == f_name]
        fcst = _forecast_one_model(sub_df, model_id)
        if fcst is None:
            logger.debug("Could not forecast %s (%s), skipping", f_name, model_id)
            continue
        
        per_feeder[model_id] = [round(float(v), 4) for v in fcst["mw"]]
        fc_parts.append(fcst.reset_index(drop=True))
        hist_parts.append(
            _prepare_history(sub_df, model_id).tail(history_points)
            .set_index("ds")["y"].rename(model_id)
        )
    
    if not fc_parts:
        logger.warning("No per-feeder models available for forecast")
        return {"error": "No trained per-feeder models found"}
    
    values = pd.concat([p["mw"] for p in fc_parts], axis=1).sum(axis=1)
    horizon = max(fc_parts, key=len)["ds"]
    hist = (
        pd.concat(hist_parts, axis=1)
          .interpolate(method="time")
          .sum(axis=1)
          .tail(history_points)
          .rename("y")
          .reset_index()
    )
    
    return {
        "method":          f"NHITS Per-Feeder (sum of {len(fc_parts)} models)",
        "model_id":        f"MULTI ({', '.join(per_feeder.keys())})",
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
