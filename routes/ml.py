"""
routes/ml.py
---------------------------------------------------------
ML API endpoints for the SCADA dashboard ML Intelligence page.

ml.html calls exactly these:
    GET  /ml/status                  -> {running, last_run, has_results}
    GET  /ml/anomalies?limit=500     -> autoencoder + classifier results
    GET  /ml/forecast                -> NHITS Active Load forecast (next 1 h)
    POST /ml/run                     -> trigger anomaly pipeline in background

Optional query params on /ml/anomalies and /ml/forecast:
    substation=<name>&feeder=<name>  (same semantics as /kpi/)

Results of the heavy anomaly pipeline are cached in-process; /ml/run
recomputes them in a background thread while /ml/status reports progress.
The NHITS forecast is fast (pre-trained bundles, no refit) and is served
fresh on each call.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Query

from services.ml_inference import run_anomaly_classification, run_load_forecast

logger = logging.getLogger(__name__)
router = APIRouter()

# --------------------------------------------------------------------------- #
# In-process state for the heavy anomaly pipeline
# --------------------------------------------------------------------------- #
_state = {
    "running": False,
    "last_run": None,
    "result": None,       # cached anomaly payload
    "error": None,
    "params": {"substation": None, "feeder": None, "limit": 2000},
}
_state_lock = threading.Lock()


def _run_anomaly_job(substation: Optional[str], feeder: Optional[str], limit: int):
    try:
        result = run_anomaly_classification(substation, feeder, limit)
        with _state_lock:
            _state["result"] = result
            _state["error"] = result.get("error")
            _state["last_run"] = datetime.utcnow().isoformat()
    except Exception as e:                                    # noqa: BLE001
        logger.exception("Anomaly pipeline failed")
        with _state_lock:
            _state["error"] = str(e)
    finally:
        with _state_lock:
            _state["running"] = False


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@router.get("/status")
def ml_status():
    with _state_lock:
        return {
            "running":     _state["running"],
            "last_run":    _state["last_run"],
            "has_results": _state["result"] is not None,
            "error":       _state["error"],
        }


@router.post("/run")
def ml_run(substation: Optional[str] = None,
           feeder: Optional[str] = None,
           limit: int = Query(2000, ge=100, le=20000)):
    with _state_lock:
        if _state["running"]:
            return {"status": "already_running"}
        _state["running"] = True
        _state["error"] = None
        _state["params"] = {"substation": substation, "feeder": feeder, "limit": limit}
    threading.Thread(
        target=_run_anomaly_job, args=(substation, feeder, limit), daemon=True
    ).start()
    return {"status": "started"}


@router.get("/anomalies")
def ml_anomalies(substation: Optional[str] = None,
                 feeder: Optional[str] = None,
                 limit: int = Query(500, ge=50, le=20000)):
    """Serve the cached run if present; compute synchronously otherwise."""
    with _state_lock:
        cached = _state["result"]
        cached_params = _state["params"]
    requested = {"substation": substation, "feeder": feeder}
    if cached is not None and all(
            cached_params.get(k) == v for k, v in requested.items()):
        return cached
    result = run_anomaly_classification(substation, feeder, limit)
    with _state_lock:
        _state["result"] = result
        _state["params"] = {**requested, "limit": limit}
        _state["last_run"] = datetime.utcnow().isoformat()
    return result


@router.get("/forecast")
def ml_forecast(substation: Optional[str] = None,
                feeder: Optional[str] = None):
    """NHITS Active Load forecast (next 1 hour). Fast — no caching needed."""
    return run_load_forecast(substation, feeder)