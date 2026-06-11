"""
services/ml_scheduler.py
---------------------------------------------------------
APScheduler background jobs
  - Runs anomaly detection + load forecast every N minutes
  - Stores results in an in-memory cache shared via ml_cache
---------------------------------------------------------
"""

import logging
import threading
from datetime import datetime
from typing import Optional

from apscheduler.schedulers.background import BackgroundScheduler

logger = logging.getLogger(__name__)

# =========================================================
# Shared Result Cache
# =========================================================

_cache_lock = threading.Lock()

_ml_cache: dict = {
    "anomaly": None,   # result from run_anomaly_classification
    "forecast": None,  # result from run_load_forecast
    "last_run": None,
    "running":  False,
    "error":    None,
}


def get_cached_anomaly() -> Optional[dict]:
    with _cache_lock:
        return _ml_cache["anomaly"]


def get_cached_forecast() -> Optional[dict]:
    with _cache_lock:
        return _ml_cache["forecast"]


def get_cache_status() -> dict:
    with _cache_lock:
        return {
            "last_run":      _ml_cache["last_run"],
            "running":       _ml_cache["running"],
            "error":         _ml_cache["error"],
            "has_anomaly":   _ml_cache["anomaly"] is not None,
            "has_forecast":  _ml_cache["forecast"] is not None,
        }


# =========================================================
# Inference Job
# =========================================================

def _run_inference_job():
    """Called by scheduler on every tick."""
    from services.ml_inference import run_anomaly_classification, run_load_forecast

    with _cache_lock:
        if _ml_cache["running"]:
            logger.info("ML job already running, skipping tick.")
            return
        _ml_cache["running"] = True
        _ml_cache["error"]   = None

    logger.info("ML inference job started …")

    try:
        anomaly_result  = run_anomaly_classification()
        forecast_result = run_load_forecast()

        with _cache_lock:
            _ml_cache["anomaly"]  = anomaly_result
            _ml_cache["forecast"] = forecast_result
            _ml_cache["last_run"] = datetime.utcnow().isoformat()

        logger.info("ML inference job complete.")

    except Exception as e:
        logger.error(f"ML inference job failed: {e}", exc_info=True)
        with _cache_lock:
            _ml_cache["error"] = str(e)

    finally:
        with _cache_lock:
            _ml_cache["running"] = False


# =========================================================
# Scheduler Setup
# =========================================================

_scheduler: Optional[BackgroundScheduler] = None


def start_scheduler(interval_minutes: int = 15):
    """
    Start the APScheduler background scheduler.
    Call this once from main.py on startup.
    """
    global _scheduler

    if _scheduler is not None and _scheduler.running:
        logger.warning("Scheduler already running.")
        return

    _scheduler = BackgroundScheduler(
        job_defaults={"misfire_grace_time": 60}
    )

    _scheduler.add_job(
        _run_inference_job,
        trigger="interval",
        minutes=interval_minutes,
        id="ml_inference",
        name="ML Anomaly + Forecast",
        next_run_time=datetime.now(),   # run immediately on startup
    )

    _scheduler.start()
    logger.info(
        f"ML scheduler started — interval: {interval_minutes} min"
    )


def stop_scheduler():
    global _scheduler
    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=False)
        logger.info("ML scheduler stopped.")


def trigger_now():
    """Manually trigger an immediate inference run (for API endpoint)."""
    import threading
    t = threading.Thread(target=_run_inference_job, daemon=True)
    t.start()