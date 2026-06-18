"""
services/ml_inference.py
---------------------------------------------------------
ML Inference Pipeline for SCADA Dashboard
  - LSTM Autoencoder  → anomaly detection gate
  - LSTM Classifier   → 6-class fault classification
  - NHITS (per-feeder)→ Active Load forecasting (1 hour ahead)
---------------------------------------------------------
"""

import threading
import logging
from datetime import datetime, timedelta
from typing import Optional
import traceback
import numpy as np
import pandas as pd
import torch

from services.data_loader import DataLoader

logger = logging.getLogger(__name__)

# =========================================================
# Fault Metadata (mirrors dataload.py)
# =========================================================

FAULT_CLASSES = {
    0: "Normal",
    1: "Voltage Sag",
    2: "Voltage Surge",
    3: "Current Imbalance",
    4: "Overload",
    5: "Transformer Stress",
}

FAULT_COLORS = {
    0: "#2ecc71",
    1: "#e74c3c",
    2: "#f39c12",
    3: "#3498db",
    4: "#9b59b6",
    5: "#c0392b",
}

# =========================================================
# Config
# =========================================================

AUTOENCODER_PATH = r"/home/sharika/Desktop/SCADA_DASHBOARD/lstm_model.pth"
CLASSIFIER_PATH  = r"/home/sharika/Desktop/SCADA_DASHBOARD/lstm_classifier.pth"
# NHITS bundles live in feeder_nhits_project/models (override: NHITS_MODELS_DIR env var)

SEQ_LEN       = 48
ANOMALY_PCT   = 95        # percentile threshold for autoencoder
CHUNK         = 256       # batch size for classifier inference

FEATURE_COLS = ["ir", "iy", "ib", "vry", "vyb", "vbr", "active_load", "hour", "day"]

# =========================================================
# Lazy Model Cache
# =========================================================

_autoencoder = None
_classifier  = None
_model_lock  = threading.Lock()


def _load_autoencoder(input_size: int):
    """Import here to avoid circular deps; model.py must be importable."""
    from model import LSTMModel
    m = LSTMModel(input_size=input_size)
    m.load_state_dict(torch.load(AUTOENCODER_PATH, weights_only=True))
    m.eval()
    return m


def _load_classifier(input_size: int):
    from classifier import LSTMClassifier
    m = LSTMClassifier(input_size=input_size)
    m.load_state_dict(torch.load(CLASSIFIER_PATH, weights_only=True))
    m.eval()
    return m


def _ensure_models(input_size: int):
    global _autoencoder, _classifier
    with _model_lock:
        if _autoencoder is None:
            logger.info("Loading autoencoder …")
            _autoencoder = _load_autoencoder(input_size)
        if _classifier is None:
            logger.info("Loading classifier …")
            _classifier = _load_classifier(input_size)


# =========================================================
# Feature Prep
# =========================================================

def _build_feature_matrix(df: pd.DataFrame) -> Optional[np.ndarray]:
    """
    Scale and assemble the 9-column feature matrix from raw sensor df.
    Returns None if df is too small.
    """
    df = df.copy()

    # Time features
    if "datetime" in df.columns:
        df["hour"] = df["datetime"].dt.hour
        df["day"]  = df["datetime"].dt.dayofweek

    available = [c for c in FEATURE_COLS if c in df.columns]
    if len(available) < 4:
        return None

    mat = df[available].copy()

    # Forward-fill then drop remaining NaNs
    mat = mat.ffill().dropna()

    # MinMax scale per column
    col_min = mat.min()
    col_max = mat.max()
    rng = (col_max - col_min).replace(0, 1)
    scaled = ((mat - col_min) / rng).values.astype(np.float32)

    return scaled


def _make_sequences(data: np.ndarray, seq_len: int = SEQ_LEN):
    X = []
    for i in range(len(data) - seq_len):
        X.append(data[i : i + seq_len])
    return np.array(X, dtype=np.float32)


# =========================================================
# Anomaly + Classification Pipeline
# =========================================================

def run_anomaly_classification(
    substation: Optional[str] = None,
    feeder: Optional[str] = None,
    limit: int = 2000,
) -> dict:
    """
    1. Pull latest rows from DataLoader
    2. Run autoencoder → flag anomalies
    3. Run classifier on anomalous sequences only
    4. Return structured result dict
    """
    df = DataLoader.get_all_data()
    df = DataLoader.filter_data(df, substation, feeder)
    df = df.sort_values("datetime").tail(limit)

    if len(df) < SEQ_LEN + 10:
        return {"error": "Not enough data", "sequences": [], "summary": {}}

    scaled = _build_feature_matrix(df)
    if scaled is None:
        return {"error": "Feature columns missing", "sequences": [], "summary": {}}

    input_size = scaled.shape[1]
    _ensure_models(input_size)

    X_np = _make_sequences(scaled, SEQ_LEN)
    if len(X_np) == 0:
        return {"error": "No sequences", "sequences": [], "summary": {}}

    X = torch.tensor(X_np, dtype=torch.float32)
    n_seq = len(X)

    # ── Autoencoder ────────────────────────────────────────
    with torch.no_grad():
        recon = _autoencoder(X)
    recon_err = torch.mean((X - recon) ** 2, dim=(1, 2)).numpy()
    threshold = np.percentile(recon_err, ANOMALY_PCT)
    is_anomaly = recon_err > threshold

    # ── Classifier (anomalous only) ────────────────────────
    anomaly_indices = np.where(is_anomaly)[0]
    anomaly_labels  = np.zeros(n_seq, dtype=int)

    if len(anomaly_indices) > 0:
        X_anom = X[anomaly_indices]
        all_preds = []
        for s in range(0, len(X_anom), CHUNK):
            batch = X_anom[s : s + CHUNK]
            with torch.no_grad():
                logits = _classifier(batch)
                probs  = torch.softmax(logits, dim=1)
                preds  = torch.argmax(probs[:, 1:], dim=1) + 1
            all_preds.append(preds.cpu().numpy())
        predicted = np.concatenate(all_preds)
        anomaly_labels[anomaly_indices] = predicted

    # ── Build time index for sequences ────────────────────
    datetimes = df["datetime"].reset_index(drop=True)
    seq_times = [
        datetimes.iloc[i + SEQ_LEN].isoformat()
        for i in range(n_seq)
    ]

    # ── Per-sequence output ────────────────────────────────
    sequences = [
        {
            "index":      int(i),
            "datetime":   seq_times[i],
            "recon_err":  round(float(recon_err[i]), 6),
            "is_anomaly": bool(is_anomaly[i]),
            "fault_id":   int(anomaly_labels[i]),
            "fault_name": FAULT_CLASSES[int(anomaly_labels[i])],
            "fault_color": FAULT_COLORS[int(anomaly_labels[i])],
        }
        for i in range(n_seq)
    ]

    # ── Summary counts ─────────────────────────────────────
    fault_counts = {}
    for cls_id, cls_name in FAULT_CLASSES.items():
        if cls_id == 0:
            continue
        count = int((anomaly_labels == cls_id).sum())
        if count:
            fault_counts[cls_name] = {
                "count": count,
                "color": FAULT_COLORS[cls_id],
                "pct":   round(count / max(len(anomaly_indices), 1) * 100, 1),
            }

    summary = {
        "total_sequences":    n_seq,
        "anomaly_count":      int(is_anomaly.sum()),
        "anomaly_pct":        round(float(is_anomaly.mean()) * 100, 1),
        "recon_threshold":    round(float(threshold), 6),
        "fault_counts":       fault_counts,
        "last_run":           datetime.utcnow().isoformat(),
    }

    return {
        "sequences": sequences,
        "summary":   summary,
        "recon_err": [round(float(e), 6) for e in recon_err],
        "threshold": round(float(threshold), 6),
        "fault_classes": FAULT_CLASSES,
        "fault_colors":  FAULT_COLORS,
    }


# =========================================================
# Load Forecast Pipeline
# =========================================================

def run_load_forecast(
    substation: Optional[str] = None,
    feeder: Optional[str] = None,
) -> dict:
    """
    Global NHITS forecast (next 1 hour, 20 × 3-min steps, MW).

    Single feeder selected  → that feeder's series from the global model.
    feeder is None / "All"  → per-feeder forecasts summed step-wise.

    DataLoader is queried WITHOUT feeder filter so the global model
    always receives data for all feeders (needed for "All" mode).
    The feeder parameter is forwarded to run_nhits_forecast() which
    does the per-feeder routing internally.

    Falls back to linear extrapolation if the NHITS bundle is not found.
    """
    # Load data: apply substation filter but NOT feeder filter here —
    # run_nhits_forecast() handles per-feeder selection internally.
    df = DataLoader.get_all_data()
    df = DataLoader.filter_data(df, substation, feeder=None)
    df = df.sort_values("datetime")

    if "active_load" not in df.columns or len(df) < 50:
        return {"error": "Insufficient data for forecast"}

    # ── Global NHITS (pre-trained single bundle, no refit) ──
    try:
        from services.nhits_forecast import run_nhits_forecast
        return run_nhits_forecast(
            df,
            substation=substation,
            feeder=feeder,          # None = all feeders summed
        )
    except FileNotFoundError as e:
        logger.error("NHITS bundle missing: %s", e)
        return {"error": str(e)}
    except Exception as e:
        logger.warning("NHITS forecast failed, falling back to linear: %s", e)
        logger.debug(traceback.format_exc())

    # ── Fallback: linear trend extrapolation ───────────────
    # Applied per-feeder if one is selected, else on the full df.
    target_df = df[df["feeder"] == feeder] if feeder else df.copy()
    target_df = target_df.sort_values("datetime")

    recent_raw    = target_df["active_load"].dropna().tail(288).values
    if len(recent_raw) < 2:
        return {"error": "Not enough data for linear fallback"}
    print(type(recent_raw))
    print(recent_raw)
    print(getattr(recent_raw, "shape", None))
    # if (recent_raw > 100.0).any():
    #     MW_DIVISOR = 1000.0
    # else:
    #     MW_DIVISOR = 1.0
    # recent_mw = recent_raw / MW_DIVISOR 
    recent_mw = recent_raw.astype(float)

    interval  = DataLoader.infer_interval()
    
    last_dt   = target_df["datetime"].max()
    hist_tail = target_df.tail(60)

    history_ds = [dt.isoformat() for dt in hist_tail["datetime"]]
    history_raw = hist_tail["active_load"].fillna(0).values
    history_mw = history_raw.astype(float)
    # history_mw  = np.where(history_raw > 100.0, history_raw / 1000.0, history_raw)
    history_mw = [round(float(v),4) for v in history_mw]
    x         = np.arange(len(recent_mw))
    coeffs    = np.polyfit(x, recent_mw, 1)
    trend_fn  = np.poly1d(coeffs)
    future_x  = np.arange(len(recent_mw), len(recent_mw) + 20)
    predicted = trend_fn(future_x)
        
    last_value = recent_mw[-1]
    predicted = np.clip(predicted,
                        0.5*last_value,
                        1.5*last_value) # no negative MW
    horizon_dts = [
        (last_dt + timedelta(minutes=interval * (i + 1))).isoformat()
        for i in range(20)
    ]
    return {
        "method":   "linear_extrapolation (NHITS unavailable)",
        "horizon":  horizon_dts,
        "values":   [round(float(v), 4) for v in predicted],
        "last_run": datetime.utcnow().isoformat(),
        "history": {
            "ds" : history_ds,
            "mw" : history_mw,
        },
        "forecast": {
        "ds": horizon_dts,
        "mw": [round(float(v),4) for v in predicted]
        },
        "error_analysis": [],
        "method": "NHITS",
        "per_feeder" : {},
        
    }