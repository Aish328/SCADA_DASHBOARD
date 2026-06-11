"""
train_patchtst.py
=================
Trains one independent PatchTST model per feeder.

Design notes
------------
*  PatchTST (NeuralForecast implementation) is a channel-independent,
   univariate model: EXOGENOUS_HIST / FUTR / STAT are all False.  This
   script therefore detects model capabilities at runtime and passes the
   exogenous feature lists ONLY when the configured model supports them
   (e.g. switch cfg.MODEL_NAME to "NHITS" or "TSMixerx" for a true
   multivariate run — no other code changes needed).
*  Evaluation uses rolling-origin cross-validation over a chronologically
   held-out test span (last 15% of each series), with the 15% before it
   used as the validation set for early stopping.  No shuffling anywhere.
*  Per feeder, the script saves:
       models/<MODEL_NAME>_<FEEDER>/        trained NeuralForecast bundle
       outputs/cv_<FEEDER>.csv              test-window forecasts (ds, cutoff, y, yhat)
       outputs/loss_<FEEDER>.json           training / validation loss trajectories

Run:
    python train_patchtst.py [--data path/to/data.csv]
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import warnings

import pandas as pd

import config as cfg
from data_preprocessing import chronological_split_sizes, load_and_preprocess, save_processed

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("train")
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)


# --------------------------------------------------------------------------- #
# Model factory with capability detection
# --------------------------------------------------------------------------- #
def make_model():
    """Instantiate the configured model; attach exogenous lists only if supported."""
    from neuralforecast.losses.pytorch import MAE
    import neuralforecast.models as nfm

    model_cls = getattr(nfm, cfg.MODEL_NAME)

    kwargs = dict(
        h=cfg.HORIZON,
        input_size=cfg.INPUT_SIZE,
        max_steps=cfg.MAX_STEPS,
        learning_rate=cfg.LEARNING_RATE,
        batch_size=cfg.BATCH_SIZE,
        windows_batch_size=cfg.BATCH_SIZE,
        loss=MAE(),
        random_seed=cfg.RANDOM_SEED,
        enable_progress_bar=False,
        logger=False,
        accelerator="auto",
    )
    if cfg.MODEL_NAME == "PatchTST":
        kwargs.update(cfg.PATCHTST_KWARGS)
    else:  # keep early stopping for alternative models too
        kwargs.update(early_stop_patience_steps=8,
                      val_check_steps=25,
                      scaler_type=None)   # series-level scaling handles it

    # ---- exogenous capability detection ----------------------------------
    if getattr(model_cls, "EXOGENOUS_HIST", False):
        kwargs["hist_exog_list"] = cfg.HIST_EXOG
    if getattr(model_cls, "EXOGENOUS_FUTR", False):
        kwargs["futr_exog_list"] = cfg.FUTR_EXOG
    if "hist_exog_list" not in kwargs and "futr_exog_list" not in kwargs:
        log.info("%s does not support exogenous features -> training univariate "
                 "(target only, RevIN-normalised). Engineered features are still "
                 "used for EDA / correlation analysis.", cfg.MODEL_NAME)
    else:
        log.info("%s supports exogenous features -> hist=%s futr=%s",
                 cfg.MODEL_NAME, kwargs.get("hist_exog_list"), kwargs.get("futr_exog_list"))

    return model_cls(**kwargs)


# --------------------------------------------------------------------------- #
# Per-feeder training + rolling-origin test evaluation
# --------------------------------------------------------------------------- #
def train_one_feeder(feeder: str, df: pd.DataFrame) -> pd.DataFrame:
    from neuralforecast import NeuralForecast

    model_id = cfg.FEEDERS[feeder]
    n = len(df)
    n_train, n_val, n_test = chronological_split_sizes(n)
    if n_train < cfg.INPUT_SIZE + cfg.HORIZON:
        raise ValueError(f"{feeder}: train split too short "
                         f"({n_train} < input_size + horizon = {cfg.INPUT_SIZE + cfg.HORIZON})")

    log.info("=== %s | n=%d  train=%d  val=%d  test=%d ===", model_id, n, n_train, n_val, n_test)

    model = make_model()
    used_cols = (["unique_id", "ds", "y"]
                 + list(getattr(model, "hist_exog_list", None) or [])
                 + list(getattr(model, "futr_exog_list", None) or []))
    df = df[[c for c in dict.fromkeys(used_cols)]]   # subset, preserve order, dedupe
    local_scaler = None if cfg.MODEL_NAME == "PatchTST" else cfg.LOCAL_SCALER_TYPE
    nf = NeuralForecast(models=[model], freq=cfg.EXPECTED_FREQ,
                        local_scaler_type=local_scaler)

    t0 = time.time()
    # Rolling-origin evaluation: fits on everything before the test span
    # (with `val_size` carved out for early stopping), then produces
    # horizon-length forecasts from every origin inside the test span.
    cv = nf.cross_validation(
        df=df,
        val_size=n_val,
        test_size=n_test,
        step_size=cfg.CV_STEP_SIZE,
        n_windows=None,
    )
    log.info("%s trained + evaluated in %.1fs (%d forecast windows)",
             model_id, time.time() - t0, cv["cutoff"].nunique())

    # ---- persist artefacts ------------------------------------------------
    cv = cv.reset_index(drop=False) if "unique_id" not in cv.columns else cv
    cv = cv.rename(columns={cfg.MODEL_NAME: "yhat"})
    cv.to_csv(cfg.RESULTS_DIR / f"cv_{model_id}.csv", index=False)

    model = nf.models[0]
    traj = {
        "train": [list(map(float, p)) for p in getattr(model, "train_trajectories", [])],
        "valid": [list(map(float, p)) for p in getattr(model, "valid_trajectories", [])],
    }
    (cfg.RESULTS_DIR / f"loss_{model_id}.json").write_text(json.dumps(traj))

    save_path = cfg.MODELS_DIR / model_id
    nf.save(path=str(save_path), overwrite=True, save_dataset=True)
    log.info("%s saved -> %s", model_id, save_path)
    return cv


def main(data_path: str) -> None:
    frames = load_and_preprocess(data_path)
    save_processed(frames)
    for feeder, df in frames.items():
        train_one_feeder(feeder, df)
    log.info("All %d feeder models trained. Run `python evaluate.py` next.", len(frames))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train per-feeder PatchTST models")
    ap.add_argument("--data", default=str(cfg.RAW_DATA_PATH), help="Path to raw data.csv")
    args = ap.parse_args()
    main(args.data)
