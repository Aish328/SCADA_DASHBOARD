"""
train_global_nhits.py
=====================
Trains ONE global NHITS model on ALL feeders simultaneously.

Architecture rationale
----------------------
NeuralForecast handles multi-series datasets natively via `unique_id`.
The global model:
  • Sees patterns from all feeders in every training batch (cross-series
    learning: e.g., weekday morning peak is learned once, not 5 times).
  • Uses LOCAL_SCALER_TYPE="robust" to normalise each feeder's series
    independently before feeding into shared weights — feeders with
    different load magnitudes (e.g. RMU1 vs 11KV KUBERA) train stably.
  • At inference time, filtering to a single unique_id automatically
    applies that series' own scaler for inversion.

Adding a new feeder
-------------------
  1. Add the feeder name → unique_id mapping to cfg.FEEDER_ID_MAP.
  2. Re-run this script with the extended dataset.
  3. The model learns the new feeder alongside existing ones.
  NO separate model file needed — one bundle serves all.

Output artefacts
----------------
  models/NHITS_GLOBAL/          NeuralForecast bundle (load with NeuralForecast.load)
  outputs/cv_global.csv         Rolling-origin test forecasts for all feeders
  outputs/loss_global.json      Train / validation loss trajectories

Run:
    python train_global_nhits.py [--start 2026-01-01] [--end 2026-05-31]
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import warnings

import pandas as pd

import config as cfg
from data_preprocessing import (
    chronological_split_sizes,
    load_and_preprocess,
    save_processed,
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("train_global")
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)


# --------------------------------------------------------------------------- #
# Model factory
# --------------------------------------------------------------------------- #
def make_nhits_model():
    """
    Build the NHITS model with exogenous feature lists.

    NHITS supports:
      EXOGENOUS_HIST = True  →  hist_exog_list (past-only channels)
      EXOGENOUS_FUTR = True  →  futr_exog_list (known-future channels)

    We use:
      hist_exog : IR, IY, IB, avg_current  (metered at every past step)
      futr_exog : tod_sin, tod_cos         (cyclical calendar, known for future)
    """
    from neuralforecast.losses.pytorch import MAE
    from neuralforecast.models import NHITS

    model = NHITS(
        h                    = cfg.HORIZON,
        input_size           = cfg.INPUT_SIZE,
        hist_exog_list       = cfg.HIST_EXOG,     # past-only exog
        futr_exog_list       = cfg.FUTR_EXOG,     # future-known exog
        max_steps            = cfg.MAX_STEPS,
        learning_rate        = cfg.LEARNING_RATE,
        batch_size           = cfg.BATCH_SIZE,
        windows_batch_size   = cfg.BATCH_SIZE,
        loss                 = MAE(),
        random_seed          = cfg.RANDOM_SEED,
        enable_progress_bar  = True,
        logger               = False,
        accelerator          = "auto",
        **cfg.NHITS_TRAINER_KWARGS,
    )
    log.info("NHITS model: hist_exog=%s  futr_exog=%s",
             cfg.HIST_EXOG, cfg.FUTR_EXOG)
    return model


# --------------------------------------------------------------------------- #
# Compute test/val sizes  (use the SMALLEST feeder so the split is valid
# for every series; NeuralForecast uses the same n_val/n_test across all)
# --------------------------------------------------------------------------- #
def _global_split_sizes(df: pd.DataFrame) -> tuple[int, int]:
    """
    Return (n_val, n_test) computed from the shortest feeder series.
    NeuralForecast's cross_validation uses a single val_size / test_size
    applied to ALL series, so we must be conservative (use the minimum).
    """
    sizes = [len(g) for _, g in df.groupby("unique_id")]
    min_n         = min(sizes)
    _, n_val, n_test = chronological_split_sizes(min_n)
    log.info("Split sizes based on shortest feeder (n=%d): val=%d  test=%d",
             min_n, n_val, n_test)
    return n_val, n_test


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train_global_model(df: pd.DataFrame) -> pd.DataFrame:
    """
    Train one global NHITS on the combined multi-series frame.

    Parameters
    ----------
    df : pd.DataFrame
        Output of load_and_preprocess() — all feeders in one frame,
        columns: unique_id | ds | y | [hist_exog] | [futr_exog] | …

    Returns
    -------
    pd.DataFrame
        Cross-validation result frame with columns:
            unique_id | ds | cutoff | y | yhat
    """
    from neuralforecast import NeuralForecast

    # Subset to model input columns only (superset frame from preprocessing
    # includes EDA features that should not be fed into the model).
    used_cols = (["unique_id", "ds", "y"]
                 + list(dict.fromkeys(cfg.HIST_EXOG + cfg.FUTR_EXOG)))
    df_model = df[used_cols].dropna()

    n_val, n_test = _global_split_sizes(df_model)

    model = make_nhits_model()

    # LOCAL_SCALER_TYPE="robust" applies an independent per-series robust
    # scaler managed by NeuralForecast, so feeders with different load
    # magnitudes are handled automatically.
    nf = NeuralForecast(
        models            = [model],
        freq              = cfg.EXPECTED_FREQ,
        local_scaler_type = cfg.LOCAL_SCALER_TYPE,
    )

    log.info("Starting global cross-validation  (%d series, val=%d, test=%d) …",
             df_model["unique_id"].nunique(), n_val, n_test)
    t0 = time.time()

    # Rolling-origin cross-validation:
    #   • Fits on all data BEFORE the test span (with val_size carved out
    #     from the end of that window for early stopping).
    #   • Produces HORIZON-step forecasts from every origin in the test span.
    #   • step_size=1  →  every timestamp becomes a forecast origin
    #     (most comprehensive evaluation; slow — increase for faster runs).
    cv = nf.cross_validation(
        df        = df_model,
        val_size  = n_val,
        test_size = n_test,
        step_size = cfg.CV_STEP_SIZE,
        n_windows = None,
    )
    elapsed = time.time() - t0
    cv = cv.reset_index(drop=False) if "unique_id" not in cv.columns else cv

    # NeuralForecast names the forecast column after the model class
    yhat_col = [c for c in cv.columns if c not in ("unique_id", "ds", "cutoff", "y")][0]
    cv = cv.rename(columns={yhat_col: "yhat"})

    log.info("Global training complete in %.1f s  |  %d forecast windows across %d feeders",
             elapsed, cv["cutoff"].nunique(), cv["unique_id"].nunique())

    # ---- persist artefacts -------------------------------------------------
    cv.to_csv(cfg.RESULTS_DIR / "cv_global.csv", index=False)
    log.info("Cross-validation results → %s/cv_global.csv", cfg.RESULTS_DIR)

    traj = {
        "train": [list(map(float, p))
                  for p in getattr(nf.models[0], "train_trajectories", [])],
        "valid": [list(map(float, p))
                  for p in getattr(nf.models[0], "valid_trajectories", [])],
    }
    (cfg.RESULTS_DIR / "loss_global.json").write_text(json.dumps(traj))

    # ---- save global model bundle ------------------------------------------
    nf.save(
        path         = str(cfg.GLOBAL_MODEL_DIR),
        overwrite    = True,
        save_dataset = True,    # bundles the scaler state for correct inverse-transform
    )
    log.info("Global NHITS bundle saved → %s", cfg.GLOBAL_MODEL_DIR)
    return cv


# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #
def main(
    start_dt: str | None = None,
    end_dt:   str | None = None,
    limit:    int | None = None,
) -> None:
    """
    Parameters
    ----------
    start_dt / end_dt : ISO datetime strings to restrict the training window.
                        Default: pull all available rows from the DB.
    limit             : Cap rows (useful for quick smoke-tests).
    """
    df = load_and_preprocess(start_dt=start_dt, end_dt=end_dt, limit=limit)
    save_processed(df)

    cv = train_global_model(df)

    # Quick per-feeder MAE preview after training
    from sklearn.metrics import mean_absolute_error
    print("\n============= GLOBAL MODEL — PER-FEEDER TEST MAE (MW) =============")
    for uid, grp in cv.groupby("unique_id"):
        name = cfg.ID_FEEDER_MAP.get(uid, uid)
        mae  = mean_absolute_error(grp["y"], grp["yhat"])
        print(f"  {uid:8s}  {name:28s}  MAE = {mae:.4f} MW")
    print()
    log.info("Training complete. Run `python evaluate_global.py` for full metrics.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Train global NHITS model — data loaded from PostgreSQL"
    )
    ap.add_argument("--start",  default=None, help="Start datetime ISO e.g. 2026-01-01 00:00:00")
    ap.add_argument("--end",    default=None, help="End datetime ISO")
    ap.add_argument("--limit",  default=None, type=int, help="Row cap for testing")
    args = ap.parse_args()
    main(start_dt=args.start, end_dt=args.end, limit=args.limit)