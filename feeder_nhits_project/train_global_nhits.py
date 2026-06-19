"""
train_global_nhits.py
=====================
Trains ONE global NHITS model on ALL feeders simultaneously.
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


def make_nhits_model():
    """
    Build the NHITS model with exogenous feature lists.
    Architecture parameters go directly to NHITS constructor.
    Trainer parameters go via **trainer_kwargs.
    """
    from neuralforecast.losses.pytorch import MAE
    from neuralforecast.models import NHITS

    model = NHITS(
        # --- Core parameters ---
        h                    = cfg.HORIZON,
        input_size           = cfg.INPUT_SIZE,
        hist_exog_list       = cfg.HIST_EXOG,
        futr_exog_list       = cfg.FUTR_EXOG,
        
        # --- Architecture parameters (direct NHITS params) ---
        stack_types          = ['identity'] * cfg.NHITS_ARCH_KWARGS.get('n_stacks', 3),
        n_blocks             = cfg.NHITS_ARCH_KWARGS.get('n_blocks', [1, 1, 1]),
        mlp_units            = cfg.NHITS_ARCH_KWARGS.get('mlp_units', [[256, 256]] * 3),
        n_pool_kernel_size   = cfg.NHITS_ARCH_KWARGS.get('n_pool_kernel_size', [2, 2, 1]),
        n_freq_downsample    = cfg.NHITS_ARCH_KWARGS.get('n_freq_downsample', [4, 2, 1]),
        dropout_prob_theta   = cfg.NHITS_ARCH_KWARGS.get('dropout_prob_theta', 0.1),
        activation           = cfg.NHITS_ARCH_KWARGS.get('activation', 'ReLU'),
        interpolation_mode   = cfg.NHITS_ARCH_KWARGS.get('interpolation_mode', 'linear'),
        
        # --- Training parameters (these become trainer_kwargs internally) ---
        max_steps            = cfg.MAX_STEPS,
        learning_rate        = cfg.LEARNING_RATE,
        batch_size           = cfg.BATCH_SIZE,
        windows_batch_size   = cfg.BATCH_SIZE,
        loss                 = MAE(),
        random_seed          = cfg.RANDOM_SEED,
        enable_progress_bar  = True,
        logger               = False,
        accelerator          = "auto",
        
        # --- Trainer kwargs (only PL Trainer compatible args) ---
        gradient_clip_val    = cfg.NHITS_TRAINER_KWARGS.get('gradient_clip_val', 1.0),
        val_check_steps      = cfg.NHITS_TRAINER_KWARGS.get('val_check_steps', 10),
        # scaler_type is handled by NeuralForecast's local_scaler_type, not here
    )
    
    log.info("NHITS model: hist_exog=%s  futr_exog=%s  stacks=%d",
             cfg.HIST_EXOG, cfg.FUTR_EXOG, 
             cfg.NHITS_ARCH_KWARGS.get('n_stacks', 3))
    return model


def _global_split_sizes(df: pd.DataFrame) -> tuple[int, int]:
    sizes = [len(g) for _, g in df.groupby("unique_id")]
    min_n = min(sizes)
    _, n_val, n_test = chronological_split_sizes(min_n)
    log.info("Split sizes based on shortest feeder (n=%d): val=%d  test=%d",
             min_n, n_val, n_test)
    return n_val, n_test


def train_global_model(df: pd.DataFrame) -> pd.DataFrame:
    from neuralforecast import NeuralForecast

    used_cols = (["unique_id", "ds", "y"]
                 + list(dict.fromkeys(cfg.HIST_EXOG + cfg.FUTR_EXOG)))
    df_model = df[used_cols].dropna()

    n_val, n_test = _global_split_sizes(df_model)

    model = make_nhits_model()

    nf = NeuralForecast(
        models=[model],
        freq=cfg.EXPECTED_FREQ,
        local_scaler_type=cfg.LOCAL_SCALER_TYPE,
    )

    log.info("Starting global cross-validation  (%d series, val=%d, test=%d) …",
             df_model["unique_id"].nunique(), n_val, n_test)
    t0 = time.time()

    cv = nf.cross_validation(
        df=df_model,
        val_size=n_val,
        test_size=n_test,
        step_size=cfg.CV_STEP_SIZE,
        n_windows=None,
    )
    elapsed = time.time() - t0
    cv = cv.reset_index(drop=False) if "unique_id" not in cv.columns else cv

    yhat_col = [c for c in cv.columns if c not in ("unique_id", "ds", "cutoff", "y")][0]
    cv = cv.rename(columns={yhat_col: "yhat"})

    log.info("Global training complete in %.1f s", elapsed)

    cv.to_csv(cfg.RESULTS_DIR / "cv_global.csv", index=False)

    traj = {
        "train": [list(map(float, p))
                  for p in getattr(nf.models[0], "train_trajectories", [])],
        "valid": [list(map(float, p))
                  for p in getattr(nf.models[0], "valid_trajectories", [])],
    }
    (cfg.RESULTS_DIR / "loss_global.json").write_text(json.dumps(traj))

    nf.save(
        path=str(cfg.GLOBAL_MODEL_DIR),
        overwrite=True,
        save_dataset=True,
    )
    log.info("Global NHITS bundle saved → %s", cfg.GLOBAL_MODEL_DIR)
    return cv


def main(
    start_dt: str | None = None,
    end_dt:   str | None = None,
    limit:    int | None = None,
) -> None:
    df = load_and_preprocess(start_dt=start_dt, end_dt=end_dt, limit=limit)
    save_processed(df)

    cv = train_global_model(df)

    from sklearn.metrics import mean_absolute_error
    print("\n============= GLOBAL MODEL — PER-FEEDER TEST MAE (MW) =============")
    for uid, grp in cv.groupby("unique_id"):
        name = cfg.ID_FEEDER_MAP.get(uid, uid)
        mae  = mean_absolute_error(grp["y"], grp["yhat"])
        print(f"  {uid:8s}  {name:28s}  MAE = {mae:.4f} MW")
    print()
    log.info("Training complete.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train global NHITS model")
    ap.add_argument("--start", default=None, help="Start datetime ISO")
    ap.add_argument("--end", default=None, help="End datetime ISO")
    ap.add_argument("--limit", default=None, type=int, help="Row cap")
    args = ap.parse_args()
    main(start_dt=args.start, end_dt=args.end, limit=args.limit)