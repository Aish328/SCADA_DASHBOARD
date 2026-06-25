from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import pandas as pd

import config as cfg
from data_preprocessing import (
    load_and_preprocess,
    save_processed,
    chronological_split_sizes,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

log = logging.getLogger("train_per_feeder")


def build_model():

    from neuralforecast.models import NHITS
    from neuralforecast.losses.pytorch import MAE

    return NHITS(
        h=cfg.HORIZON,
        input_size=max(cfg.INPUT_SIZE, 96),

        hist_exog_list=cfg.HIST_EXOG,
        futr_exog_list=cfg.FUTR_EXOG,

        stack_types=['identity'] * 3,
        n_blocks=[1, 1, 1],
        mlp_units=[[256, 256]] * 3,
        n_pool_kernel_size=[2, 2, 1],
        n_freq_downsample=[4, 2, 1],

        dropout_prob_theta=0.1,
        activation="ReLU",
        interpolation_mode="linear",

        learning_rate=1e-4,
        batch_size=32,
        windows_batch_size=32,

        max_steps=1500,

        loss=MAE(),

        random_seed=cfg.RANDOM_SEED,

        accelerator="auto",

        enable_progress_bar=True,
        logger=False,

        early_stop_patience_steps=50,
    )


def train_single_feeder(feeder_df: pd.DataFrame, uid: str):

    from neuralforecast import NeuralForecast

    n = len(feeder_df)

    if n < 300:
        log.warning(
            "%s skipped (only %d rows)",
            uid,
            n
        )
        return

    train_size, val_size, test_size = chronological_split_sizes(n)

    model = build_model()

    nf = NeuralForecast(
        models=[model],
        freq=cfg.EXPECTED_FREQ,
        local_scaler_type="standard"
    )

    log.info(
        "Training %s | rows=%d train=%d val=%d test=%d",
        uid,
        n,
        train_size,
        val_size,
        test_size,
    )

    start = time.time()

    cv = nf.cross_validation(
        df=feeder_df,
        val_size=val_size,
        test_size=test_size,
        step_size=1,
        n_windows=None,
    )

    elapsed = time.time() - start

    model_dir = cfg.MODELS_DIR / uid
    model_dir.mkdir(parents=True, exist_ok=True)

    nf.save(
        path=str(model_dir),
        overwrite=True,
        save_dataset=True,
    )

    cv.to_csv(
        model_dir / "cv_predictions.csv",
        index=False
    )

    metrics = {
        "uid": uid,
        "rows": int(n),
        "train_rows": int(train_size),
        "val_rows": int(val_size),
        "test_rows": int(test_size),
        "training_seconds": round(elapsed, 2),
    }

    with open(model_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    log.info("%s saved -> %s", uid, model_dir)


def main():

    df = load_and_preprocess()

    save_processed(df)

    feeders = sorted(df["unique_id"].unique())

    for uid in feeders:

        feeder_df = (
            df[df["unique_id"] == uid]
            .copy()
            .reset_index(drop=True)
        )

        train_single_feeder(
            feeder_df,
            uid
        )

    log.info("All feeder models completed")


if __name__ == "__main__":
    main()