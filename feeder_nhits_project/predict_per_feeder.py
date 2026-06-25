from pathlib import Path

import pandas as pd
import numpy as np

from neuralforecast import NeuralForecast

import config as cfg
from data_preprocessing import load_and_preprocess


def build_future_exog(uid, last_ds):

    future_ds = pd.date_range(
        last_ds,
        periods=cfg.HORIZON + 1,
        freq=cfg.EXPECTED_FREQ
    )[1:]

    futr = pd.DataFrame()

    futr["unique_id"] = uid
    futr["ds"] = future_ds

    tod = (
        futr["ds"].dt.hour +
        futr["ds"].dt.minute / 60
    ) / 24

    futr["tod_sin"] = np.sin(
        2 * np.pi * tod
    )

    futr["tod_cos"] = np.cos(
        2 * np.pi * tod
    )

    return futr


def forecast(uid):

    model_dir = cfg.MODELS_DIR / uid

    if not model_dir.exists():
        raise FileNotFoundError(
            f"Model not found: {uid}"
        )

    nf = NeuralForecast.load(
        path=str(model_dir)
    )

    df = load_and_preprocess()

    feeder_df = (
        df[df["unique_id"] == uid]
        .copy()
    )

    futr_df = build_future_exog(
        uid,
        feeder_df["ds"].max()
    )

    pred = nf.predict(
        df=feeder_df,
        futr_df=futr_df
    )

    output = Path("predictions")
    output.mkdir(exist_ok=True)

    pred.to_csv(
        output / f"{uid}_forecast.csv",
        index=False
    )

    print(pred.tail())


if __name__ == "__main__":

    forecast("EP_FD01")