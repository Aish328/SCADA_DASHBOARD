import json

import pandas as pd

from pathlib import Path

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

import config as cfg


def mape(y_true, y_pred):

    y_true = pd.Series(y_true)

    return (
        ((y_true - y_pred).abs())
        / y_true.replace(0, 1e-6)
    ).mean() * 100


rows = []

for model_dir in cfg.MODELS_DIR.iterdir():

    if not model_dir.is_dir():
        continue

    cv_file = model_dir / "cv_predictions.csv"

    if not cv_file.exists():
        continue

    cv = pd.read_csv(cv_file)

    pred_col = [
        c
        for c in cv.columns
        if c not in (
            "unique_id",
            "ds",
            "cutoff",
            "y",
        )
    ][0]

    mae = mean_absolute_error(
        cv["y"],
        cv[pred_col]
    )

    rmse = mean_squared_error(
        cv["y"],
        cv[pred_col],
        squared=False
    )

    r2 = r2_score(
        cv["y"],
        cv[pred_col]
    )

    rows.append({
        "feeder": model_dir.name,
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape(
            cv["y"],
            cv[pred_col]
        ),
        "R2": r2,
    })

metrics = pd.DataFrame(rows)

metrics.to_csv(
    "feeder_metrics.csv",
    index=False
)

print(metrics.sort_values("MAE"))