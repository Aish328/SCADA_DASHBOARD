"""
evaluate.py
===========
Computes test metrics and produces all required visualisations from the
artefacts written by train_patchtst.py.

Per feeder
----------
* MAE, RMSE, MAPE, R^2 over every rolling test window
* Actual vs Forecast plot (non-overlapping horizon trajectories on the test span)
* Residual diagnostics (residuals over time + histogram)
* Feature correlation heatmap (target + exogenous/engineered features)
* Training / validation loss curves

Global
------
* outputs/comparison_table.csv — feeders ranked by forecasting performance
  (primary key: lowest MAPE; R^2 shown for variance explained).

Run:
    python evaluate.py
"""

from __future__ import annotations

import json
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import config as cfg

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("evaluate")

EPS = 1e-6  # MAPE guard against (near-)zero actuals


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def compute_metrics(y: np.ndarray, yhat: np.ndarray) -> dict[str, float]:
    mae = mean_absolute_error(y, yhat)
    rmse = float(np.sqrt(mean_squared_error(y, yhat)))
    mape = float(np.mean(np.abs((y - yhat) / np.clip(np.abs(y), EPS, None))) * 100.0)
    r2 = r2_score(y, yhat)
    return {"MAE (MW)": mae, "RMSE (MW)": rmse, "MAPE (%)": mape, "R2": r2}


# --------------------------------------------------------------------------- #
# Plot helpers
# --------------------------------------------------------------------------- #
def plot_actual_vs_forecast(cv: pd.DataFrame, model_id: str) -> None:
    """Stitch non-overlapping horizon trajectories across the test span."""
    cutoffs = sorted(cv["cutoff"].unique())
    selected = cutoffs[::cfg.HORIZON] if len(cutoffs) >= cfg.HORIZON else cutoffs[:1]
    fig, ax = plt.subplots(figsize=(13, 4.5))

    actual = cv.drop_duplicates("ds").sort_values("ds")
    ax.plot(actual["ds"], actual["y"], color="black", lw=1.4, label="Actual")
    for i, c in enumerate(selected):
        w = cv[cv["cutoff"] == c].sort_values("ds")
        ax.plot(w["ds"], w["yhat"], lw=1.4, alpha=0.9, color="tab:red",
                label=f"{cfg.MODEL_NAME} forecast" if i == 0 else None)
        ax.axvline(w["ds"].iloc[0], color="grey", lw=0.5, ls=":")
    win_min = cfg.HORIZON * 3
    ax.set_title(f"{model_id} — Actual vs Forecast (test span, "
                 f"{cfg.HORIZON}-step / {win_min}-min windows)")
    ax.set_xlabel("Time"); ax.set_ylabel("Active Load (MW)")
    ax.legend(); fig.autofmt_xdate(); fig.tight_layout()
    fig.savefig(cfg.PLOTS_DIR / f"{model_id}_actual_vs_forecast.png", dpi=150)
    plt.close(fig)


def plot_residuals(cv: pd.DataFrame, model_id: str) -> None:
    res = cv["y"].to_numpy() - cv["yhat"].to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].scatter(cv["ds"], res, s=4, alpha=0.35, color="tab:blue")
    axes[0].axhline(0, color="red", lw=1)
    axes[0].set_title(f"{model_id} — Residuals over test span")
    axes[0].set_xlabel("Time"); axes[0].set_ylabel("Residual (MW)")
    axes[1].hist(res, bins=40, color="tab:blue", alpha=0.8)
    axes[1].axvline(0, color="red", lw=1)
    axes[1].set_title("Residual distribution")
    axes[1].set_xlabel("Residual (MW)")
    fig.autofmt_xdate(); fig.tight_layout()
    fig.savefig(cfg.PLOTS_DIR / f"{model_id}_residuals.png", dpi=150)
    plt.close(fig)


def plot_correlation_heatmap(model_id: str) -> None:
    proc = pd.read_csv(cfg.PROCESSED_DIR / f"{model_id}_processed.csv")
    cols = ["y"] + list(dict.fromkeys(cfg.EDA_FEATURES + cfg.CALENDAR_FEATURES))
    corr = proc[cols].corr()
    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(cols)), cols, fontsize=8)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im, fraction=0.046)
    ax.set_title(f"{model_id} — Feature correlation (target = y, Active Load MW)")
    fig.tight_layout()
    fig.savefig(cfg.PLOTS_DIR / f"{model_id}_correlation_heatmap.png", dpi=150)
    plt.close(fig)


def plot_loss_curve(model_id: str) -> None:
    fp = cfg.RESULTS_DIR / f"loss_{model_id}.json"
    if not fp.exists():
        return
    traj = json.loads(fp.read_text())
    fig, ax = plt.subplots(figsize=(7, 4))
    if traj.get("train"):
        t = np.array(traj["train"]); ax.plot(t[:, 0], t[:, 1], label="train loss")
    if traj.get("valid"):
        v = np.array(traj["valid"]); ax.plot(v[:, 0], v[:, 1], marker="o", label="val loss")
    ax.set_title(f"{model_id} — Training loss curve (MAE, scaled space)")
    ax.set_xlabel("Step"); ax.set_ylabel("Loss"); ax.legend()
    fig.tight_layout()
    fig.savefig(cfg.PLOTS_DIR / f"{model_id}_loss_curve.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> pd.DataFrame:
    rows = []
    for feeder, model_id in cfg.FEEDERS.items():
        cv_path = cfg.RESULTS_DIR / f"cv_{model_id}.csv"
        if not cv_path.exists():
            log.warning("Missing %s — run train_patchtst.py first.", cv_path)
            continue
        cv = pd.read_csv(cv_path, parse_dates=["ds", "cutoff"])
        m = compute_metrics(cv["y"].to_numpy(), cv["yhat"].to_numpy())
        rows.append({"Feeder": feeder, "Model": model_id, **m,
                     "Test windows": cv["cutoff"].nunique()})
        plot_actual_vs_forecast(cv, model_id)
        plot_residuals(cv, model_id)
        plot_correlation_heatmap(model_id)
        plot_loss_curve(model_id)
        log.info("%-32s MAE=%.4f RMSE=%.4f MAPE=%.2f%% R2=%.4f",
                 model_id, m["MAE (MW)"], m["RMSE (MW)"], m["MAPE (%)"], m["R2"])

    table = pd.DataFrame(rows).sort_values("MAPE (%)").reset_index(drop=True)
    table.insert(0, "Rank", table.index + 1)
    table.to_csv(cfg.RESULTS_DIR / "comparison_table.csv", index=False)
    print("\n================ FEEDER FORECASTING PERFORMANCE (ranked by MAPE) ================")
    print(table.to_string(index=False,
                          float_format=lambda x: f"{x:.4f}"))
    return table


if __name__ == "__main__":
    main()
