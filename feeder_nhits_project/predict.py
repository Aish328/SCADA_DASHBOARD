"""
predict.py
==========
Inference: load a saved per-feeder PatchTST bundle and forecast the next
HORIZON steps (24 x 3 min ≈ 72 minutes) of Active Load beyond the latest
timestamp available for that feeder.

Usage
-----
    # Forecast a single feeder from the latest SCADA export
    python predict.py --feeder "11KV KUBERA" --data data.csv

    # Forecast every feeder
    python predict.py --all --data data.csv

Outputs
-------
    outputs/forecast_<MODEL_ID>.csv      columns: unique_id, ds, forecast_MW
    plots/<MODEL_ID>_future_forecast.png recent history + forecast plot
"""

from __future__ import annotations

import argparse
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import config as cfg
from data_preprocessing import load_and_preprocess

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("predict")


def load_model(model_id: str):
    from neuralforecast import NeuralForecast
    path = cfg.MODELS_DIR / model_id
    if not path.exists():
        raise FileNotFoundError(f"No saved model at {path}. Run train_patchtst.py first.")
    return NeuralForecast.load(path=str(path))


def forecast_feeder(feeder: str, frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    model_id = cfg.FEEDERS[feeder]
    nf = load_model(model_id)
    hist = frames[feeder]

    # PatchTST is univariate; if an exog-capable model was trained instead,
    # NeuralForecast will pick the needed columns from `df` automatically.
    # Future calendar exog (if the model uses futr_exog) is built here too.
    futr = None
    model = nf.models[0]
    if getattr(model, "futr_exog_list", None):
        last = hist["ds"].max()
        future_ds = pd.date_range(last, periods=cfg.HORIZON + 1, freq=cfg.EXPECTED_FREQ)[1:]
        futr = pd.DataFrame({"unique_id": model_id, "ds": future_ds})
        from data_preprocessing import add_calendar_features
        futr = add_calendar_features(futr)
        futr = futr[["unique_id", "ds"] + list(model.futr_exog_list)]

    used_cols = (["unique_id", "ds", "y"]
                 + list(getattr(model, "hist_exog_list", None) or [])
                 + list(getattr(model, "futr_exog_list", None) or []))
    hist = hist[[c for c in dict.fromkeys(used_cols)]]
    fcst = nf.predict(df=hist, futr_df=futr)
    fcst = fcst.reset_index() if "unique_id" not in fcst.columns else fcst
    yhat_col = [c for c in fcst.columns if c not in ("unique_id", "ds")][0]
    fcst = fcst.rename(columns={yhat_col: "forecast_MW"})

    out = cfg.RESULTS_DIR / f"forecast_{model_id}.csv"
    fcst.to_csv(out, index=False)
    log.info("%s: forecast %d steps (%s -> %s) saved to %s",
             model_id, len(fcst), fcst["ds"].min(), fcst["ds"].max(), out)

    # ---- plot recent history + forecast -----------------------------------
    fig, ax = plt.subplots(figsize=(11, 4))
    recent = hist.tail(cfg.INPUT_SIZE * 2)
    ax.plot(recent["ds"], recent["y"], color="black", lw=1.3, label="History (Actual)")
    ax.plot(fcst["ds"], fcst["forecast_MW"], color="tab:red", lw=1.6,
            marker=".", label=f"Forecast (+{cfg.HORIZON} steps = {cfg.HORIZON*3} min)")
    ax.axvline(hist["ds"].max(), color="grey", ls="--", lw=1)
    ax.set_title(f"{model_id} — Future load forecast")
    ax.set_xlabel("Time"); ax.set_ylabel("Active Load (MW)")
    ax.legend(); fig.autofmt_xdate(); fig.tight_layout()
    fig.savefig(cfg.PLOTS_DIR / f"{model_id}_future_forecast.png", dpi=150)
    plt.close(fig)
    return fcst


def main() -> None:
    ap = argparse.ArgumentParser(description="Forecast future feeder load")
    ap.add_argument("--data", default=str(cfg.RAW_DATA_PATH), help="Path to raw data.csv")
    ap.add_argument("--feeder", choices=list(cfg.FEEDERS), help="Feeder to forecast")
    ap.add_argument("--all", action="store_true", help="Forecast all feeders")
    args = ap.parse_args()

    if not args.all and not args.feeder:
        ap.error("Provide --feeder \"<NAME>\" or --all")

    frames = load_and_preprocess(args.data)
    targets = list(cfg.FEEDERS) if args.all else [args.feeder]
    for feeder in targets:
        fcst = forecast_feeder(feeder, frames)
        print(f"\n--- {cfg.FEEDERS[feeder]} : next {cfg.HORIZON} steps ---")
        print(fcst.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
