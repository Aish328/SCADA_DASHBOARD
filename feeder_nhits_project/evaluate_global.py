"""
evaluate_global.py
==================
Evaluation and benchmarking for the global NHITS model.

Produces
--------
Per feeder (from cv_global.csv):
  • MAE, RMSE, MAPE, R² on the rolling test span
  • Actual vs Forecast plot
  • Residual diagnostics (scatter + histogram)
  • Feature correlation heatmap

Global:
  • Training / validation loss curves
  • outputs/comparison_table_global.csv  — feeders ranked by MAPE

Comparison mode (optional):
  • If per-feeder cv_*.csv files from the old pipeline also exist,
    outputs/model_comparison.csv  shows Global vs Individual side-by-side.
    Run:  python evaluate_global.py --compare

Run:
    python evaluate_global.py [--compare]
"""

from __future__ import annotations

import argparse
import json
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import config as cfg

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("evaluate_global")
EPS = 1e-6


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def compute_metrics(y: np.ndarray, yhat: np.ndarray) -> dict[str, float]:
    mae  = mean_absolute_error(y, yhat)
    rmse = float(np.sqrt(mean_squared_error(y, yhat)))
    mape = float(np.mean(np.abs((y - yhat) / np.clip(np.abs(y), EPS, None))) * 100.0)
    r2   = r2_score(y, yhat)
    return {"MAE (MW)": mae, "RMSE (MW)": rmse, "MAPE (%)": mape, "R2": r2}


# --------------------------------------------------------------------------- #
# Per-feeder plots
# --------------------------------------------------------------------------- #
def plot_actual_vs_forecast(cv: pd.DataFrame, uid: str) -> None:
    """Stitch non-overlapping HORIZON-step trajectories across the test span."""
    name     = cfg.ID_FEEDER_MAP.get(uid, uid)
    cutoffs  = sorted(cv["cutoff"].unique())
    selected = cutoffs[::cfg.HORIZON] if len(cutoffs) >= cfg.HORIZON else cutoffs[:1]

    fig, ax = plt.subplots(figsize=(13, 4.5))
    actual  = cv.drop_duplicates("ds").sort_values("ds")
    ax.plot(actual["ds"], actual["y"], color="black", lw=1.4, label="Actual")
    for i, c in enumerate(selected):
        w = cv[cv["cutoff"] == c].sort_values("ds")
        ax.plot(w["ds"], w["yhat"], lw=1.4, alpha=0.9, color="tab:red",
                label="NHITS Global forecast" if i == 0 else None)
        ax.axvline(w["ds"].iloc[0], color="grey", lw=0.5, ls=":")
    ax.set_title(f"{uid} ({name}) — Actual vs Forecast  "
                 f"(test span, {cfg.HORIZON}-step / {cfg.HORIZON*3}-min windows)")
    ax.set_xlabel("Time"); ax.set_ylabel("Active Load (MW)")
    ax.legend(); fig.autofmt_xdate(); fig.tight_layout()
    fig.savefig(cfg.PLOTS_DIR / f"{uid}_actual_vs_forecast.png", dpi=150)
    plt.close(fig)


def plot_residuals(cv: pd.DataFrame, uid: str) -> None:
    name = cfg.ID_FEEDER_MAP.get(uid, uid)
    res  = cv["y"].to_numpy() - cv["yhat"].to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].scatter(cv["ds"], res, s=4, alpha=0.35, color="tab:blue")
    axes[0].axhline(0, color="red", lw=1)
    axes[0].set_title(f"{uid} ({name}) — Residuals over test span")
    axes[0].set_xlabel("Time"); axes[0].set_ylabel("Residual (MW)")
    axes[1].hist(res, bins=40, color="tab:blue", alpha=0.8)
    axes[1].axvline(0, color="red", lw=1)
    axes[1].set_title("Residual distribution")
    axes[1].set_xlabel("Residual (MW)")
    fig.autofmt_xdate(); fig.tight_layout()
    fig.savefig(cfg.PLOTS_DIR / f"{uid}_residuals.png", dpi=150)
    plt.close(fig)


def plot_correlation_heatmap(uid: str) -> None:
    proc_path = cfg.PROCESSED_DIR / f"{uid}_processed.csv"
    if not proc_path.exists():
        log.warning("Processed CSV not found for %s — skipping heatmap.", uid)
        return
    proc = pd.read_csv(proc_path)
    cols = ["y"] + list(dict.fromkeys(cfg.EDA_FEATURES + cfg.CALENDAR_FEATURES))
    cols = [c for c in cols if c in proc.columns]
    corr = proc[cols].corr()
    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(cols)), cols, fontsize=8)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}",
                    ha="center", va="center", fontsize=6)
    fig.colorbar(im, fraction=0.046)
    name = cfg.ID_FEEDER_MAP.get(uid, uid)
    ax.set_title(f"{uid} ({name}) — Feature correlation (y = Active Load MW)")
    fig.tight_layout()
    fig.savefig(cfg.PLOTS_DIR / f"{uid}_correlation_heatmap.png", dpi=150)
    plt.close(fig)


def plot_loss_curve() -> None:
    fp = cfg.RESULTS_DIR / "loss_global.json"
    if not fp.exists():
        log.warning("loss_global.json not found — skipping loss curve.")
        return
    traj = json.loads(fp.read_text())
    fig, ax = plt.subplots(figsize=(8, 4))
    if traj.get("train"):
        t = np.array(traj["train"])
        ax.plot(t[:, 0], t[:, 1], label="Train loss (MAE)")
    if traj.get("valid"):
        v = np.array(traj["valid"])
        ax.plot(v[:, 0], v[:, 1], marker="o", ms=3, label="Validation loss (MAE)")
    ax.set_title("NHITS Global — Training loss curve (scaled space)")
    ax.set_xlabel("Step"); ax.set_ylabel("Loss (MAE)")
    ax.legend(); fig.tight_layout()
    fig.savefig(cfg.PLOTS_DIR / "global_loss_curve.png", dpi=150)
    plt.close(fig)
    log.info("Loss curve → %s/global_loss_curve.png", cfg.PLOTS_DIR)


# --------------------------------------------------------------------------- #
# Comparison: global vs per-feeder individual models
# --------------------------------------------------------------------------- #
def compare_global_vs_individual(cv_global: pd.DataFrame) -> None:
    """
    Side-by-side metric comparison.

    Reads per-feeder cv_<MODEL_ID>.csv files if they exist (produced by the
    old train_patchtst.py / train_nhits.py pipeline), and computes the same
    metrics so results are directly comparable.

    Outputs outputs/model_comparison.csv.
    """
    rows = []
    for uid, grp in cv_global.groupby("unique_id"):
        name = cfg.ID_FEEDER_MAP.get(uid, uid)
        m_g  = compute_metrics(grp["y"].to_numpy(), grp["yhat"].to_numpy())
        row  = {"Feeder": name, "unique_id": uid,
                **{f"Global_{k}": v for k, v in m_g.items()}}

        # Try to find the per-feeder result file from the old pipeline
        # Old model IDs follow the pattern used by the original config:
        # e.g.  NHITS_11KV_KUBERA,  PatchTST_11KV_KUBERA …
        for old_id in cfg.RESULTS_DIR.glob(f"cv_*{uid}*.csv"):
            old_cv = pd.read_csv(old_id, parse_dates=["ds", "cutoff"])
            if "yhat" not in old_cv.columns:
                # handle different column naming from the old pipeline
                yhat_col = [c for c in old_cv.columns
                            if c not in ("unique_id", "ds", "cutoff", "y")][0]
                old_cv = old_cv.rename(columns={yhat_col: "yhat"})
            m_i = compute_metrics(old_cv["y"].to_numpy(), old_cv["yhat"].to_numpy())
            row.update({f"Individual_{k}": v for k, v in m_i.items()})
            row["Individual_model"] = old_id.stem
            break
        rows.append(row)

    comp = pd.DataFrame(rows)
    out  = cfg.RESULTS_DIR / "model_comparison.csv"
    comp.to_csv(out, index=False)

    # Pretty-print the comparison
    print("\n========== GLOBAL vs INDIVIDUAL MODEL COMPARISON ==========")
    g_cols = [c for c in comp.columns if c.startswith("Global_")]
    i_cols = [c for c in comp.columns if c.startswith("Individual_")]
    if i_cols:
        for _, row in comp.iterrows():
            print(f"\n  {row['unique_id']}  {row['Feeder']}")
            for g, i in zip(g_cols, i_cols):
                metric = g.replace("Global_", "")
                gv = row.get(g, float("nan"))
                iv = row.get(i, float("nan"))
                winner = "← Global wins" if gv < iv else "← Individual wins"
                print(f"    {metric:12s}  Global={gv:.4f}  Individual={iv:.4f}  {winner}")
    else:
        log.info("No per-feeder cv_*.csv files found — global metrics only.")
        print(comp[["Feeder", "unique_id"] + g_cols].to_string(index=False))

    log.info("Comparison table → %s", out)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(compare: bool = False) -> None:
    cv_path = cfg.RESULTS_DIR / "cv_global.csv"
    if not cv_path.exists():
        log.error("cv_global.csv not found. Run train_global_nhits.py first.")
        return

    cv = pd.read_csv(cv_path, parse_dates=["ds", "cutoff"])

    rows = []
    for uid, grp in cv.groupby("unique_id"):
        name = cfg.ID_FEEDER_MAP.get(uid, uid)
        m    = compute_metrics(grp["y"].to_numpy(), grp["yhat"].to_numpy())
        rows.append({"Feeder": name, "unique_id": uid, **m,
                     "Test windows": grp["cutoff"].nunique()})
        plot_actual_vs_forecast(grp, uid)
        plot_residuals(grp, uid)
        plot_correlation_heatmap(uid)
        log.info("%-8s  %-26s  MAE=%.4f  RMSE=%.4f  MAPE=%.2f%%  R2=%.4f",
                 uid, name, m["MAE (MW)"], m["RMSE (MW)"], m["MAPE (%)"], m["R2"])

    plot_loss_curve()

    table = (pd.DataFrame(rows)
               .sort_values("MAPE (%)")
               .reset_index(drop=True))
    table.insert(0, "Rank", table.index + 1)
    table.to_csv(cfg.RESULTS_DIR / "comparison_table_global.csv", index=False)

    print("\n======== GLOBAL NHITS — FEEDER PERFORMANCE (ranked by MAPE) ========")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    if compare:
        compare_global_vs_individual(cv)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Evaluate global NHITS model; optionally compare to per-feeder models"
    )
    ap.add_argument(
        "--compare", action="store_true",
        help="Compare global model against per-feeder cv_*.csv files if available"
    )
    args = ap.parse_args()
    main(compare=args.compare)