# Per-Feeder PatchTST Load Forecasting (SCADA 3-min data)

Independent PatchTST models for 5 feeders across MALWADI and EMBASSY PARK
substations, forecasting Active Load (MW) 20 steps (1 hour) ahead from
96 steps (~4.8 h) of history.

## Quick start
```bash
pip install -r requirements.txt
python data_preprocessing.py --data data.csv   # optional standalone step
python train_patchtst.py     --data data.csv   # trains + rolling test eval, saves models/
python evaluate.py                              # metrics, comparison table, all plots
python predict.py --feeder "11KV KUBERA" --data data.csv   # next 72 min
python predict.py --all --data data.csv
```

## Files
| File | Purpose |
|---|---|
| `config.py` | All paths, feeder→model map, hyper-parameters, split ratios |
| `data_preprocessing.py` | Unit parsing (MW/kW, A, kV), MM/DD vs DD/MM auto-detection, 3-min grid resampling, outlier (rolling-MAD) repair, electrical + calendar feature engineering |
| `train_patchtst.py` | Per-feeder PatchTST training with chronological 70/15/15 split, early stopping on val, rolling-origin test forecasting, model saving |
| `evaluate.py` | MAE/RMSE/MAPE/R² per feeder, ranked comparison table, actual-vs-forecast / residual / correlation-heatmap / loss-curve plots |
| `predict.py` | Loads saved models, forecasts the next 20×3-min steps (1 hour) beyond the latest timestamp |

## Artefacts
- `models/PatchTST_<FEEDER>/` — trained NeuralForecast bundles (reload with `NeuralForecast.load`)
- `outputs/comparison_table.csv` — ranked performance summary
- `outputs/cv_*.csv` — every rolling test-window forecast (ds, cutoff, y, yhat)
- `outputs/forecast_*.csv` — future 72-min forecasts
- `plots/` — 4 diagnostic plots per feeder + future-forecast plot

## Important design notes
1. **PatchTST is univariate in NeuralForecast** (`EXOGENOUS_HIST/FUTR/STAT = False`).
   The pipeline detects this at runtime; currents/voltages/engineered features are
   still parsed and used for EDA/correlation. To train with exogenous features,
   set `MODEL_NAME = "NHITS"` or `"TSMixerx"` in `config.py` — nothing else changes.
2. **Timestamps are MM/DD/YYYY** in this export; the loader verifies by checking
   which format reproduces the expected ~3-minute interval.
3. **Mixed units handled**: `Active Load` contains both MW and kW rows (kW→MW),
   currents contain occasional "AM" unit typos, voltages are kV.
4. **R² caveat with one day of data**: within a 72-min window load is nearly flat,
   so window-level R² is harsh even when MAPE is ~4–5%. Accumulate ≥2–4 weeks of
   history (so the model sees repeated daily cycles) and increase `MAX_STEPS`;
   R² and longer-horizon skill will improve substantially.

## NHITS run notes (current configuration)
- `MODEL_NAME = "NHITS"`, `HORIZON = 20` (20 × 3 min = **1 hour ahead**).
- Day-pattern learning: future-known cyclical time-of-day features
  (`tod_sin`, `tod_cos`) condition every forecast on clock time; raw
  hour/minute/day_of_week are EDA-only (constant within a 4.8 h window →
  they break per-window scaling).
- Historical exog: IR, IY, IB, avg_current (informative). Voltages and
  imbalance % are near-constant — excluded from model inputs (kept for EDA).
- Scaling: `NeuralForecast(local_scaler_type="robust")` (whole-series, with
  automatic inverse transform) instead of per-window scaling, which diverges
  on near-constant exog channels.
