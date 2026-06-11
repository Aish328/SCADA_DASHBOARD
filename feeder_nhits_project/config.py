"""
config.py
=========
Central configuration for the per-feeder PatchTST load-forecasting project.

All paths, column maps, feeder definitions, model hyper-parameters and
split ratios live here so the other modules stay free of magic numbers.
"""

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_DIR = Path(__file__).resolve().parent
RAW_DATA_PATH = Path("/home/sharika/Desktop/SCADA_DASHBOARD/data/data.csv")          # raw SCADA export
PROCESSED_DIR = PROJECT_DIR / "outputs" / "processed"
MODELS_DIR = PROJECT_DIR / "models"
RESULTS_DIR = PROJECT_DIR / "outputs"
PLOTS_DIR = PROJECT_DIR / "plots"

for _d in (PROCESSED_DIR, MODELS_DIR, RESULTS_DIR, PLOTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Raw-data schema
# --------------------------------------------------------------------------- #
TIME_COL = "Time"
TARGET_COL = "Active Load"            # parsed to MW
FEEDER_COL = "FEEDER"
SUBSTATION_COL = "SUBSTATION"

CURRENT_COLS = ["IR", "IY", "IB"]     # Amps  (e.g. "74.9 A", occasional "AM" typo)
VOLTAGE_COLS = ["VRY", "VYB", "VBR"]  # kV    (e.g. "10.8 kV")

# Candidate timestamp formats; the loader auto-selects the one whose median
# sampling interval best matches EXPECTED_FREQ (the export uses MM/DD/YYYY).
TIME_FORMATS = ["%m/%d/%Y %H:%M:%S", "%d/%m/%Y %H:%M:%S"]

EXPECTED_FREQ = "3min"                # nominal SCADA polling interval

# --------------------------------------------------------------------------- #
# Model selection (placed before FEEDERS so model IDs can use it)
# --------------------------------------------------------------------------- #
MODEL_NAME = "NHITS"   # exog-capable: uses currents/voltages (hist) + calendar (futr)

# --------------------------------------------------------------------------- #
# Feeders -> model names
# --------------------------------------------------------------------------- #
# Keys are the exact FEEDER strings found in data.csv.
_FEEDER_SUFFIXES = {
    "11KV BHOSALE NAGAR":    "11KV_BHOSALENAGAR",
    "11KV KUBERA":           "11KV_KUBERA",
    "11KV MALWADI HADAPSAR": "11KV_MALWADI_HADAPSAR",
    "RMU1":                  "RMU1",
    "11KV LUMAX":            "11KV_LUMAX",
}
FEEDERS = {k: f"{MODEL_NAME}_{v}" for k, v in _FEEDER_SUFFIXES.items()}

# --------------------------------------------------------------------------- #
# Engineered / calendar feature names (created in data_preprocessing.py)
# --------------------------------------------------------------------------- #
ENGINEERED_FEATURES = [
    "avg_current",
    "current_imbalance_pct",
    "avg_voltage",
    "voltage_imbalance_pct",
]
CALENDAR_FEATURES = ["hour", "minute", "day_of_week", "is_weekend", "tod_sin", "tod_cos"]

# Exogenous feature lists (used automatically when the chosen model
# supports them — PatchTST does not, NHITS / TSMixerx etc. do).
# Model inputs deliberately EXCLUDE the near-constant channels (voltages sit
# at ~10.8-10.9 kV; imbalance %% barely moves): they add no signal and their
# tiny variance destabilises normalisation. They remain in EDA_FEATURES for
# the correlation heatmaps and in the processed CSVs.
HIST_EXOG = CURRENT_COLS + ["avg_current"]          # informative past-only channels
FUTR_EXOG = ["tod_sin", "tod_cos"]   # future-known day-pattern conditioning (cyclical)
EDA_FEATURES = CURRENT_COLS + VOLTAGE_COLS + ENGINEERED_FEATURES

# Series-level (per-feeder, whole-series) robust scaling applied by
# NeuralForecast to target + exog, with automatic inverse-transform of
# forecasts. Window-level scaling (model scaler_type) explodes on
# near-constant exog channels, so exog-capable models use this instead.
LOCAL_SCALER_TYPE = "robust"

# --------------------------------------------------------------------------- #
# Model hyper-parameters
# --------------------------------------------------------------------------- #

INPUT_SIZE = 96           # 96 x 3 min  ≈ 4.8 h of history
HORIZON = 20              # 20 x 3 min = 1 hour forecast
BATCH_SIZE = 32
LEARNING_RATE = 5e-4
MAX_STEPS = 500
RANDOM_SEED = 42

# Capacity is deliberately small: with one day of 3-minute data there are
# only ~350 training samples per feeder, and larger configurations overfit
# within ~50 optimisation steps. Tight early stopping (frequent val checks,
# low patience) matters because Lightning keeps the LAST weights, not the
# best — stopping close to the validation optimum is the checkpointing.
PATCHTST_KWARGS = dict(
    patch_len=16,
    stride=8,
    hidden_size=64,
    n_heads=8,
    encoder_layers=2,
    revin=True,            # instance normalisation — important for load data
    scaler_type="robust",
    dropout=0.3,
    early_stop_patience_steps=3,
    val_check_steps=10,
)

# --------------------------------------------------------------------------- #
# Chronological split
# --------------------------------------------------------------------------- #
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15

# Rolling-origin evaluation over the test span: stride between forecast
# origins (1 = every timestamp becomes a forecast origin).
CV_STEP_SIZE = 1

# --------------------------------------------------------------------------- #
# Outlier handling
# --------------------------------------------------------------------------- #
OUTLIER_ROLLING_WINDOW = 21    # samples (~1 h) for rolling-median filter
OUTLIER_MAD_THRESHOLD = 5.0    # flag |x - rolling_median| > k * rolling_MAD
