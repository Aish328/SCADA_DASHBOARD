
from __future__ import annotations
import logging, os
from pathlib import Path
import pandas as pd

log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_DIR      = Path(__file__).resolve().parent
PROCESSED_DIR    = PROJECT_DIR / "outputs" / "processed"
MODELS_DIR       = PROJECT_DIR / "models"
RESULTS_DIR      = PROJECT_DIR / "outputs"
PLOTS_DIR        = PROJECT_DIR / "plots"
GLOBAL_MODEL_ID  = "NHITS_GLOBAL"
GLOBAL_MODEL_DIR = MODELS_DIR / GLOBAL_MODEL_ID

for _d in (PROCESSED_DIR, MODELS_DIR, RESULTS_DIR, PLOTS_DIR, GLOBAL_MODEL_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── PostgreSQL ────────────────────────────────────────────────────────────────
DB_HOST     = os.getenv("DB_HOST",     "localhost")
DB_PORT     = int(os.getenv("DB_PORT", "5432"))
DB_NAME     = os.getenv("DB_NAME",     "scada_db")
DB_USER     = os.getenv("DB_USER",     "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "root")
DB_TABLE    = os.getenv("DB_TABLE",    "scada_db")
DB_SCHEMA   = os.getenv("DB_SCHEMA",   "public")

# ── DB column names ───────────────────────────────────────────────────────────
DB_COL_SUBSTATION  = "substation"
DB_COL_FEEDER      = "feeder"
DB_COL_IR          = "ir"
DB_COL_IY          = "iy"
DB_COL_IB          = "ib"
DB_COL_VRY         = "vry"
DB_COL_VYB         = "vyb"
DB_COL_VBR         = "vbr"
DB_COL_TIME        = "time"

DB_COL_ACTIVE_LOAD = "active_load"

DB_SELECT_COLS = [
    DB_COL_TIME, DB_COL_SUBSTATION, DB_COL_FEEDER,
    DB_COL_IR, DB_COL_IY, DB_COL_IB,
    DB_COL_VRY, DB_COL_VYB, DB_COL_VBR,
    DB_COL_ACTIVE_LOAD,
]

# ── Pipeline aliases ──────────────────────────────────────────────────────────
TIME_COL       = DB_COL_TIME
TARGET_COL     = DB_COL_ACTIVE_LOAD
FEEDER_COL     = DB_COL_FEEDER
SUBSTATION_COL = DB_COL_SUBSTATION
CURRENT_COLS   = [DB_COL_IR, DB_COL_IY, DB_COL_IB]
VOLTAGE_COLS   = [DB_COL_VRY, DB_COL_VYB, DB_COL_VBR]
EXPECTED_FREQ  = "3min"
TIME_FORMATS   = [
    "%m/%d/%Y %H:%M:%S", "%d/%m/%Y %H:%M:%S",
    "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S",
]

# ── Feeder registry ───────────────────────────────────────────────────────────
FEEDER_ID_MAP: dict[str, str] = {
    "11KV BHOSALE NAGAR":    "EP_FD01",
    "11KV KUBERA":           "EP_FD02",
    "RMU1":                  "ML_FD01",
    "11KV LUMAX":            "ML_FD02",
    "11KV MALWADI HADAPSAR": "ML_FD03",
}
ID_FEEDER_MAP: dict[str, str] = {v: k for k, v in FEEDER_ID_MAP.items()}

# ── Features ──────────────────────────────────────────────────────────────────
HIST_EXOG = CURRENT_COLS + ["avg_current"]
FUTR_EXOG = ["tod_sin", "tod_cos"]
ENGINEERED_FEATURES = [
    "avg_current", "current_imbalance_pct",
    "avg_voltage",  "voltage_imbalance_pct",
]
CALENDAR_FEATURES = ["hour", "minute", "day_of_week", "is_weekend", "tod_sin", "tod_cos"]
EDA_FEATURES      = CURRENT_COLS + VOLTAGE_COLS + ENGINEERED_FEATURES

# ── Hyperparameters ───────────────────────────────────────────────────────────
MODEL_NAME    = "NHITS"
INPUT_SIZE    = 48      # 48 × 3 min = 2.4 h  (raise to 96 once you have 7+ days in DB)
HORIZON       = 20      # 20 × 3 min = 1 h ahead
BATCH_SIZE    = 16      # small batch — less noisy gradient on single-day dataset
LEARNING_RATE = 1e-4    # reduced from 5e-4 to prevent training loss spikes
MAX_STEPS     = 1000    # early stopping will terminate well before this
RANDOM_SEED   = 42

# Trainer-level kwargs (safe to pass through NeuralForecast to Lightning)
NHITS_TRAINER_KWARGS = dict(
    scaler_type               = "standard",  # z-score per window — more stable than robust
    # early_stop_patience_steps = 5,           # stop if val doesn't improve for 5 checks
    gradient_clip_val         = 1.0,  
    val_check_steps           = 10,          # validate every 10 steps
)

# Architecture kwargs (passed directly to NHITS constructor, never to Trainer)
NHITS_ARCH_KWARGS = dict(
    n_stacks           = 3,
    n_blocks           = [1, 1, 1],
    mlp_units          = [[256, 256]] * 3,
    n_pool_kernel_size = [2, 2, 1],
    n_freq_downsample  = [4, 2, 1],
    dropout_prob_theta = 0.1,
    activation         = "ReLU",
    interpolation_mode = "linear",
    revin              = True,   # per-window instance norm — fixes level-shift bias
)

# Per-series (whole-series) scaler applied by NeuralForecast across all windows
LOCAL_SCALER_TYPE = "standard"   # z-score; fixes level bias vs "robust" on single-day data

# ── Split ratios ──────────────────────────────────────────────────────────────
TRAIN_FRAC   = 0.70
VAL_FRAC     = 0.15
TEST_FRAC    = 0.15
CV_STEP_SIZE = 1

# ── Outlier handling ──────────────────────────────────────────────────────────
OUTLIER_ROLLING_WINDOW = 21
OUTLIER_MAD_THRESHOLD  = 5.0

# ── PostgreSQL loader ─────────────────────────────────────────────────────────
def load_raw_data(
    feeders:  list[str] | None = None,
    start_dt: str | None       = None,
    end_dt:   str | None       = None,
    limit:    int | None       = None,
) -> pd.DataFrame:
    try:
        import psycopg2
    except ImportError:
        raise ImportError("pip install psycopg2-binary")

    target_feeders = feeders or list(FEEDER_ID_MAP.keys())
    placeholders   = ", ".join(["%s"] * len(target_feeders))
    where_parts    = [f"{DB_COL_FEEDER} IN ({placeholders})"]
    params         = list(target_feeders)

    if start_dt:
        where_parts.append(f"{DB_COL_TIME} >= %s"); params.append(start_dt)
    if end_dt:
        where_parts.append(f"{DB_COL_TIME} <= %s"); params.append(end_dt)

    col_list  = ", ".join(DB_SELECT_COLS)
    where_sql = "WHERE " + " AND ".join(where_parts)
    limit_sql = f"LIMIT {int(limit)}" if limit else ""
    query = (f"SELECT {col_list} FROM {DB_SCHEMA}.{DB_TABLE} "
             f"{where_sql} ORDER BY {DB_COL_TIME} ASC {limit_sql};")

    log.info("Querying %s.%s  feeders=%s  start=%s  end=%s  limit=%s",
             DB_SCHEMA, DB_TABLE, target_feeders, start_dt, end_dt, limit)

    dsn  = (f"host={DB_HOST} port={DB_PORT} dbname={DB_NAME} "
            f"user={DB_USER} password={DB_PASSWORD}")
    conn = psycopg2.connect(dsn)
    try:
        df = pd.read_sql_query(query, conn, params=params)
    finally:
        conn.close()

    if df.empty:
        raise RuntimeError(
            f"No rows returned from {DB_SCHEMA}.{DB_TABLE} "
            f"for feeders {target_feeders}."
        )
    log.info("Loaded %d rows from PostgreSQL (%d feeders)",
             len(df), df[DB_COL_FEEDER].nunique())
    return df