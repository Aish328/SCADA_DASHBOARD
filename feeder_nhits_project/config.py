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

# ── Data Source ───────────────────────────────────────────────────────────────
DATA_SOURCE_TYPE = os.getenv("DATA_SOURCE_TYPE", "csv")
CSV_FILE_PATH    = os.getenv("CSV_FILE_PATH", "SCADA_All_Feeders_Combined.csv")

# ── PostgreSQL ────────────────────────────────────────────────────────────────
DB_HOST     = os.getenv("DB_HOST",     "localhost")
DB_PORT     = int(os.getenv("DB_PORT", "5432"))
DB_NAME     = os.getenv("DB_NAME",     "scada_db")
DB_USER     = os.getenv("DB_USER",     "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "root")
DB_TABLE    = os.getenv("DB_TABLE",    "scada_db")
DB_SCHEMA   = os.getenv("DB_SCHEMA",   "public")

# ── CSV column names (exactly as they appear in the CSV header) ────────────────
CSV_COL_TIMESTAMP      = "Timestamp"
CSV_COL_FEEDER         = "Feeder"
CSV_COL_SUBSTATION     = "Substation"
CSV_COL_IR             = "IR_A"
CSV_COL_IY             = "IY_A"
CSV_COL_IB             = "IB_A"
CSV_COL_VRY            = "VRY_kV"
CSV_COL_VYB            = "VYB_kV"
CSV_COL_VBR            = "VBR_kV"
CSV_COL_ACTIVE_LOAD    = "Active_Load_MW"
CSV_COL_DATA_SOURCE    = "data_source"

# ── DB column names ───────────────────────────────────────────────────────────
DB_COL_TIME        = "time"
DB_COL_FEEDER      = "feeder"
DB_COL_SUBSTATION  = "substation"
DB_COL_IR          = "ir"
DB_COL_IY          = "iy"
DB_COL_IB          = "ib"
DB_COL_VRY         = "vry"
DB_COL_VYB         = "vyb"
DB_COL_VBR         = "vbr"
DB_COL_ACTIVE_LOAD = "active_load"

# ── Pipeline aliases ──────────────────────────────────────────────────────────
# Set ONCE at import time based on data source
if DATA_SOURCE_TYPE == "csv":
    TIME_COL       = CSV_COL_TIMESTAMP
    TARGET_COL     = CSV_COL_ACTIVE_LOAD
    FEEDER_COL     = CSV_COL_FEEDER
    SUBSTATION_COL = CSV_COL_SUBSTATION
    CURRENT_COLS   = [CSV_COL_IR, CSV_COL_IY, CSV_COL_IB]
    VOLTAGE_COLS   = [CSV_COL_VRY, CSV_COL_VYB, CSV_COL_VBR]
else:
    TIME_COL       = DB_COL_TIME
    TARGET_COL     = DB_COL_ACTIVE_LOAD
    FEEDER_COL     = DB_COL_FEEDER
    SUBSTATION_COL = DB_COL_SUBSTATION
    CURRENT_COLS   = [DB_COL_IR, DB_COL_IY, DB_COL_IB]
    VOLTAGE_COLS   = [DB_COL_VRY, DB_COL_VYB, DB_COL_VBR]

EXPECTED_FREQ  = "3min"
TIME_FORMATS   = [
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%m/%d/%Y %H:%M:%S",
    "%d/%m/%Y %H:%M:%S",
]

# ── Feeder registry ───────────────────────────────────────────────────────────
FEEDER_ID_MAP: dict[str, str] = {
    "11KV TAMHANI FEEDER":      "ML_FD04",
    "11 KV TATA EV":            "EP_FD03",
    "RMU1":                     "ML_FD01",
    "RMU2":                     "ML_FD05",
    "11KV LUMAX":               "ML_FD02",
    "11KV BHOSALE NAGAR":       "EP_FD01",
    "11KV KUBERA":              "EP_FD02",
    "11KV MALWADI HADAPSAR":    "ML_FD03",
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
INPUT_SIZE    = 48
HORIZON       = 20
BATCH_SIZE    = 16
LEARNING_RATE = 1e-4
MAX_STEPS     = 1000
RANDOM_SEED   = 42

NHITS_TRAINER_KWARGS = dict(
    scaler_type       = "standard",
    gradient_clip_val = 1.0,
    val_check_steps   = 10,
)
 
NHITS_ARCH_KWARGS = dict(
    n_stacks           = 3,
    n_blocks           = [1, 1, 1],
    mlp_units          = [[256, 256]] * 3,
    n_pool_kernel_size = [2, 2, 1],
    n_freq_downsample  = [4, 2, 1],
    dropout_prob_theta = 0.1,
    activation         = "ReLU",
    interpolation_mode = "linear",
    # revin removed - not supported in this NHITS version
)

LOCAL_SCALER_TYPE = "standard"
NHITS_TRAINER_KWARGS = dict(
    gradient_clip_val = 1.0,  
    val_check_steps   = 10,
)
TRAIN_FRAC   = 0.70
VAL_FRAC     = 0.15
TEST_FRAC    = 0.15
CV_STEP_SIZE = 1

OUTLIER_ROLLING_WINDOW = 21
OUTLIER_MAD_THRESHOLD  = 5.0


# ── Data loading ───────────────────────────────────────────────────────────────
def load_raw_data(
    feeders:  list[str] | None = None,
    start_dt: str | None       = None,
    end_dt:   str | None       = None,
    limit:    int | None       = None,
) -> pd.DataFrame:
    if DATA_SOURCE_TYPE == "csv":
        return _load_from_csv(feeders, start_dt, end_dt, limit)
    else:
        return _load_from_postgres(feeders, start_dt, end_dt, limit)


def _load_from_csv(
    feeders:  list[str] | None = None,
    start_dt: str | None       = None,
    end_dt:   str | None       = None,
    limit:    int | None       = None,
) -> pd.DataFrame:
    csv_path = Path(CSV_FILE_PATH)
    if not csv_path.is_absolute():
        csv_path = PROJECT_DIR / csv_path

    log.info("Loading data from CSV: %s", csv_path)

    df = pd.read_csv(csv_path)

    # ── Deduplicate: prefer actual over synthetic ───────────────────────────
    if CSV_COL_DATA_SOURCE in df.columns:
        df = df.sort_values([CSV_COL_FEEDER, CSV_COL_TIMESTAMP, CSV_COL_DATA_SOURCE])
        df = df.groupby([CSV_COL_FEEDER, CSV_COL_TIMESTAMP], group_keys=False).apply(
            lambda x: x[x[CSV_COL_DATA_SOURCE] == "actual"].iloc[0]
            if (x[CSV_COL_DATA_SOURCE] == "actual").any()
            else x.iloc[0]
        ).reset_index(drop=True)

    # ── Filter feeders ──────────────────────────────────────────────────────
    target_feeders = feeders or list(FEEDER_ID_MAP.keys())
    df = df[df[CSV_COL_FEEDER].isin(target_feeders)]

    # ── Filter date range ───────────────────────────────────────────────────
    df[CSV_COL_TIMESTAMP] = pd.to_datetime(df[CSV_COL_TIMESTAMP])
    if start_dt:
        df = df[df[CSV_COL_TIMESTAMP] >= pd.to_datetime(start_dt)]
    if end_dt:
        df = df[df[CSV_COL_TIMESTAMP] <= pd.to_datetime(end_dt)]

    if limit:
        df = df.head(limit)

    # ── Select ONLY the columns we need (keep CSV names, DO NOT rename) ─────
    keep_cols = [
        CSV_COL_TIMESTAMP,
        CSV_COL_SUBSTATION,
        CSV_COL_FEEDER,
        CSV_COL_IR,
        CSV_COL_IY,
        CSV_COL_IB,
        CSV_COL_VRY,
        CSV_COL_VYB,
        CSV_COL_VBR,
        CSV_COL_ACTIVE_LOAD,
    ]
    # Defensive: only keep columns that actually exist
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols]

    if df.empty:
        raise RuntimeError(f"No rows returned from CSV for feeders {target_feeders}")

    log.info("Loaded %d rows from CSV (%d feeders). Columns: %s",
             len(df), df[CSV_COL_FEEDER].nunique(), df.columns.tolist())

    return df


def _load_from_postgres(
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

    col_list  = ", ".join([
        DB_COL_TIME, DB_COL_SUBSTATION, DB_COL_FEEDER,
        DB_COL_IR, DB_COL_IY, DB_COL_IB,
        DB_COL_VRY, DB_COL_VYB, DB_COL_VBR,
        DB_COL_ACTIVE_LOAD,
    ])
    where_sql = "WHERE " + " AND ".join(where_parts)
    limit_sql = f"LIMIT {int(limit)}" if limit else ""
    query = (f"SELECT {col_list} FROM {DB_SCHEMA}.{DB_TABLE} "
             f"{where_sql} ORDER BY {DB_COL_TIME} ASC {limit_sql};")

    log.info("Querying %s.%s  feeders=%s", DB_SCHEMA, DB_TABLE, target_feeders)

    dsn  = (f"host={DB_HOST} port={DB_PORT} dbname={DB_NAME} "
            f"user={DB_USER} password={DB_PASSWORD}")
    conn = psycopg2.connect(dsn)
    try:
        df = pd.read_sql_query(query, conn, params=params)
    finally:
        conn.close()

    if df.empty:
        raise RuntimeError(f"No rows returned from PostgreSQL for feeders {target_feeders}")

    log.info("Loaded %d rows from PostgreSQL (%d feeders)", len(df), df[DB_COL_FEEDER].nunique())
    return df