import pandas as pd
from pathlib import Path
from typing import Optional
from database import engine
from sqlalchemy import create_engine, text
# DATA_PATH = Path(__file__).parent.parent / "data" / "data.csv"
# XLSX_PATH = Path(__file__).parent.parent / "data" / "MAIIN_DATA.xlsx"
# CSV_PATH  = Path(__file__).parent.parent / "data" / "MAIIN_DATA.csv"

# print("DATA DIR:", DATA_PATH.parent)

_df_cache: Optional[pd.DataFrame] = None

VOLTAGE_HIGH = 95
VOLTAGE_LOW  = 75
CURRENT_HIGH = 11.1

CURRENT_LOW  = 10.9
DEFAULT_INTERVAL_MINUTES = 15


# def _find_data_file():
#     for p in [DATA_PATH, XLSX_PATH, CSV_PATH]:
#         if p.exists():
#             return p
#     return None


def _infer_interval_minutes(df: pd.DataFrame) -> float:
    if "datetime" not in df.columns or len(df) < 2:
        return DEFAULT_INTERVAL_MINUTES
    times = df["datetime"].dropna().sort_values()
    gaps = times.diff().dropna().dt.total_seconds() / 60
    median_gap = gaps[gaps > 0].median()
    return float(median_gap) if pd.notna(median_gap) else DEFAULT_INTERVAL_MINUTES


def _compute_episodes(flag_series: pd.Series, value_series: pd.Series, interval_minutes: float):
    """
    Given a binary flag series and the actual value series, compute:
      - episode_id : integer id for each contiguous anomaly run (0 = no anomaly)
      - episode_duration_min : total duration of the episode this row belongs to (0 if not anomaly)

    Returns (episode_id_series, episode_duration_series) both indexed like flag_series.
    """
    ep_id  = pd.Series(0,   index=flag_series.index, dtype=int)
    ep_dur = pd.Series(0.0, index=flag_series.index)
    # episode_summaries = []
    # First pass: assign episode ids
    episode = 0
    run_indices = []
    flag_vals = flag_series.fillna(0).astype(int)

    # Group contiguous runs
    runs = []   # list of (episode_num, [indices])
    cur_run = []
    in_anomaly = False
    for idx in flag_series.index:
        if flag_vals[idx] == 1:
            if not in_anomaly:
                episode += 1
                in_anomaly = True
                cur_run = []
            cur_run.append(idx)
            ep_id[idx] = episode
        else:
            if in_anomaly:
                runs.append((episode, cur_run))
                cur_run = []
            in_anomaly = False

    if in_anomaly and cur_run:
        runs.append((episode, cur_run))

    

    # Second pass: assign duration = count of rows in that episode * interval
    # for ep_num, indices in runs:
    #     duration = len(indices) * interval_minutes
    #     for idx in indices:
    #         ep_dur[idx] = round(duration, 1)
    for ep_num, indices in runs:
        duration = len(indices) * interval_minutes

        # Store duration only at first anomaly point
        ep_dur[indices[0]] = round(duration, 1)
        # if datetime_series is not None:

        #     episode_summary = {
        #         "episode_id": ep_num,
        #         "start": datetime_series.loc[indices[0]],
        #         "end": datetime_series.loc[indices[-1]],
        #         "duration_min": round(duration, 1),
        #         "peak_value": value_series.loc[indices].max()
        #     }

            # episode_summaries.append(episode_summary)
    return ep_id, ep_dur


def _derive_kpis(df: pd.DataFrame) -> pd.DataFrame:
    interval = _infer_interval_minutes(df)
    print(f"  Inferred reading interval: {interval:.1f} min")

    volt_cols = [c for c in ["vry", "vyb", "vbr"] if c in df.columns]
    curr_cols = [c for c in ["ir", "iy", "ib"] if c in df.columns]

    if volt_cols:
        vdf = df[volt_cols]
        v_max = vdf.max(axis=1)
        v_min = vdf.min(axis=1)
        df["fvsm"] = vdf.mean(axis=1).round(3)

        # Binary flags
        df["fvhi"] = (v_max > VOLTAGE_HIGH).astype(int)
        df["fvli"] = (v_min < VOLTAGE_LOW).astype(int)

        # Episode id + duration per episode — per (substation,feeder) group
        df["fvhi_ep"]  = 0;  df["fvhd"] = 0.0
        df["fvli_ep"]  = 0;  df["fvld"] = 0.0

        for _, grp in df.groupby(["substation", "feeder"], sort=False):
            g = grp.sort_values("datetime")
            hi_ep, hi_dur = _compute_episodes(g["fvhi"], g["fvsm"], interval)
            lo_ep, lo_dur = _compute_episodes(g["fvli"], g["fvsm"], interval)
            df.loc[g.index, "fvhi_ep"] = hi_ep.values
            df.loc[g.index, "fvhd"]    = hi_dur.values
            df.loc[g.index, "fvli_ep"] = lo_ep.values
            df.loc[g.index, "fvld"]    = lo_dur.values

        # Total episode counts per feeder group → use max of episode id
        df["fvhi_count"] = df.groupby(["substation","feeder"])["fvhi_ep"].transform("max")
        df["fvli_count"] = df.groupby(["substation","feeder"])["fvli_ep"].transform("max")
    else:
        for col in ["fvhi","fvhd","fvhi_ep","fvhi_count","fvli","fvld","fvli_ep","fvli_count","fvsm"]:
            if col not in df.columns: df[col] = 0

    if curr_cols:
        cdf = df[curr_cols]
        c_max = cdf.max(axis=1)
        c_min = cdf.min(axis=1)
        df["fcsm"] = cdf.mean(axis=1).round(3)

        df["fchi"] = (c_max > CURRENT_HIGH).astype(int)
        df["fcli"] = (c_min < CURRENT_LOW).astype(int)

        df["fchi_ep"]  = 0;  df["fchd"] = 0.0
        df["fcli_ep"]  = 0;  df["fcld"] = 0.0

        for _, grp in df.groupby(["substation", "feeder"], sort=False):
            g = grp.sort_values("datetime")
            hi_ep, hi_dur = _compute_episodes(g["fchi"], g["fcsm"], interval)
            lo_ep, lo_dur = _compute_episodes(g["fcli"], g["fcsm"], interval) #COMPUTING EPISODE
            df.loc[g.index, "fchi_ep"] = hi_ep.values
            df.loc[g.index, "fchd"]    = hi_dur.values
            df.loc[g.index, "fcli_ep"] = lo_ep.values
            df.loc[g.index, "fcld"]    = lo_dur.values

        df["fchi_count"] = df.groupby(["substation","feeder"])["fchi_ep"].transform("max")
        df["fcli_count"] = df.groupby(["substation","feeder"])["fcli_ep"].transform("max")
    else:
        for col in ["fchi","fchd","fchi_ep","fchi_count","fcli","fcld","fcli_ep","fcli_count","fcsm"]:
            if col not in df.columns: df[col] = 0

    return df

def parse_active_load(x):
    if pd.isna(x):
        return None

    s = str(x).strip().lower()

    try:
        if "mw" in s:
            return float(s.replace("mw", "").strip())

        elif "kw" in s:
            return float(s.replace("kw", "").strip()) / 1000.0

        else:
            return float(s)
    except:
        return None
    
def _load() -> pd.DataFrame:
    global _df_cache
    if _df_cache is not None:
        return _df_cache.copy()
    query = "SELECT * FROM test3_db"
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    # data_file = engine.connect().execute(text("SELECT * FROM test_db"))
    
    # df = (pd.read_sql(data_file, engine)
    #         #   if data_file.suffix.lower() in [".xlsx", ".xls"]
    #           else pd.read_csv(data_file))

    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    for tc in ["time", "datetime", "date", "timestamp"]:
        if tc in df.columns:
            df["datetime"] = pd.to_datetime(df[tc], errors="coerce")
            if tc != "datetime":
                df.drop(columns=[tc], inplace=True, errors="ignore")
            break
    else:
        df["datetime"] = pd.Timestamp.now()

    for col in ["substation", "feeder"]:
        if col not in df.columns:
            df[col] = "Unknown"
        df[col] = df[col].astype(str).str.strip()

    for col in ["ir", "iy", "ib", "vry", "vyb", "vbr", "active_load", "active load"]:
        if "active_load" in df.columns:
            df["active_load"] = df["active_load"].apply(parse_active_load)
        if col in df.columns:
            df[col] = (
                df[col].astype(str)
                .str.replace(r"[^0-9.\-]+", "", regex=True)
                .replace("", pd.NA)
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = _derive_kpis(df)
    _df_cache = df
    return df.copy()


class DataLoader:
    @staticmethod
    def get_all_data() -> pd.DataFrame:
        return _load()

    @staticmethod
    def refresh():
        global _df_cache
        _df_cache = None
        return _load()

    @staticmethod
    def get_unique_substations() -> list:
        return sorted(_load()["substation"].dropna().unique().tolist())

    @staticmethod
    def get_unique_feeders(substation: Optional[str] = None) -> list:
        df = _load()
        if substation:
            df = df[df["substation"] == substation]
        return sorted(df["feeder"].dropna().unique().tolist())

    @staticmethod
    def filter_data(df, substation=None, feeder=None, date_from=None, date_to=None):
        if substation:
            df = df[df["substation"] == substation]
        if feeder:
            df = df[df["feeder"] == feeder]
        if date_from:
            df = df[df["datetime"] >= pd.to_datetime(date_from)]
        if date_to:
            df = df[df["datetime"] <= pd.to_datetime(date_to)]
        return df

    @staticmethod
    def count_incidents(series: pd.Series) -> int:
        return int((series.fillna(0) > 0).sum())

    @staticmethod
    def infer_interval() -> float:
        return _infer_interval_minutes(_load())