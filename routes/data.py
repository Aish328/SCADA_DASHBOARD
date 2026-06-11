from fastapi import APIRouter, Query
from services.data_loader import DataLoader

router = APIRouter()

EMPTY = {
    "categories": [],
    # actual phase values
    "vry": [], "vyb": [], "vbr": [], "v_avg": [],
    "ir":  [], "iy":  [], "ib":  [], "i_avg": [],
    # anomaly flags (1/0) — for shading
    "fvhi_flag": [], "fvli_flag": [],
    "fchi_flag": [], "fcli_flag": [],
    # per-row episode duration (minutes) — non-zero only during anomaly rows
    "fvhd": [], "fvld": [],
    "fchd": [], "fcld": [],
    # thresholds (scalar, sent once for reference lines)
    "voltage_high": 250, "voltage_low": 10,
    "current_high": 100, "current_low":  50,
}


@router.get("/series")
def get_chart_series(
    substation: str = None,
    feeder: str = None,
    limit: int = Query(96, ge=1, le=2000),
):
    df = DataLoader.get_all_data()
    df = DataLoader.filter_data(df, substation, feeder)
    if df.empty:
        return EMPTY

    df = df.sort_values("datetime").tail(limit)
    cats = df["datetime"].dt.strftime("%Y-%m-%d %H:%M").tolist()

    def col(name):
        if name in df.columns:
            return df[name].round(3).tolist()
        return [None] * len(df)

    def flag(name):
        if name in df.columns:
            return df[name].fillna(0).astype(int).tolist()
        return [0] * len(df)

    # Compute per-phase averages
    volt_cols = [c for c in ["vry", "vyb", "vbr"] if c in df.columns]
    curr_cols = [c for c in ["ir", "iy", "ib"] if c in df.columns]
    v_avg = df[volt_cols].mean(axis=1).round(3).tolist() if volt_cols else [None]*len(df)
    i_avg = df[curr_cols].mean(axis=1).round(3).tolist() if curr_cols else [None]*len(df)

    from services.data_loader import VOLTAGE_HIGH, VOLTAGE_LOW, CURRENT_HIGH, CURRENT_LOW

    return {
        "categories": cats,
        # phase values
        "vry": col("vry"), "vyb": col("vyb"), "vbr": col("vbr"), "v_avg": v_avg,
        "ir":  col("ir"),  "iy":  col("iy"),  "ib":  col("ib"),  "i_avg": i_avg,
        # anomaly flags
        "fvhi_flag": flag("fvhi"), "fvli_flag": flag("fvli"),
        "fchi_flag": flag("fchi"), "fcli_flag": flag("fcli"),
        # per-row episode duration
        "fvhd": col("fvhd"), "fvld": col("fvld"),
        "fchd": col("fchd"), "fcld": col("fcld"),
        # threshold reference lines
        "voltage_high": VOLTAGE_HIGH, "voltage_low": VOLTAGE_LOW,
        "current_high": CURRENT_HIGH, "current_low":  CURRENT_LOW,
    }