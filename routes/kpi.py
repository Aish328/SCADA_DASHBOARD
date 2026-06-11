from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import StreamingResponse
from typing import Optional
import pandas as pd
import io
from services.data_loader import DataLoader, VOLTAGE_HIGH, VOLTAGE_LOW, CURRENT_HIGH, CURRENT_LOW

router = APIRouter()


def _empty_kpis():
    return {
        # episode counts
        "fvhi_count": 0, "fvli_count": 0,
        "fchi_count": 0, "fcli_count": 0,
        # total anomaly duration (sum of all episode durations in minutes)
        "fvhd_total": 0, "fvld_total": 0,
        "fchd_total": 0, "fcld_total": 0,
        # average V/I during anomaly windows
        "fvhi_avg_v": 0, "fvli_avg_v": 0,
        "fchi_avg_i": 0, "fcli_avg_i": 0,
        # overall averages
        "avg_voltage": 0, "avg_current": 0,
        "max_voltage": 0, "max_current": 0,
        "min_voltage": 0, "min_current": 0,
        "total_records": 0,
        "reading_interval_min": 0,
        "substations": [], "feeders": [],
        "thresholds": {
            "voltage_high": VOLTAGE_HIGH, "voltage_low": VOLTAGE_LOW,
            "current_high": CURRENT_HIGH, "current_low":  CURRENT_LOW,
        },
    }


@router.get("/")
def get_kpis(substation: Optional[str] = None, feeder: Optional[str] = None , limit: int = 96):
    df = DataLoader.get_all_data()
    df = DataLoader.filter_data(df, substation, feeder)
    df = df.sort_values("datetime").tail(limit)
    # if limit:
    #     df = df.sort_values("datetime").tail(limit)
    if df.empty:
        return _empty_kpis()

    interval = DataLoader.infer_interval()

    volt_cols = [c for c in ["vry", "vyb", "vbr"] if c in df.columns]
    curr_cols = [c for c in ["ir", "iy", "ib"] if c in df.columns]

    vdf = df[volt_cols] if volt_cols else pd.DataFrame()
    cdf = df[curr_cols] if curr_cols else pd.DataFrame()

    # ── Episode counts: sum of max episode id across all feeder groups ───────
    def total_eps(col):
        if col not in df.columns: return 0
        return int(df.groupby(["substation","feeder"])[col].max().sum())

    # ── Total anomaly duration: count anomaly rows * interval ────────────────
    # def total_dur(flag_col):
    #     if flag_col not in df.columns: return 0.0
    #     return round(float((df[flag_col] > 0).sum() * interval), 1)
    def total_dur(ep_col, dur_col):
        if ep_col not in df.columns or dur_col not in df.columns:
            return 0.0

        ep_df = df[df[ep_col] > 0]

        if ep_df.empty:
            return 0.0

        return round(
            float(
                ep_df.groupby(ep_col)[dur_col]
                .max()
                .sum()
            ),
            1
            )
    # ── Average V/I during anomaly windows ───────────────────────────────────
    def anomaly_avg(flag_col, value_col):
        if flag_col not in df.columns or value_col not in df.columns: return 0.0
        mask = df[flag_col] > 0
        if not mask.any(): return 0.0
        return round(float(df.loc[mask, value_col].mean()), 3)

    return {
        "fvhi_count":   total_eps("fvhi_count"),
        "fvli_count":   total_eps("fvli_count"),
        "fchi_count":   total_eps("fchi_count"),
        "fcli_count":   total_eps("fcli_count"),
        "fvhd_total": total_dur("fvhi_ep", "fvhd"),
        "fvld_total": total_dur("fvli_ep", "fvld"),
        "fchd_total": total_dur("fchi_ep", "fchd"),
        "fcld_total": total_dur("fcli_ep", "fcld"),
        "fvhi_avg_v":   anomaly_avg("fvhi", "fvsm"),
        "fvli_avg_v":   anomaly_avg("fvli", "fvsm"),
        "fchi_avg_i":   anomaly_avg("fchi", "fcsm"),
        "fcli_avg_i":   anomaly_avg("fcli", "fcsm"),
        "avg_voltage":  round(float(vdf.mean().mean()), 3) if not vdf.empty else 0,
        "avg_current":  round(float(cdf.mean().mean()), 3) if not cdf.empty else 0,
        "max_voltage":  round(float(vdf.max().max()), 3)  if not vdf.empty else 0,
        "max_current":  round(float(cdf.max().max()), 3)  if not cdf.empty else 0,
        "min_voltage":  round(float(vdf.min().min()), 3)  if not vdf.empty else 0,
        "min_current":  round(float(cdf.min().min()), 3)  if not cdf.empty else 0,
        "total_records": len(df),
        "reading_interval_min": round(interval, 1),
        "substations":  DataLoader.get_unique_substations(),
        "feeders":      (DataLoader.get_unique_feeders(substation)
                         if substation else DataLoader.get_unique_feeders()),
        "thresholds": {
            "voltage_high": VOLTAGE_HIGH, "voltage_low": VOLTAGE_LOW,
            "current_high": CURRENT_HIGH, "current_low":  CURRENT_LOW,
        },
    }


@router.get("/timeseries")
def kpi_timeseries(
    substation: Optional[str] = None,
    feeder: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    resample: Optional[str] = None,
):
    df = DataLoader.get_all_data()
    df = DataLoader.filter_data(df, substation, feeder, date_from, date_to)
    df = df.sort_values("datetime")
    if df.empty:
        raise HTTPException(status_code=404, detail="No data found")
    if resample:
        df = df.set_index("datetime").resample(resample).mean(numeric_only=True).reset_index()
    df["datetime"] = df["datetime"].astype(str)
    return df.where(pd.notnull(df), None).to_dict(orient="records")


@router.get("/daily")
def kpi_daily(substation: Optional[str] = None, feeder: Optional[str] = None):
    df = DataLoader.get_all_data()
    df = DataLoader.filter_data(df, substation, feeder)
    df = df.copy()
    df["date"] = pd.to_datetime(df["datetime"]).dt.date
    interval = DataLoader.infer_interval()
    rows = []

    for (sub, fd, dt), g in df.groupby(["substation", "feeder", "date"]):
        g = g.sort_values("datetime")

        def ep(col):
            return int(g[col].max()) if col in g.columns else 0

        def dur_min(flag_col):
            if flag_col not in g.columns: return 0.0
            return round(float((g[flag_col] > 0).sum() * interval), 1)

        rows.append({
            "date": str(dt), "substation": sub, "feeder": fd,
            "fvhi_episodes": ep("fvhi_count"),
            "fvhd_min":      dur_min("fvhi"),
            "fvli_episodes": ep("fvli_count"),
            "fvld_min":      dur_min("fvli"),
            "fchi_episodes": ep("fchi_count"),
            "fchd_min":      dur_min("fchi"),
            "fcli_episodes": ep("fcli_count"),
            "fcld_min":      dur_min("fcli"),
            "fvsm_avg": round(float(g["fvsm"].mean()), 3) if "fvsm" in g.columns else 0,
            "fcsm_avg": round(float(g["fcsm"].mean()), 3) if "fcsm" in g.columns else 0,
        })
    return rows


@router.get("/export")
def export_csv(substation: Optional[str] = None, feeder: Optional[str] = None):
    df = DataLoader.get_all_data()
    df = DataLoader.filter_data(df, substation, feeder)
    if df.empty:
        raise HTTPException(status_code=404, detail="No data to export")
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    filename = f"kpi_{substation or 'all'}_{feeder or 'all'}.csv".replace(" ", "_")
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )