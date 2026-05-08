from fastapi import APIRouter, Query
from services.data_loader import DataLoader

router = APIRouter()

@router.get("/series")
def get_chart_series(substation: str = None, feeder: str = None, limit: int = Query(12, ge=1, le=50)):
    """Get chart series for SCADA data"""
    if substation and feeder:
        df = DataLoader.get_data_by_substation_and_feeder(substation, feeder)
    elif substation:
        df = DataLoader.get_data_by_substation(substation)
    elif feeder:
        df = DataLoader.get_data_by_feeder(feeder)
    else:
        df = DataLoader.get_all_data()

    if df.empty:
        return {
            "categories": [],
            "voltage1": [],
            "voltage2": [],
            "voltage3": [],
            "voltage4": [],
            "current1": [],
            "current2": [],
            "current3": [],
            "current4": [],
            "trend1": [],
            "trend2": [],
            "trend3": []
        }

    df = df.sort_values("datetime").tail(limit)
    categories = df["datetime"].dt.strftime("%Y-%m-%d %H:%M").tolist()

    voltage1 = df["fvhi"].tolist()
    voltage2 = df["fvhd"].tolist()
    voltage3 = df["fvli"].tolist()
    voltage4 = df["fvld"].tolist()

    current1 = df["fchi"].tolist()
    current2 = df["fchd"].tolist()
    current3 = df["fcli"].tolist()
    current4 = df["fcld"].tolist()

    trend1 = df["fvsm"].tolist()
    trend2 = df["fcsm"].tolist()
    trend3 = ((df["fvsm"] + df["fcsm"]) / 2).tolist()

    return {
        "categories": categories,
        "voltage1": voltage1,
        "voltage2": voltage2,
        "voltage3": voltage3,
        "voltage4": voltage4,
        "current1": current1,
        "current2": current2,
        "current3": current3,
        "current4": current4,
        "trend1": trend1,
        "trend2": trend2,
        "trend3": trend3
    }
