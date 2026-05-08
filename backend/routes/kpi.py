from fastapi import APIRouter, Query
from services.data_loader import DataLoader

router = APIRouter()

@router.get("/")
def get_kpis(substation: str = None, feeder: str = None):
    """Get KPI metrics for SCADA data"""
    
    # Load data based on filters
    if substation and feeder:
        df = DataLoader.get_data_by_substation_and_feeder(substation, feeder)
    elif substation:
        df = DataLoader.get_data_by_substation(substation)
    elif feeder:
        df = DataLoader.get_data_by_feeder(feeder)
    else:
        df = DataLoader.get_all_data()
    
    # Calculate voltage metrics
    voltage_cols = ['fvhi', 'fvhd', 'fvli', 'fvld', 'fvsm']
    voltage_data = df[voltage_cols].select_dtypes(include=['float64', 'int64'])
    
    # Calculate current metrics
    current_cols = ['fchi', 'fchd', 'fcli', 'fcld', 'fcsm']
    current_data = df[current_cols].select_dtypes(include=['float64', 'int64'])
    
    # Calculate individual field averages
    individual_kpis = {}
    for col in voltage_cols + current_cols:
        if col in df.columns:
            individual_kpis[col] = float(df[col].mean())
    
    return {
        # Individual SCADA metrics
        "fvhi": individual_kpis.get('fvhi', 0),
        "fvhd": individual_kpis.get('fvhd', 0),
        "fvli": individual_kpis.get('fvli', 0),
        "fvld": individual_kpis.get('fvld', 0),
        "fvsm": individual_kpis.get('fvsm', 0),
        "fchi": individual_kpis.get('fchi', 0),
        "fchd": individual_kpis.get('fchd', 0),
        "fcli": individual_kpis.get('fcli', 0),
        "fcld": individual_kpis.get('fcld', 0),
        "fcsm": individual_kpis.get('fcsm', 0),
        # Aggregated metrics
        "avg_feeder_voltage": float(voltage_data.mean().mean()),
        "avg_feeder_current": float(current_data.mean().mean()),
        "max_voltage": float(voltage_data.max().max()),
        "max_current": float(current_data.max().max()),
        "min_voltage": float(voltage_data.min().min()),
        "min_current": float(current_data.min().min()),
        "total_records": len(df),
        "substations": DataLoader.get_unique_substations(),
        "feeders": DataLoader.get_unique_feeders(substation) if substation else DataLoader.get_unique_feeders()
    }