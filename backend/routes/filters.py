from fastapi import APIRouter
from services.data_loader import DataLoader

router = APIRouter()

@router.get("/")
def get_filters():
    """Get available filter options"""
    substations = DataLoader.get_unique_substations()
    all_feeders = DataLoader.get_unique_feeders()
    
    # Group feeders by substation
    feeder_by_substation = {}
    for substation in substations:
        feeder_by_substation[substation] = DataLoader.get_unique_feeders(substation)
    
    return {
        "substations": substations,
        "feeders": all_feeders,
        "feeders_by_substation": feeder_by_substation
    }