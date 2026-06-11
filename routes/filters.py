from fastapi import APIRouter
from services.data_loader import DataLoader

router = APIRouter()

@router.get("/")
def get_filters():
    substations = DataLoader.get_unique_substations()
    feeder_by_substation = {s: DataLoader.get_unique_feeders(s) for s in substations}
    return {
        "substations": substations,
        "feeders": DataLoader.get_unique_feeders(),
        "feeders_by_substation": feeder_by_substation,
    }
