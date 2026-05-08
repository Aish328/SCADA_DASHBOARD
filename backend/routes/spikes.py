from fastapi import APIRouter, Query
from services.spike_detector import SpikeDetector

router = APIRouter()

@router.get("/")
def get_spikes(spike_type: str = Query("all", enum=["all", "voltage", "current"]), 
               substation: str = None, 
               feeder: str = None):
    """Get detected spikes in SCADA data"""
    
    if spike_type == "voltage":
        spikes = SpikeDetector.detect_voltage_spikes(substation, feeder)
    elif spike_type == "current":
        spikes = SpikeDetector.detect_current_spikes(substation, feeder)
    else:
        spikes = SpikeDetector.detect_all_spikes(substation, feeder)
    
    return {
        "spike_type": spike_type,
        "total_spikes": len(spikes),
        "spikes": spikes
    }