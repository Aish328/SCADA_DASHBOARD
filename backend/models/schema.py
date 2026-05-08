from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class SCADARecord(BaseModel):
    """Schema for a single SCADA data record"""
    datetime: datetime
    substation: str
    feeder: str
    fvhi: float  # Feeder Voltage High
    fvhd: float  # Feeder Voltage High Deviation
    fvli: float  # Feeder Voltage Low
    fvld: float  # Feeder Voltage Low Deviation
    fvsm: float  # Feeder Voltage Sum
    fchi: float  # Feeder Current High
    fchd: float  # Feeder Current High Deviation
    fcli: float  # Feeder Current Low
    fcld: float  # Feeder Current Low Deviation
    fcsm: float  # Feeder Current Sum

class KPIResponse(BaseModel):
    """Schema for KPI response"""
    avg_feeder_voltage: float
    avg_feeder_current: float
    max_voltage: float
    max_current: float
    min_voltage: float
    min_current: float
    total_records: int
    substations: list
    feeders: list

class SpikeResponse(BaseModel):
    """Schema for spike detection response"""
    datetime: datetime
    substation: str
    feeder: str
    spike_type: str
    value: float
    threshold: float
