import pandas as pd
from typing import Optional
from services.data_loader import DataLoader


def _to_records(df: pd.DataFrame, cols: list, spike_type: str) -> list:
    records = []
    for _, row in df.iterrows():
        record = {
            "spike_type": spike_type,
            "substation": row.get("substation", ""),
            "feeder": row.get("feeder", ""),
            "datetime": str(row.get("datetime", "")),
        }
        for col in cols:
            if col in row:
                record[col] = round(float(row[col]), 3) if pd.notna(row[col]) else None
        records.append(record)
    return records


class SpikeDetector:

    VOLTAGE_HIGH = 250
    VOLTAGE_LOW = 200
    CURRENT_HIGH = 100
    CURRENT_LOW = 50

    @staticmethod
    def detect_voltage_spikes(substation: Optional[str] = None, feeder: Optional[str] = None) -> list:
        df = DataLoader.get_all_data()
        df = DataLoader.filter_data(df, substation, feeder)
        volt_cols = [c for c in ["vry", "vyb", "vbr"] if c in df.columns]
        if not volt_cols:
            return []
        mask = (df[volt_cols].max(axis=1) > SpikeDetector.VOLTAGE_HIGH) | \
               (df[volt_cols].min(axis=1) < SpikeDetector.VOLTAGE_LOW)
        spikes = df[mask].copy()
        return _to_records(spikes, volt_cols, "voltage")

    @staticmethod
    def detect_current_spikes(substation: Optional[str] = None, feeder: Optional[str] = None) -> list:
        df = DataLoader.get_all_data()
        df = DataLoader.filter_data(df, substation, feeder)
        curr_cols = [c for c in ["ir", "iy", "ib"] if c in df.columns]
        if not curr_cols:
            return []
        mask = (df[curr_cols].max(axis=1) > SpikeDetector.CURRENT_HIGH) | \
               (df[curr_cols].min(axis=1) < SpikeDetector.CURRENT_LOW)
        spikes = df[mask].copy()
        return _to_records(spikes, curr_cols, "current")

    @staticmethod
    def detect_all_spikes(substation: Optional[str] = None, feeder: Optional[str] = None) -> list:
        v = SpikeDetector.detect_voltage_spikes(substation, feeder)
        c = SpikeDetector.detect_current_spikes(substation, feeder)
        return v + c
