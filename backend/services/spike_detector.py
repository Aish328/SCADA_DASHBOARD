import pandas as pd
import numpy as np
from services.data_loader import DataLoader

class SpikeDetector:
    """Detect anomalies and spikes in SCADA data"""
    
    # Threshold multiplier for standard deviation
    THRESHOLD_STD_DEV = 2.5
    
    @staticmethod
    def detect_voltage_spikes(substation: str = None, feeder: str = None) -> list:
        """Detect voltage spikes using statistical method"""
        if substation and feeder:
            df = DataLoader.get_data_by_substation_and_feeder(substation, feeder)
        elif substation:
            df = DataLoader.get_data_by_substation(substation)
        elif feeder:
            df = DataLoader.get_data_by_feeder(feeder)
        else:
            df = DataLoader.get_all_data()
        
        spikes = []
        voltage_cols = ['fvhi', 'fvhd', 'fvli', 'fvld']
        
        for col in voltage_cols:
            if col in df.columns:
                mean = df[col].mean()
                std = df[col].std()
                threshold = mean + (SpikeDetector.THRESHOLD_STD_DEV * std)
                
                spike_rows = df[df[col] > threshold]
                for idx, row in spike_rows.iterrows():
                    spikes.append({
                        'datetime': row['datetime'],
                        'substation': row['substation'],
                        'feeder': row['feeder'],
                        'spike_type': f'voltage_{col}',
                        'value': row[col],
                        'threshold': threshold,
                        'deviation': row[col] - mean
                    })
        
        return spikes
    
    @staticmethod
    def detect_current_spikes(substation: str = None, feeder: str = None) -> list:
        """Detect current spikes using statistical method"""
        if substation and feeder:
            df = DataLoader.get_data_by_substation_and_feeder(substation, feeder)
        elif substation:
            df = DataLoader.get_data_by_substation(substation)
        elif feeder:
            df = DataLoader.get_data_by_feeder(feeder)
        else:
            df = DataLoader.get_all_data()
        
        spikes = []
        current_cols = ['fchi', 'fchd', 'fcli', 'fcld']
        
        for col in current_cols:
            if col in df.columns:
                mean = df[col].mean()
                std = df[col].std()
                threshold = mean + (SpikeDetector.THRESHOLD_STD_DEV * std)
                
                spike_rows = df[df[col] > threshold]
                for idx, row in spike_rows.iterrows():
                    spikes.append({
                        'datetime': row['datetime'],
                        'substation': row['substation'],
                        'feeder': row['feeder'],
                        'spike_type': f'current_{col}',
                        'value': row[col],
                        'threshold': threshold,
                        'deviation': row[col] - mean
                    })
        
        return spikes
    
    @staticmethod
    def detect_all_spikes(substation: str = None, feeder: str = None) -> list:
        """Detect all voltage and current spikes"""
        voltage_spikes = SpikeDetector.detect_voltage_spikes(substation, feeder)
        current_spikes = SpikeDetector.detect_current_spikes(substation, feeder)
        return voltage_spikes + current_spikes
