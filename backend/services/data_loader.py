import pandas as pd
from db.connection import connection

class DataLoader:
    """Load and process SCADA data"""
    
    @staticmethod
    def get_all_data() -> pd.DataFrame:
        """Get all SCADA data"""
        return connection.get_dataframe()
    
    @staticmethod
    def get_data_by_substation(substation: str) -> pd.DataFrame:
        """Get data filtered by substation"""
        df = connection.get_dataframe()
        return df[df['substation'] == substation]
    
    @staticmethod
    def get_data_by_feeder(feeder: str) -> pd.DataFrame:
        """Get data filtered by feeder"""
        df = connection.get_dataframe()
        return df[df['feeder'] == feeder]
    
    @staticmethod
    def get_data_by_substation_and_feeder(substation: str, feeder: str) -> pd.DataFrame:
        """Get data filtered by substation and feeder"""
        df = connection.get_dataframe()
        return df[(df['substation'] == substation) & (df['feeder'] == feeder)]
    
    @staticmethod
    def get_unique_substations() -> list:
        """Get list of unique substations"""
        df = connection.get_dataframe()
        return df['substation'].unique().tolist()
    
    @staticmethod
    def get_unique_feeders(substation: str = None) -> list:
        """Get list of unique feeders, optionally filtered by substation"""
        df = connection.get_dataframe()
        if substation:
            df = df[df['substation'] == substation]
        return df['feeder'].unique().tolist()
    
    @staticmethod
    def get_data_by_date_range(start_date: str, end_date: str) -> pd.DataFrame:
        """Get data within a date range"""
        df = connection.get_dataframe()
        df = df[(df['datetime'] >= start_date) & (df['datetime'] <= end_date)]
        return df
