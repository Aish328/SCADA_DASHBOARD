import pandas as pd
from pathlib import Path

class DataConnection:
    """Handle connection and caching of SCADA data"""
    
    _instance = None
    _df = None
    CSV_PATH = r"C:\Users\sharika\Desktop\SCADA_DASHBOARD\data\scada_dashboard_formatted_data.csv"
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DataConnection, cls).__new__(cls)
        return cls._instance
    
    def get_dataframe(self) -> pd.DataFrame:
        """Get the data frame, loading from CSV if not already cached"""
        if self._df is None:
            self._df = self._load_data()
        return self._df
    
    def _load_data(self) -> pd.DataFrame:
        """Load data from CSV"""
        df = pd.read_csv(self.CSV_PATH)
        # Convert datetime column to datetime type
        df['datetime'] = pd.to_datetime(df['datetime'])
        # Convert column names to lowercase for consistency
        df.columns = df.columns.str.lower()
        return df
    
    def refresh(self):
        """Refresh data from CSV"""
        self._df = None
        return self.get_dataframe()

# Singleton instance
connection = DataConnection()
