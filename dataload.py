import pandas as pd
import numpy as np
import re


class FeederPreprocessor:

    def __init__(self):
        pass

    @staticmethod
    def extract_numeric(x):
        """Extract numeric value from strings like '11.2 kV', '863 kW'."""
        if pd.isna(x):
            return np.nan

        nums = re.findall(r'[-+]?\d*\.?\d+', str(x))
        return float(nums[0]) if nums else np.nan

    @staticmethod
    def convert_load_to_mw(val):
        """
        Convert Active Load to MW.
        Supports:
        875 kW
        1.2 MW
        """

        if pd.isna(val):
            return np.nan

        val_str = str(val)

        num = FeederPreprocessor.extract_numeric(val_str)

        if "kw" in val_str.lower():
            return num / 1000

        return num

    @staticmethod
    def convert_voltage_to_kv(val):
        """
        Convert voltages to kV.
        Supports:
        11.2 kV
        11200 V
        """

        if pd.isna(val):
            return np.nan

        val_str = str(val)
        num = FeederPreprocessor.extract_numeric(val_str)

        if " v" in val_str.lower() and "kv" not in val_str.lower():
            num = num / 1000

        return num

    def preprocess(self, file_path):

        df = pd.read_csv(file_path)

        # ----------------------
        # Standardize columns
        # ----------------------

        df.columns = (
            df.columns
            .str.strip()
            .str.upper()
            .str.replace(" ", "_")
        )

        # Find feeder column
        feeder_col = next(
            col for col in df.columns
            if "FEEDER" in col and "CODE" not in col
        )

        # Time column
        time_col = next(
            col for col in df.columns
            if "TIME" in col
        )

        # Parse datetime
        df[time_col] = pd.to_datetime(
            df[time_col],
            errors="coerce"
        )

        # ----------------------
        # Current columns
        # ----------------------

        current_cols = ["IR", "IY", "IB"]

        for col in current_cols:
            df[col] = df[col].apply(self.extract_numeric)

        # ----------------------
        # Voltage columns
        # ----------------------

        voltage_cols = ["VRY", "VYB", "VBR"]

        for col in voltage_cols:
            df[col] = df[col].apply(self.convert_voltage_to_kv)

        # Average voltage
        df["AVG_VOLTAGE_KV"] = df[voltage_cols].mean(axis=1)

        # ----------------------
        # Active Load
        # ----------------------

        load_col = next(
            col for col in df.columns
            if "ACTIVE" in col
        )

        df["ACTIVE_LOAD_MW"] = df[load_col].apply(
            self.convert_load_to_mw
        )

        # ----------------------
        # Remove negative values
        # ----------------------

        numeric_cols = [
            "IR",
            "IY",
            "IB",
            "VRY",
            "VYB",
            "VBR",
            "AVG_VOLTAGE_KV",
            "ACTIVE_LOAD_MW"
        ]

        for col in numeric_cols:
            df.loc[df[col] < 0, col] = np.nan

        df.dropna(inplace=True)

        # =====================================================
        # ADAPTIVE SAG / SURGE THRESHOLDS
        # =====================================================

        # Compute nominal voltage per feeder
        feeder_nominal = (
            df.groupby(feeder_col)["AVG_VOLTAGE_KV"]
            .median()
            .to_dict()
        )

        df["NOMINAL_VOLTAGE"] = df[feeder_col].map(
            feeder_nominal
        )

        # IEEE-like adaptive thresholds
        df["SAG_THRESHOLD"] = (
            0.90 * df["NOMINAL_VOLTAGE"]
        )

        df["SURGE_THRESHOLD"] = (
            1.10 * df["NOMINAL_VOLTAGE"]
        )

        # Voltage state
        conditions = [
            df["AVG_VOLTAGE_KV"] < df["SAG_THRESHOLD"],
            df["AVG_VOLTAGE_KV"] > df["SURGE_THRESHOLD"]
        ]

        choices = [
            "Voltage Sag",
            "Voltage Surge"
        ]

        df["VOLTAGE_STATUS"] = np.select(
            conditions,
            choices,
            default="Normal"
        )

        # ----------------------
        # Current imbalance
        # ----------------------

        df["CURRENT_IMBALANCE"] = (
            df[current_cols].max(axis=1)
            - df[current_cols].min(axis=1)
        )

        # ----------------------
        # Calendar features
        # ----------------------

        df["HOUR"] = df[time_col].dt.hour
        df["DAY"] = df[time_col].dt.day
        df["MONTH"] = df[time_col].dt.month
        df["DAY_OF_WEEK"] = df[time_col].dt.dayofweek

        return df


# ==================================================
# Usage
# ==================================================

