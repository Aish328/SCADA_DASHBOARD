import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch


# ==========================================================
# Fault Classes
# ==========================================================
FAULT_CLASSES = {
    0: "Normal",
    1: "Voltage Sag",
    2: "Voltage Surge",
    3: "Current Imbalance",
    4: "Overload",
    5: "Transformer Stress"
}


FAULT_COLORS = {
    0: "#2ecc71",   # green
    1: "#e74c3c",   # red
    2: "#f39c12",   # orange
    3: "#3498db",   # blue
    4: "#9b59b6",   # purple
    5: "#c0392b"    # dark red
}


# ==========================================================
# Fault Thresholds (scaled values)
# ==========================================================

# Voltage thresholds
# Only trigger at extreme voltage levels
V_SAG_LIMIT = 0.05
V_SURGE_LIMIT = 0.95

# Current imbalance threshold
IMBAL_THRESHOLD = 0.04

# Load thresholds
OVERLOAD_LIMIT = 0.15

# Transformer stress thresholds
XFMR_LOAD_LIM = 0.20
XFMR_CURR_LIM = 0.75

# Fault persistence
# Require fault to exist for N consecutive samples
FAULT_WINDOW = 5


class SmartGridDataLoader:
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None
        self.scaler = None

    # ======================================================
    # Load CSV Data
    # ======================================================
    def load_data(self):

        self.df = pd.read_csv(
            self.filepath,
            sep=',',
            encoding='latin1',
            engine='python',
            na_values=['?'],
            on_bad_lines='skip'
        )

        # Remove extra spaces in column names
        self.df.columns = self.df.columns.str.strip()

        print("\nColumns in Dataset:\n")
        print(self.df.columns.tolist())

        return self.df

    # ======================================================
    # Data Cleaning + Feature Engineering
    # ======================================================
    def preprocess(self):

        # Convert Time column to datetime
        self.df['Time'] = pd.to_datetime(
            self.df['Time'],
            errors='coerce'
        )

        # Remove invalid timestamps
        self.df = self.df.dropna(subset=['Time'])

        # Set Time as index for time-series operations
        self.df.set_index('Time', inplace=True)

        features = [
            'IR',
            'IY',
            'IB',
            'VRY',
            'VYB',
            'VBR',
            'Active Load'
        ]

        # --------------------------------------------------
        # Clean numeric values
        # Extract only numeric values from strings
        # Example: "45A" -> 45
        # --------------------------------------------------
        for col in features:

            self.df[col] = (
                self.df[col]
                .astype(str)
                .str.extract(r'([-+]?\d*\.?\d+)')[0]
                .astype(float)
            )

        # Forward fill missing values
        self.df = self.df.ffill()

        # --------------------------------------------------
        # Clip extreme outliers (1% - 99%)
        # Helps model stability
        # --------------------------------------------------
        for col in features:

            q1 = self.df[col].quantile(0.01)
            q99 = self.df[col].quantile(0.99)

            self.df[col] = self.df[col].clip(q1, q99)

        # --------------------------------------------------
        # Add Time Features
        # --------------------------------------------------
        self.df['hour'] = self.df.index.hour
        self.df['day'] = self.df.index.dayofweek

        return self.df

    # ======================================================
    # Feature Scaling
    # ======================================================
    def scale_data(self):

        features = [
            'IR',
            'IY',
            'IB',
            'VRY',
            'VYB',
            'VBR',
            'Active Load',
            'hour',
            'day'
        ]

        self.scaler = MinMaxScaler()

        scaled = self.scaler.fit_transform(
            self.df[features]
        )

        return scaled

    # ======================================================
    # Generate Fault Labels
    # ======================================================
    def generate_fault_labels(self, data):

        labels = np.zeros(len(data), dtype=np.int64)

        # ---------------------------------------------
        # Mean voltage
        # Prevents phase oscillation misfires
        # ---------------------------------------------
        voltage_mean = (
            data[:, 3] +
            data[:, 4] +
            data[:, 5]
        ) / 3

        # ---------------------------------------------
        # Smooth voltage signal
        # Helps reduce square-wave quantization issue
        # ---------------------------------------------
        voltage_smooth = pd.Series(
            voltage_mean
        ).rolling(
            window=5,
            center=True,
            min_periods=1
        ).mean()

        # ---------------------------------------------
        # Create fault masks
        # ---------------------------------------------
        surge_mask = (
            voltage_smooth > V_SURGE_LIMIT
        )

        sag_mask = (
            voltage_smooth < V_SAG_LIMIT
        )

        # ---------------------------------------------
        # Persistence logic
        # Fault must exist for N samples
        # ---------------------------------------------
        surge_fault = (
            surge_mask
            .rolling(FAULT_WINDOW)
            .sum() >= FAULT_WINDOW
        )

        sag_fault = (
            sag_mask
            .rolling(FAULT_WINDOW)
            .sum() >= FAULT_WINDOW
        )

        # ---------------------------------------------
        # Label generation
        # ---------------------------------------------
        for i in range(len(data)):

            IR = data[i, 0]
            IY = data[i, 1]
            IB = data[i, 2]
            LOAD = data[i, 6]

            # Mean phase current
            current_mean = (IR + IY + IB) / 3

            # Max deviation from mean
            imbalance = max(
                abs(IR - current_mean),
                abs(IY - current_mean),
                abs(IB - current_mean)
            )

            # -----------------------------------------
            # Priority Order Matters
            # -----------------------------------------

            # Voltage faults
            if sag_fault.iloc[i]:
                labels[i] = 1

            elif surge_fault.iloc[i]:
                labels[i] = 2

            # Current imbalance
            elif imbalance > IMBAL_THRESHOLD:
                labels[i] = 3

            # Transformer stress
            # Must come BEFORE overload
            elif (
                LOAD > XFMR_LOAD_LIM
                and (
                    IR > XFMR_CURR_LIM
                    or IY > XFMR_CURR_LIM
                    or IB > XFMR_CURR_LIM
                )
            ):
                labels[i] = 5

            # Overload
            elif LOAD > OVERLOAD_LIMIT:
                labels[i] = 4

        return labels

    # ======================================================
    # Create Sequences
    # ======================================================
    def create_sequences(
        self,
        data,
        labels,
        seq_length=48
    ):

        X = []
        y = []

        for i in range(len(data) - seq_length):

            # Input sequence
            X.append(
                data[i:i + seq_length]
            )

            # Label at sequence end
            y.append(
                labels[i + seq_length]
            )

        return np.array(X), np.array(y)

    # ======================================================
    # Main Pipeline
    # ======================================================
    def get_processed_data(
        self,
        seq_length=48
    ):

        self.load_data()

        self.preprocess()

        scaled_data = self.scale_data()

        labels = self.generate_fault_labels(
            scaled_data
        )

        X, _ = self.create_sequences(
            scaled_data,
            labels,
            seq_length
        )

        X = torch.tensor(
            X,
            dtype=torch.float32
        )

        return X