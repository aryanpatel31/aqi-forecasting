"""
Train and evaluate an LSTM baseline for next-day AQI forecasting,
compared against the XGBoost baseline.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error


BASE_DIR = Path(__file__).resolve().parent.parent
FEATURES_PATH = BASE_DIR / "data/processed/features.csv"

FEATURE_COLS = [
    "temperature_2m", "relative_humidity_2m", "wind_speed_10m",
    "wind_direction_10m", "surface_pressure", "precipitation",
    "aqi_lag_1", "aqi_lag_2", "aqi_lag_3",
    "aqi_roll_3", "aqi_roll_7",
    "month", "day_of_week", "day_of_year",
]
TARGET_COL = "target_aqi_next_day"
SEQ_LEN = 7

def load_data():
    df = pd.read_csv(FEATURES_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df
