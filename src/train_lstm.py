"""
Train and evaluate an LSTM baseline for next-day AQI forecasting,
compared against the XGBoost baseline.
"""

# Note: LSTM is overfitting potentially due to small dataset

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

def make_sequences(X, y, seq_len):
    """Turn flat feature rows into overlapping sequences of length seq_len."""

    X_seq, y_seq = [], []
    for i in range(len(X) - seq_len):
        X_seq.append(X[i:i + seq_len])
        y_seq.append(y[i + seq_len])

    return np.array(X_seq), np.array(y_seq)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=32, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # last timestep's output
        return self.fc(out).squeeze(-1)

def train_lstm():
    df = load_data()
    split_idx = int(len(df) * 0.8)

    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train = scaler_X.fit_transform(train_df[FEATURE_COLS])
    X_test = scaler_X.transform(test_df[FEATURE_COLS])
    y_train = scaler_y.fit_transform(train_df[[TARGET_COL]]).flatten()
    y_test = scaler_y.transform(test_df[[TARGET_COL]]).flatten()

    X_train_seq, y_train_seq = make_sequences(X_train, y_train, SEQ_LEN)
    X_test_seq, y_test_seq = make_sequences(X_test, y_test, SEQ_LEN)

    X_train_t = torch.tensor(X_train_seq, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_seq, dtype=torch.float32)
    X_test_t = torch.tensor(X_test_seq, dtype=torch.float32)
    y_test_t = torch.tensor(y_test_seq, dtype=torch.float32)

    model = LSTMModel(input_size=len(FEATURE_COLS))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    epochs = 100
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        preds = model(X_train_t)
        loss = loss_fn(preds, y_train_t)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs}  Loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        test_preds_scaled = model(X_test_t).numpy()

    # Unscale predictions back to real AQI values
    test_preds = scaler_y.inverse_transform(test_preds_scaled.reshape(-1, 1)).flatten()
    y_test_actual = scaler_y.inverse_transform(y_test_seq.reshape(-1, 1)).flatten()

    mae = mean_absolute_error(y_test_actual, test_preds)
    rmse = np.sqrt(mean_squared_error(y_test_actual, test_preds))
    print(f"LSTM                 MAE: {mae:.2f}  RMSE: {rmse:.2f}")

    return model, test_preds, y_test_actual

if __name__ == "__main__":
    train_lstm()