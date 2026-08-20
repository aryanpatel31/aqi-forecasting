"""
Cleans and merges raw AQI (AirNow) and weather (Open-Meteo) data
into a single daily-resolution dataframe for modeling.
"""

from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_AQI_PATH = BASE_DIR / "data/raw/raw_aqi.csv"
RAW_WEATHER_PATH = BASE_DIR / "data/raw/raw_weather.csv"
MERGED_PATH = BASE_DIR / "data/processed/merged_daily.csv"

def clean_aqi():
    """Collapse AQI to one row per day: max AQI across pollutants."""
    
    df = pd.read_csv(RAW_AQI_PATH)
    daily = df.groupby("DateObserved")["AQI"].max().reset_index()
    daily.rename(columns={"DateObserved": "date"}, inplace=True)
    daily["date"] = pd.to_datetime(daily["date"]).dt.date

    return daily


def clean_weather():
    """Aggregate hourly weather to daily summaries."""

    df = pd.read_csv(RAW_WEATHER_PATH)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["date"] = df["datetime"].dt.date

    daily = df.groupby("date").agg({
        "temperature_2m": "mean",
        "relative_humidity_2m": "mean",
        "wind_speed_10m": "max",
        "wind_direction_10m": "mean",
        "surface_pressure": "mean",
        "precipitation": "sum",
    }).reset_index()

    return daily

def merge_and_save():
    """Merge cleaned AQI and weather on date, save to processed/."""

    aqi_daily = clean_aqi()
    weather_daily = clean_weather()

    merged = pd.merge(weather_daily, aqi_daily, on="date", how="inner")

    print(f"AQI rows: {len(aqi_daily)}")
    print(f"Weather rows: {len(weather_daily)}")
    print(f"Merged rows: {len(merged)}")
    print(f"Missing values:\n{merged.isnull().sum()}")

    MERGED_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(MERGED_PATH, index=False)
    print(f"Saved merged dataset to {MERGED_PATH}")

    return merged

if __name__ == "__main__":
    merge_and_save()


