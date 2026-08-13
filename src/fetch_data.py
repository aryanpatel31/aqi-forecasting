"""
Pulls historical AQI data (AirNow API) and weather data (Open-Meteo)
for Chicago, IL and saves raw CSVs to data/raw/.
"""

import os
import requests
import pandas as pd
from dotenv import load_dotenv
import time
from datetime import datetime, timedelta

load_dotenv()

# Config
CITY = "Chicago, IL"
LATITUDE = 41.8781
LONGITUDE = -87.6298
AIRNOW_ZIP = "60601"
START_DATE = "2025-06-01"
END_DATE = "2026-06-01"

AIRNOW_API_KEY = os.getenv("AIRNOW_API_KEY")
RAW_AQI_PATH = "data/raw/raw_aqi.csv"
RAW_WEATHER_PATH = "data/raw/raw_weather.csv"


def fetch_aqi():
    """Pull historical AQI from AirNow API in monthly chunks."""
    start = datetime.strptime(START_DATE, "%Y-%m-%d")
    end = datetime.strptime(END_DATE, "%Y-%m-%d")

    all_records = []
    current = start

    while current <= end:
        date_str = current.strftime("%Y-%m-%dT00-0000")
        url = "https://www.airnowapi.org/aq/observation/zipCode/historical/"
        params = {
            "format": "application/json",
            "zipCode": AIRNOW_ZIP,
            "date": date_str,
            "distance": 25,
            "API_KEY": AIRNOW_API_KEY,
        }

        response = requests.get(url, params=params)
        if response.status_code == 200:
            all_records.extend(response.json())
        else:
            print(f"Failed on {date_str}: {response.status_code}")

        current += timedelta(days=1)
        time.sleep(0.2)  # staying under rate limit

    df = pd.DataFrame(all_records)
    df.to_csv(RAW_AQI_PATH, index=False)
    print(f"Saved {len(df)} AQI rows to {RAW_AQI_PATH}")
    return df


def fetch_weather():
    """Pull historical weather from Open-Meteo archive API."""

    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m,surface_pressure,precipitation",
        "timezone": "America/Chicago",
    }

    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()["hourly"]

    df = pd.DataFrame(data)
    df.rename(columns={"time": "datetime"}, inplace=True)
    df.to_csv(RAW_WEATHER_PATH, index=False)
    print(f"Saved {len(df)} weather rows to {RAW_WEATHER_PATH}")
    return df
    


if __name__ == "__main__":
    fetch_aqi()
    fetch_weather()
