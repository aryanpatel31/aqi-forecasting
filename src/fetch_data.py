"""
Pulls historical AQI data (AirNow API) and weather data (Open-Meteo)
for Chicago, IL and saves raw CSVs to data/raw/.
"""

import os
from dotenv import load_dot_env

load_dotenv()

# Config
CITY = "Chicago, IL"
LATITUDE = 41.8781
LONGITUDE = -87.6298
AIRNOW_ZIP = "60601"
START_DATE = "2024-06-01"
END_DATE = "2026-06-01"

AIRNOW_API_KEY = os.getenv("AIRNOW_API_KEY")
RAW_AQI_PATH = "data/raw/raw_aqi.csv"
RAW_WEATHER_PATH = "data/raw/raw_weather.csv"


def fetch_aqi():
    """Pull historical AQI from AirNow API in monthly chunks. TODO."""
    pass


def fetch_weather():
    """Pull historical weather from Open-Meteo archive API. TODO."""
    pass


if __name__ == "__main__":
    fetch_aqi()
    fetch_weather()
