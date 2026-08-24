"""
Live feature pipeline — runs hourly via GitHub Actions.

Fetches current air pollution + weather for Islamabad from OpenWeather
(both free, no credit card needed for these endpoints), engineers
features, and writes one row to the Hopsworks Feature Group
'aqi_features'.

Environment variables required (set as GitHub Secrets in production,
or in a local .env file for manual testing):
    OPENWEATHER_API_KEY
    HOPSWORKS_API_KEY
    HOPSWORKS_PROJECT
"""

import os
import sys
from datetime import datetime, timezone

import requests
import pandas as pd
import hopsworks
from dotenv import load_dotenv

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.city_config import CITY_NAME, LATITUDE, LONGITUDE
from utils.feature_eng import pm25_to_aqi, add_time_features

load_dotenv()

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT")


def fetch_openweather_pollution() -> dict:
    """Current air pollution (pollutant concentrations) for Islamabad."""
    if not OPENWEATHER_API_KEY:
        raise RuntimeError("OPENWEATHER_API_KEY is not set")

    url = "http://api.openweathermap.org/data/2.5/air_pollution"
    params = {"lat": LATITUDE, "lon": LONGITUDE, "appid": OPENWEATHER_API_KEY}
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    payload = resp.json()

    if "list" not in payload or not payload["list"]:
        raise RuntimeError(f"OpenWeather pollution response missing data: {payload}")

    return payload["list"][0]  # current reading


def fetch_openweather_weather() -> dict:
    """Current weather for Islamabad."""
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"lat": LATITUDE, "lon": LONGITUDE, "appid": OPENWEATHER_API_KEY, "units": "metric"}
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    payload = resp.json()

    if "main" not in payload:
        raise RuntimeError(f"OpenWeather weather response missing data: {payload}")

    return payload


def build_feature_row(pollution: dict, weather: dict) -> pd.DataFrame:
    """Combine pollution + weather responses into one feature row matching
    the Hopsworks Feature Group schema."""
    comp = pollution.get("components", {})
    pm25 = comp.get("pm2_5")
    aqi = pm25_to_aqi(pm25)

    ingestion_time = pd.Timestamp.now(tz="UTC")
    openweather_timestamp = pd.to_datetime(pollution.get("dt"), unit="s", utc=True)

    row = {
        "city": CITY_NAME,
        "timestamp": ingestion_time,
        "openweather_timestamp": openweather_timestamp,
        "aqi": aqi,
        "pm25_median": pm25,
        "pm10_median": comp.get("pm10"),
        "o3_median": comp.get("o3"),
        "no2_median": comp.get("no2"),
        "so2_median": comp.get("so2"),
        "co_median": comp.get("co"),
        "temperature_median": weather.get("main", {}).get("temp"),
        "humidity_median": weather.get("main", {}).get("humidity"),
        "pressure_median": weather.get("main", {}).get("pressure"),
        "wind_speed_median": weather.get("wind", {}).get("speed"),
    }

    df = pd.DataFrame([row])
    df = add_time_features(df, timestamp_col="timestamp")

    # Explicit dtype enforcement — must exactly match
    # push_backfill_to_hopsworks.py's casting, otherwise Hopsworks will
    # reject inserts with a schema-mismatch error (this happened once
    # already: OpenWeather sometimes returns whole-number humidity/
    # pressure, which pandas silently infers as int rather than float).
    float_cols = [
        "pm25_median", "pm10_median", "o3_median", "no2_median",
        "so2_median", "co_median", "temperature_median",
        "humidity_median", "pressure_median", "wind_speed_median",
    ]
    int_cols = ["aqi", "hour_of_day", "day_of_week", "month", "is_weekend"]

    for col in float_cols:
        df[col] = df[col].astype("float64")
    for col in int_cols:
        df[col] = df[col].astype("int64")
    df["city"] = df["city"].astype(str)

    return df


def push_to_hopsworks(df: pd.DataFrame):
    if not HOPSWORKS_API_KEY or not HOPSWORKS_PROJECT:
        raise RuntimeError("HOPSWORKS_API_KEY / HOPSWORKS_PROJECT not set")

    project = hopsworks.login(
        project=HOPSWORKS_PROJECT,
        host="eu-west.cloud.hopsworks.ai",
        port=443,
        api_key_value=HOPSWORKS_API_KEY,
    )
    fs = project.get_feature_store()

    fg = fs.get_or_create_feature_group(
        name="aqi_features",
        version=1,
        primary_key=["city", "timestamp"],
        event_time="timestamp",
        description="Hourly AQI, pollutant and weather features for Islamabad (OpenWeather)",
        time_travel_format="HUDI",
    )
    fg.insert(df)


def main():
    print(f"[{datetime.now(timezone.utc).isoformat()}] Fetching OpenWeather data for {CITY_NAME}...")
    pollution = fetch_openweather_pollution()
    weather = fetch_openweather_weather()

    df = build_feature_row(pollution, weather)
    print("Built feature row:")
    print(df.to_string(index=False))

    push_to_hopsworks(df)
    print("Inserted into Hopsworks Feature Group 'aqi_features'.")


if __name__ == "__main__":
    main()