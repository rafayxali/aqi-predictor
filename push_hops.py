
import os
import sys

import pandas as pd
import hopsworks
from dotenv import load_dotenv

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.city_config import CITY_NAME

load_dotenv()

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT")

DATA_PATH = "data.csv"


RAW_COLUMNS = [
    "city",
    "timestamp",
    "openweather_timestamp",
    "aqi",
    "pm25_median",
    "pm10_median",
    "o3_median",
    "no2_median",
    "so2_median",
    "co_median",
    "temperature_median",
    "humidity_median",
    "pressure_median",
    "wind_speed_median",
    "hour_of_day",
    "day_of_week",
    "month",
    "is_weekend",
]


def main():
    if not os.path.exists(DATA_PATH):
        raise SystemExit(f"{DATA_PATH} not found — run this from the project root.")

    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    print(f"Loaded {len(df)} rows from {DATA_PATH}")

    # Historical rows have no real "fetch timestamp" the way live rows
    # do — use the daily timestamp itself as a stand-in so the column
    # exists with a sensible value rather than being null.
    df["openweather_timestamp"] = df["timestamp"]

    # Ensure city matches (should already, but guard against typos)
    df["city"] = CITY_NAME

    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing:
        raise SystemExit(f"data.csv is missing expected columns: {missing}")

    df = df[RAW_COLUMNS].copy()

    # Hopsworks primary key is (city, timestamp) — make sure timestamps
    # are timezone-aware UTC, matching what the live pipeline writes.
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
    if df["openweather_timestamp"].dt.tz is None:
        df["openweather_timestamp"] = df["openweather_timestamp"].dt.tz_localize("UTC")

    # Explicit dtype enforcement — avoids relying on pandas' implicit
    # type inference, which caused a schema mismatch earlier (Windows'
    # default int is 32-bit, and whole-number floats get inferred as
    # ints on small samples). Every numeric column gets a fixed,
    # deliberate dtype so future inserts (live or backfill) always
    # match the Feature Group schema exactly.
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

    print(f"Prepared {len(df)} rows for insert. Sample:")
    print(df.head(3).to_string(index=False))

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
    print(f"\nInserted {len(df)} historical rows into Hopsworks Feature Group 'aqi_features'.")


if __name__ == "__main__":
    main()