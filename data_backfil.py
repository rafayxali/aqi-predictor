"""
Historical data backfill for Pearls AQI Predictor.

Builds a 2-year DAILY historical training dataset for Islamabad by combining:

    - OpenWeather Air Pollution History API -> hourly pollutant concentrations
    - Open-Meteo Archive API               -> hourly historical weather

Hourly data is aggregated into DAILY MEDIANS.

Output:
    data.csv

The script is RESUMABLE:
    - If data.csv exists, it detects the latest completed date.
    - It continues from the next date.
    - Data is saved after every successful chunk.
    - If the program terminates, previously saved data remains available.
    - Duplicate dates are removed automatically.

Run locally:

    pip install requests pandas python-dotenv

    Set OPENWEATHER_API_KEY in .env

    python data_backfil.py
"""

import os
import sys
import time
from datetime import datetime, timedelta, timezone

import requests
import pandas as pd
from dotenv import load_dotenv


# ---------------------------------------------------------
# Project imports
# ---------------------------------------------------------

sys.path.append(
    os.path.join(os.path.dirname(__file__), "..")
)

from utils.city_config import (
    CITY_NAME,
    LATITUDE,
    LONGITUDE,
)

from utils.feature_eng import (
    pm25_to_aqi,
    add_time_features,
)


# ---------------------------------------------------------
# Environment
# ---------------------------------------------------------

load_dotenv()

OPENWEATHER_API_KEY = os.getenv(
    "OPENWEATHER_API_KEY"
)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

OUTPUT_FILE = "data.csv"

LOCAL_TZ = "Asia/Karachi"

# Two years of historical data
END_DATE = datetime.now(
    timezone.utc
).date()

START_DATE = (
    END_DATE
    - timedelta(days=730)
)

# OpenWeather historical API requests
# are made in 30-day chunks.
CHUNK_DAYS = 30


# ---------------------------------------------------------
# OpenWeather pollution history
# ---------------------------------------------------------

def fetch_openweather_pollution_history(
    start_date,
    end_date,
):
    """
    Fetch historical OpenWeather pollution data.

    OpenWeather returns hourly pollution observations.
    """

    if not OPENWEATHER_API_KEY:
        raise RuntimeError(
            "OPENWEATHER_API_KEY is not set "
            "in the .env file."
        )

    url = (
        "http://api.openweathermap.org/data/2.5/"
        "air_pollution/history"
    )

    start_ts = int(
        datetime.combine(
            start_date,
            datetime.min.time(),
            tzinfo=timezone.utc,
        ).timestamp()
    )

    end_ts = int(
        datetime.combine(
            end_date,
            datetime.min.time(),
            tzinfo=timezone.utc,
        ).timestamp()
    )

    params = {
        "lat": LATITUDE,
        "lon": LONGITUDE,
        "start": start_ts,
        "end": end_ts,
        "appid": OPENWEATHER_API_KEY,
    }

    print(
        f"Fetching pollution "
        f"{start_date} -> {end_date}..."
    )

    response = requests.get(
        url,
        params=params,
        timeout=60,
    )

    response.raise_for_status()

    payload = response.json()

    records = []

    for entry in payload.get("list", []):

        components = entry.get(
            "components",
            {}
        )

        records.append(
            {
                "timestamp_utc": pd.to_datetime(
                    entry["dt"],
                    unit="s",
                    utc=True,
                ),

                "pm25": components.get(
                    "pm2_5"
                ),

                "pm10": components.get(
                    "pm10"
                ),

                "o3": components.get(
                    "o3"
                ),

                "no2": components.get(
                    "no2"
                ),

                "so2": components.get(
                    "so2"
                ),

                "co": components.get(
                    "co"
                ),
            }
        )

    df = pd.DataFrame(records)

    print(
        f"  Received "
        f"{len(df)} hourly pollution records."
    )

    return df


# ---------------------------------------------------------
# Open-Meteo weather history
# ---------------------------------------------------------

def fetch_openmeteo_weather_history():
    """
    Fetch historical weather for the entire
    2-year period from Open-Meteo.

    Open-Meteo returns hourly data in
    Islamabad local time.
    """

    url = (
        "https://archive-api.open-meteo.com/v1/archive"
    )

    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,

        "start_date":
            START_DATE.isoformat(),

        "end_date":
            END_DATE.isoformat(),

        "hourly": (
            "temperature_2m,"
            "relative_humidity_2m,"
            "surface_pressure,"
            "wind_speed_10m"
        ),

        "timezone": LOCAL_TZ,
    }

    print(
        f"Fetching weather "
        f"{START_DATE} -> {END_DATE}..."
    )

    response = requests.get(
        url,
        params=params,
        timeout=120,
    )

    response.raise_for_status()

    payload = response.json()

    hourly = payload["hourly"]

    df = pd.DataFrame(
        {
            "timestamp_local":
                pd.to_datetime(
                    hourly["time"]
                ),

            "temperature":
                hourly["temperature_2m"],

            "humidity":
                hourly[
                    "relative_humidity_2m"
                ],

            "pressure":
                hourly[
                    "surface_pressure"
                ],

            "wind_speed":
                hourly[
                    "wind_speed_10m"
                ],
        }
    )

    print(
        f"  Received "
        f"{len(df)} hourly weather records."
    )

    return df


# ---------------------------------------------------------
# Pollution → daily median
# ---------------------------------------------------------

def aggregate_pollution_daily(
    pollution_df,
):
    """
    Convert hourly pollution observations
    into daily median values.

    Pollution timestamps are converted from
    UTC to Islamabad local time before grouping.
    """

    if pollution_df.empty:
        return pd.DataFrame()

    pollution_df[
        "timestamp_local"
    ] = (
        pollution_df[
            "timestamp_utc"
        ]
        .dt
        .tz_convert(LOCAL_TZ)
        .dt
        .tz_localize(None)
    )

    pollution_df["date"] = (
        pollution_df[
            "timestamp_local"
        ].dt.date
    )

    daily = (
        pollution_df
        .groupby("date")
        .agg(
            pm25_median=(
                "pm25",
                "median",
            ),

            pm10_median=(
                "pm10",
                "median",
            ),

            o3_median=(
                "o3",
                "median",
            ),

            no2_median=(
                "no2",
                "median",
            ),

            so2_median=(
                "so2",
                "median",
            ),

            co_median=(
                "co",
                "median",
            ),
        )
        .reset_index()
    )

    return daily


# ---------------------------------------------------------
# Weather → daily median
# ---------------------------------------------------------

def aggregate_weather_daily(
    weather_df,
):
    """
    Convert hourly weather observations
    into daily median values.
    """

    if weather_df.empty:
        return pd.DataFrame()

    weather_df["date"] = (
        weather_df[
            "timestamp_local"
        ].dt.date
    )

    daily = (
        weather_df
        .groupby("date")
        .agg(
            temperature_median=(
                "temperature",
                "median",
            ),

            humidity_median=(
                "humidity",
                "median",
            ),

            pressure_median=(
                "pressure",
                "median",
            ),

            wind_speed_median=(
                "wind_speed",
                "median",
            ),
        )
        .reset_index()
    )

    return daily


# ---------------------------------------------------------
# Build daily feature rows
# ---------------------------------------------------------

def build_daily_features(
    pollution_daily,
    weather_daily,
):
    """
    Merge daily pollution + weather data
    and create the training features.
    """

    merged = pd.merge(
        pollution_daily,
        weather_daily,
        on="date",
        how="inner",
    )

    if merged.empty:
        return pd.DataFrame()

    merged["timestamp"] = pd.to_datetime(
        merged["date"]
    )

    # -----------------------------------------------------
    # Target
    # -----------------------------------------------------

    # AQI is calculated from DAILY MEDIAN PM2.5
    merged["aqi"] = (
        merged[
            "pm25_median"
        ].apply(pm25_to_aqi)
    )

    # -----------------------------------------------------
    # City
    # -----------------------------------------------------

    merged["city"] = CITY_NAME

    # -----------------------------------------------------
    # Time features
    # -----------------------------------------------------

    merged = add_time_features(
        merged,
        timestamp_col="timestamp",
    )

    merged = (
        merged
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    return merged


# ---------------------------------------------------------
# Rolling / change features
# ---------------------------------------------------------

def add_rolling_features(
    df,
):
    """
    Add features that depend on previous days.

    These are calculated over the COMPLETE dataset,
    rather than separately inside each 30-day API chunk.
    """

    if df.empty:
        return df

    df = (
        df
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    # AQI change from previous day
    df["aqi_change_rate"] = (
        df["aqi"].diff()
    )

    # 3-day rolling AQI
    df["rolling_avg_aqi_3d"] = (
        df["aqi"]
        .rolling(
            3,
            min_periods=1,
        )
        .mean()
    )

    # 7-day rolling AQI
    df["rolling_avg_aqi_7d"] = (
        df["aqi"]
        .rolling(
            7,
            min_periods=1,
        )
        .mean()
    )

    return df


# ---------------------------------------------------------
# Existing data
# ---------------------------------------------------------

def load_existing_data():
    """
    Load existing data.csv.

    Handles:
        - file doesn't exist
        - file is empty
        - file has no rows
        - normal existing dataset
    """

    if not os.path.exists(
        OUTPUT_FILE
    ):

        print(
            "No existing data.csv found."
        )

        return (
            pd.DataFrame(),
            None,
        )

    # Empty file
    if os.path.getsize(
        OUTPUT_FILE
    ) == 0:

        print(
            "Existing data.csv is empty. "
            "Starting fresh."
        )

        return (
            pd.DataFrame(),
            None,
        )

    print(
        f"Found existing "
        f"{OUTPUT_FILE}. Loading..."
    )

    try:

        df = pd.read_csv(
            OUTPUT_FILE,
            parse_dates=[
                "timestamp"
            ],
        )

    except (
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
    ) as e:

        print(
            f"Could not read existing "
            f"data.csv ({e})."
        )

        print(
            "Starting fresh."
        )

        return (
            pd.DataFrame(),
            None,
        )

    if df.empty:

        print(
            "Existing data.csv contains "
            "no rows. Starting fresh."
        )

        return (
            df,
            None,
        )

    df = df.drop_duplicates(
        subset=["timestamp"]
    )

    df = (
        df
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    latest_date = (
        df["timestamp"]
        .max()
        .date()
    )

    print(
        f"Existing rows: {len(df)}"
    )

    print(
        f"Latest completed date: "
        f"{latest_date}"
    )

    return (
        df,
        latest_date,
    )


# ---------------------------------------------------------
# Safe save
# ---------------------------------------------------------

def save_data(
    df,
):
    """
    Safely save data.csv.

    Writes to a temporary file first and
    replaces the original after successful write.
    """

    df = (
        df
        .drop_duplicates(
            subset=["timestamp"]
        )
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    temp_file = (
        OUTPUT_FILE
        + ".tmp"
    )

    df.to_csv(
        temp_file,
        index=False,
    )

    os.replace(
        temp_file,
        OUTPUT_FILE,
    )

    print(
        f"Saved {len(df)} rows "
        f"to {OUTPUT_FILE}"
    )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    print("=" * 60)

    print(
        "Pearls AQI Predictor - "
        "Historical Backfill"
    )

    print("=" * 60)

    print(
        f"City: {CITY_NAME}"
    )

    print(
        f"Coordinates: "
        f"{LATITUDE}, {LONGITUDE}"
    )

    print(
        f"Target date range: "
        f"{START_DATE} -> {END_DATE}"
    )

    print(
        f"Output file: "
        f"{OUTPUT_FILE}"
    )

    # -----------------------------------------------------
    # Load existing data
    # -----------------------------------------------------

    existing_df, latest_date = (
        load_existing_data()
    )

    # -----------------------------------------------------
    # Determine starting date
    # -----------------------------------------------------

    if latest_date is not None:

        current_start = (
            latest_date
            + timedelta(days=1)
        )

        if current_start > END_DATE:

            print(
                "\nHistorical dataset "
                "is already up to date."
            )

            return

        print(
            f"\nResuming from "
            f"{current_start}"
        )

    else:

        current_start = START_DATE

        print(
            f"\nStarting from "
            f"{current_start}"
        )

    # -----------------------------------------------------
    # Fetch Open-Meteo weather once
    # -----------------------------------------------------

    print(
        "\nFetching weather history..."
    )

    weather_hourly = (
        fetch_openmeteo_weather_history()
    )

    weather_daily = (
        aggregate_weather_daily(
            weather_hourly
        )
    )

    print(
        f"Weather daily rows: "
        f"{len(weather_daily)}"
    )

    # -----------------------------------------------------
    # Process OpenWeather chunks
    # -----------------------------------------------------

    while current_start < END_DATE:

        current_end = min(
            current_start
            + timedelta(days=CHUNK_DAYS),
            END_DATE,
        )

        try:

            # ---------------------------------------------
            # Fetch pollution
            # ---------------------------------------------

            pollution_hourly = (
                fetch_openweather_pollution_history(
                    current_start,
                    current_end,
                )
            )

            # ---------------------------------------------
            # Aggregate pollution
            # ---------------------------------------------

            pollution_daily = (
                aggregate_pollution_daily(
                    pollution_hourly
                )
            )

            # ---------------------------------------------
            # Select weather for this chunk
            # ---------------------------------------------

            chunk_weather = (
                weather_daily[
                    (
                        weather_daily[
                            "date"
                        ]
                        >= current_start
                    )
                    &
                    (
                        weather_daily[
                            "date"
                        ]
                        <= current_end
                    )
                ]
                .copy()
            )

            # ---------------------------------------------
            # Build daily features
            # ---------------------------------------------

            chunk_df = (
                build_daily_features(
                    pollution_daily,
                    chunk_weather,
                )
            )

            if chunk_df.empty:

                print(
                    "WARNING: No usable rows "
                    f"for {current_start} "
                    f"-> {current_end}"
                )

            else:

                # -----------------------------------------
                # Append chunk
                # -----------------------------------------

                if existing_df.empty:

                    existing_df = (
                        chunk_df
                    )

                else:

                    existing_df = pd.concat(
                        [
                            existing_df,
                            chunk_df,
                        ],
                        ignore_index=True,
                    )

                # -----------------------------------------
                # Remove duplicate dates
                # -----------------------------------------

                existing_df = (
                    existing_df
                    .drop_duplicates(
                        subset=["timestamp"]
                    )
                    .sort_values(
                        "timestamp"
                    )
                    .reset_index(
                        drop=True
                    )
                )

                # -----------------------------------------
                # Recalculate rolling features
                # over COMPLETE dataset
                # -----------------------------------------

                existing_df = (
                    add_rolling_features(
                        existing_df
                    )
                )

                # -----------------------------------------
                # Save immediately
                # -----------------------------------------

                save_data(
                    existing_df
                )

                print(
                    f"Completed "
                    f"{current_start} "
                    f"-> {current_end}"
                )

                print(
                    f"Rows in dataset: "
                    f"{len(existing_df)}"
                )

        except Exception as e:

            print(
                "\nERROR while processing "
                f"{current_start} "
                f"-> {current_end}"
            )

            print(
                f"{type(e).__name__}: {e}"
            )

            print(
                "\nPreviously saved data "
                "has been preserved."
            )

            print(
                "Run the script again "
                "to resume."
            )

            raise

        # Move to next chunk
        current_start = current_end

        # Small delay between API calls
        time.sleep(1)

    # -----------------------------------------------------
    # Final summary
    # -----------------------------------------------------

    print(
        "\n" + "=" * 60
    )

    print(
        "BACKFILL COMPLETE"
    )

    print(
        "=" * 60
    )

    print(
        f"Rows: {len(existing_df)}"
    )

    print(
        f"Columns: "
        f"{len(existing_df.columns)}"
    )

    print(
        f"Date range: "
        f"{existing_df['timestamp'].min()} "
        f"-> "
        f"{existing_df['timestamp'].max()}"
    )

    print(
        "\nMissing values:"
    )

    print(
        existing_df.isna().sum()
    )

    print(
        "\nAQI statistics:"
    )

    print(
        existing_df["aqi"].describe()
    )

    print(
        f"\nSaved to: "
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()