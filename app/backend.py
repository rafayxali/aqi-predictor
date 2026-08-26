"""
Pearls AQI Predictor - Backend API (FastAPI)

Loads the latest promoted model from the Hopsworks Model Registry and
the latest features from the Feature Store, and serves 3-day AQI
predictions for Islamabad.

Run locally:
    pip install fastapi uvicorn
    uvicorn backend:app --reload --port 8000

Endpoints:
    GET /predict   -> current AQI + 3-day forecast + hazard alerts
    GET /health    -> simple liveness check
"""

import os
import sys
import shutil
import tempfile
from datetime import timedelta

import joblib
import numpy as np
import pandas as pd
import requests
import hopsworks
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.city_config import CITY_NAME, LATITUDE, LONGITUDE

load_dotenv()

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT")

FEATURE_COLUMNS = [
    "aqi", "aqi_lag_1", "aqi_lag_2", "aqi_lag_3",
    "pm25_median", "pm10_median", "o3_median", "no2_median", "so2_median", "co_median",
    "temperature_median", "humidity_median", "pressure_median", "wind_speed_median",
    "day_of_week", "month", "is_weekend",
    "aqi_change_rate", "rolling_avg_aqi_3d", "rolling_avg_aqi_7d",
]

# AQI category breakpoints (standard EPA scale)
AQI_CATEGORIES = [
    (0, 50, "Good", "#00E400"),
    (51, 100, "Moderate", "#FFFF00"),
    (101, 150, "Unhealthy for Sensitive Groups", "#FF7E00"),
    (151, 200, "Unhealthy", "#FF0000"),
    (201, 300, "Very Unhealthy", "#8F3F97"),
    (301, 500, "Hazardous", "#7E0023"),
]

app = FastAPI(title="Pearls AQI Predictor API")

_cache = {"project": None, "daily_df": None, "daily_df_ts": None}

CACHE_TTL_SECONDS = 300  # avoid redundant Hopsworks reads across /predict + /history


def get_project():
    if _cache["project"] is None:
        _cache["project"] = hopsworks.login(
            project=HOPSWORKS_PROJECT,
            host="eu-west.cloud.hopsworks.ai",
            port=443,
            api_key_value=HOPSWORKS_API_KEY,
        )
    return _cache["project"]


def get_daily_df(project):
    """Cached daily-aggregated feature data, shared by /predict and
    /history so a single page load doesn't trigger two separate slow
    Hopsworks Query Service reads."""
    import time
    now = time.time()
    if _cache["daily_df"] is not None and (now - _cache["daily_df_ts"]) < CACHE_TTL_SECONDS:
        return _cache["daily_df"].copy()

    fs = project.get_feature_store()
    fg = fs.get_feature_group("aqi_features", version=1)
    df = fg.read()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date

    agg_cols = [
        "aqi", "pm25_median", "pm10_median", "o3_median", "no2_median",
        "so2_median", "co_median", "temperature_median", "humidity_median",
        "pressure_median", "wind_speed_median",
    ]
    daily = df.groupby("date")[agg_cols].median().reset_index()
    daily["timestamp"] = pd.to_datetime(daily["date"])
    daily["day_of_week"] = daily["timestamp"].dt.dayofweek
    daily["month"] = daily["timestamp"].dt.month
    daily["is_weekend"] = daily["day_of_week"].isin([5, 6]).astype(int)
    daily = daily.sort_values("timestamp").reset_index(drop=True)

    daily["aqi_lag_1"] = daily["aqi"].shift(1)
    daily["aqi_lag_2"] = daily["aqi"].shift(2)
    daily["aqi_lag_3"] = daily["aqi"].shift(3)
    daily["aqi_change_rate"] = daily["aqi"].diff()
    daily["rolling_avg_aqi_3d"] = daily["aqi"].rolling(3, min_periods=1).mean()
    daily["rolling_avg_aqi_7d"] = daily["aqi"].rolling(7, min_periods=1).mean()

    _cache["daily_df"] = daily
    _cache["daily_df_ts"] = now
    return daily.copy()


def categorize_aqi(aqi_value):
    for lo, hi, label, color in AQI_CATEGORIES:
        if lo <= aqi_value <= hi:
            return {"category": label, "color": color}
    return {"category": "Hazardous", "color": "#7E0023"}


def fetch_weather_forecast(days=3):
    """Real 3-day weather forecast from Open-Meteo (not model-predicted —
    genuine meteorological forecast data). Fetches hourly and aggregates
    to daily median, matching the same methodology used elsewhere."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m",
        "forecast_days": days + 1,  # +1 so "today" plus the next `days`
        "timezone": "Asia/Karachi",
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    payload = resp.json()

    hourly = payload["hourly"]
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(hourly["time"]),
        "temperature": hourly["temperature_2m"],
        "humidity": hourly["relative_humidity_2m"],
        "pressure": hourly["surface_pressure"],
        "wind_speed": hourly["wind_speed_10m"],
    })
    df["date"] = df["timestamp"].dt.date

    daily = df.groupby("date").median(numeric_only=True).reset_index()
    daily = daily.sort_values("date")

    today = pd.Timestamp.now(tz="Asia/Karachi").date()
    daily = daily[daily["date"] > today].head(days)

    return [
        {
            "date": row["date"].strftime("%Y-%m-%d"),
            "temperature": round(float(row["temperature"]), 1),
            "humidity": round(float(row["humidity"]), 1),
            "pressure": round(float(row["pressure"]), 1),
            "wind_speed": round(float(row["wind_speed"]), 1),
        }
        for _, row in daily.iterrows()
    ]


def load_latest_model(project):
    """Downloads whichever model is currently the latest (best) version
    in the registry, and figures out how to load it based on which
    artifact files it contains (Ridge/RF are .pkl, TensorFlow is
    .keras + scaler + metadata — see training_pipeline.py's ARTIFACT_MAP)."""
    mr = project.get_model_registry()
    models = mr.get_models("aqi_predictor")
    if not models:
        raise HTTPException(status_code=503, detail="No model registered yet.")

    latest = max(models, key=lambda m: m.version)
    model_dir = latest.download()

    files = os.listdir(model_dir)

    if "tensorflow_model.keras" in files:
        import tensorflow as tf
        model = tf.keras.models.load_model(os.path.join(model_dir, "tensorflow_model.keras"))
        scaler = joblib.load(os.path.join(model_dir, "tensorflow_scaler.pkl"))
        return {"type": "tensorflow", "model": model, "scaler": scaler, "version": latest.version}

    elif "random_forest_model.pkl" in files:
        bundle = joblib.load(os.path.join(model_dir, "random_forest_model.pkl"))
        return {"type": "sklearn", "model": bundle["model"], "scaler": None, "version": latest.version}

    elif "ridge_model.pkl" in files:
        bundle = joblib.load(os.path.join(model_dir, "ridge_model.pkl"))
        return {"type": "sklearn", "model": bundle["model"], "scaler": bundle["scaler"], "version": latest.version}

    raise HTTPException(status_code=500, detail=f"Unrecognized model artifact files: {files}")


def load_latest_features(project):
    """Returns the single most recent complete day (with lag/rolling
    features already computed) as the prediction input row."""
    daily = get_daily_df(project)
    latest_row = daily.dropna(subset=FEATURE_COLUMNS).iloc[-1]
    return latest_row


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/history")
def history(days: int = 30):
    """Historical daily AQI + weather trend, for charting. Defaults to last 30 days."""
    project = get_project()
    daily = get_daily_df(project)
    daily = daily.sort_values("timestamp").tail(days)

    records = []
    for _, row in daily.iterrows():
        cat = categorize_aqi(row["aqi"])
        records.append({
            "date": row["timestamp"].strftime("%Y-%m-%d"),
            "aqi": round(float(row["aqi"])),
            "category": cat["category"],
            "color": cat["color"],
            "temperature": round(float(row["temperature_median"]), 1),
            "humidity": round(float(row["humidity_median"]), 1),
            "pressure": round(float(row["pressure_median"]), 1),
            "wind_speed": round(float(row["wind_speed_median"]), 1),
        })

    return {"city": CITY_NAME, "days_requested": days, "history": records}


@app.get("/predict")
def predict():
    project = get_project()
    model_info = load_latest_model(project)
    latest_row = load_latest_features(project)

    X = latest_row[FEATURE_COLUMNS].values.reshape(1, -1).astype(np.float64)

    if model_info["type"] == "tensorflow":
        X_scaled = model_info["scaler"].transform(X)
        preds = model_info["model"].predict(X_scaled, verbose=0)[0]
    else:
        if model_info["scaler"] is not None:
            X = model_info["scaler"].transform(X)
        preds = model_info["model"].predict(X)[0]

    base_date = pd.to_datetime(latest_row["timestamp"])
    current_aqi = float(latest_row["aqi"])

    forecast = []
    for i, horizon_days in enumerate([1, 2, 3]):
        predicted_aqi = round(float(preds[i]))
        cat = categorize_aqi(predicted_aqi)
        forecast.append({
            "date": (base_date + timedelta(days=horizon_days)).strftime("%Y-%m-%d"),
            "horizon": f"day_{horizon_days}",
            "predicted_aqi": predicted_aqi,
            "category": cat["category"],
            "color": cat["color"],
        })

    hazardous = [f for f in forecast if f["predicted_aqi"] > 150]

    weather_forecast = fetch_weather_forecast(days=3)

    return {
        "city": CITY_NAME,
        "as_of_date": base_date.strftime("%Y-%m-%d"),
        "current_aqi": round(current_aqi),
        "current_category": categorize_aqi(current_aqi),
        "current_weather": {
            "temperature": round(float(latest_row["temperature_median"]), 1),
            "humidity": round(float(latest_row["humidity_median"]), 1),
            "pressure": round(float(latest_row["pressure_median"]), 1),
            "wind_speed": round(float(latest_row["wind_speed_median"]), 1),
        },
        "forecast": forecast,
        "weather_forecast": weather_forecast,
        "hazard_alert": len(hazardous) > 0,
        "hazard_days": [f["date"] for f in hazardous],
        "model_used": {"type": model_info["type"], "registry_version": model_info["version"]},
    }