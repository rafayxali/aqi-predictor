

import os
import sys
import shutil
import tempfile
import time
from datetime import timedelta

import joblib
import numpy as np
import pandas as pd
import requests
import shap
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

CACHE_TTL_SECONDS = 300  


def with_retry(fn, attempts=3, delay_seconds=3):
    
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as e:
            last_error = e
            print(f"Hopsworks call failed (attempt {attempt}/{attempts}): {type(e).__name__}: {e}")
            if attempt < attempts:
                time.sleep(delay_seconds)
    raise last_error


def get_project():
    if _cache["project"] is None:
        _cache["project"] = with_retry(lambda: hopsworks.login(
            project=HOPSWORKS_PROJECT,
            host="eu-west.cloud.hopsworks.ai",
            port=443,
            api_key_value=HOPSWORKS_API_KEY,
        ))
    return _cache["project"]


def get_daily_df(project):
    
    now = time.time()
    if _cache["daily_df"] is not None and (now - _cache["daily_df_ts"]) < CACHE_TTL_SECONDS:
        return _cache["daily_df"].copy()

    def _read():
        fs = project.get_feature_store()
        fg = fs.get_feature_group("aqi_features", version=1)
        return fg.read()

    df = with_retry(_read)
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

    mr = project.get_model_registry()
    models = with_retry(lambda: mr.get_models("aqi_predictor"))
    if not models:
        raise HTTPException(status_code=503, detail="No model registered yet.")

    latest = max(models, key=lambda m: m.version)
    model_dir = with_retry(lambda: latest.download())

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


@app.get("/explain")
def explain(horizon: str = "day_1"):
  
    horizon_index = {"day_1": 0, "day_2": 1, "day_3": 2}
    if horizon not in horizon_index:
        raise HTTPException(status_code=400, detail="horizon must be day_1, day_2, or day_3")

    project = get_project()
    model_info = load_latest_model(project)

    if model_info["type"] != "sklearn" or not hasattr(model_info["model"], "estimators_"):
        return {
            "supported": False,
            "message": f"SHAP explanation not available for model type '{model_info['type']}'.",
        }

    latest_row = load_latest_features(project)
    X = latest_row[FEATURE_COLUMNS].values.reshape(1, -1).astype(np.float64)

    explainer = shap.TreeExplainer(model_info["model"])
    shap_values = explainer.shap_values(X)

    # Multi-output RF: shap_values has shape (1, n_features, n_outputs)
    # or is a list of per-output arrays depending on SHAP version.
    idx = horizon_index[horizon]
    if isinstance(shap_values, list):
        values = shap_values[idx][0]
    else:
        values = shap_values[0, :, idx]

    contributions = sorted(
        [{"feature": f, "shap_value": float(v)} for f, v in zip(FEATURE_COLUMNS, values)],
        key=lambda x: abs(x["shap_value"]),
        reverse=True,
    )

    base_value = explainer.expected_value
    if hasattr(base_value, "__len__"):
        base_value = base_value[idx]

    return {
        "supported": True,
        "horizon": horizon,
        "model_version": model_info["version"],
        "base_value": float(base_value),
        "contributions": contributions,
    }