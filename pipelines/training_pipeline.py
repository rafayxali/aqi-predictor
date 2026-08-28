

import json
import os
import shutil
import sys

import joblib
import numpy as np
import pandas as pd
import hopsworks

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import tensorflow as tf
from tensorflow import keras
from dotenv import load_dotenv

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.city_config import CITY_NAME

load_dotenv()

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT")

MODELS_DIR = "models"
RESULTS_DIR = "training_results"

TEST_SIZE = 0.20
RANDOM_STATE = 42

TARGET_COLUMNS = [
    "aqi_next_1d",
    "aqi_next_2d",
    "aqi_next_3d",
]

FEATURE_COLUMNS = [
    "aqi",
    "aqi_lag_1",
    "aqi_lag_2",
    "aqi_lag_3",
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
    "day_of_week",
    "month",
    "is_weekend",
    "aqi_change_rate",
    "rolling_avg_aqi_3d",
    "rolling_avg_aqi_7d",
]


ARTIFACT_MAP = {
    "Ridge Regression": ["ridge_model.pkl"],
    "Random Forest": ["random_forest_model.pkl"],
    "TensorFlow": ["tensorflow_model.keras", "tensorflow_scaler.pkl", "tensorflow_metadata.pkl"],
}


#load from hopwork

def load_data_from_hopsworks():
    print("=" * 60)
    print("Pearls AQI Predictor - Phase 3 Training (Hopsworks)")
    print("=" * 60)

    project = hopsworks.login(
        project=HOPSWORKS_PROJECT,
        host="eu-west.cloud.hopsworks.ai",
        port=443,
        api_key_value=HOPSWORKS_API_KEY,
    )
    fs = project.get_feature_store()
    fg = fs.get_feature_group("aqi_features", version=1)

    print("Reading full Feature Group (historical backfill + all live rows)...")
    df = fg.read()
    print(f"Raw rows read from Hopsworks: {len(df)}")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return project, df


def aggregate_to_daily(df):
   
    df = df.copy()
    df["date"] = df["timestamp"].dt.date

    agg_cols = [
        "aqi", "pm25_median", "pm10_median", "o3_median", "no2_median",
        "so2_median", "co_median", "temperature_median", "humidity_median",
        "pressure_median", "wind_speed_median",
    ]

    daily = df.groupby("date")[agg_cols].median().reset_index()
    daily["timestamp"] = pd.to_datetime(daily["date"])
    daily["city"] = CITY_NAME

    daily["day_of_week"] = daily["timestamp"].dt.dayofweek
    daily["month"] = daily["timestamp"].dt.month
    daily["is_weekend"] = daily["day_of_week"].isin([5, 6]).astype(int)

    daily = daily.sort_values("timestamp").reset_index(drop=True)
    print(f"Aggregated to {len(daily)} daily rows.")
    return daily


def create_lag_features(df):
    df = df.copy()
    df["aqi_lag_1"] = df["aqi"].shift(1)
    df["aqi_lag_2"] = df["aqi"].shift(2)
    df["aqi_lag_3"] = df["aqi"].shift(3)
    return df


def create_rolling_features(df):
    df = df.copy()
    df["aqi_change_rate"] = df["aqi"].diff()
    df["rolling_avg_aqi_3d"] = df["aqi"].rolling(3, min_periods=1).mean()
    df["rolling_avg_aqi_7d"] = df["aqi"].rolling(7, min_periods=1).mean()
    return df


def create_targets(df):
    print("\nCreating 3-day forecasting targets...")
    df = df.copy()
    df["aqi_next_1d"] = df["aqi"].shift(-1)
    df["aqi_next_2d"] = df["aqi"].shift(-2)
    df["aqi_next_3d"] = df["aqi"].shift(-3)

    before = len(df)
    df = df.dropna(subset=TARGET_COLUMNS).reset_index(drop=True)
    print(f"Removed {before - len(df)} rows without future targets.")
    print(f"Training rows available: {len(df)}")
    return df




def ensure_directories():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)


def validate_features(df):
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    missing_targets = [c for c in TARGET_COLUMNS if c not in df.columns]
    if missing_targets:
        raise ValueError(f"Missing target columns: {missing_targets}")


def prepare_train_test_data(df):
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMNS].copy()

    split_index = int(len(df) * (1 - TEST_SIZE))
    X_train, X_test = X.iloc[:split_index].copy(), X.iloc[split_index:].copy()
    y_train, y_test = y.iloc[:split_index].copy(), y.iloc[split_index:].copy()

    print("\nChronological train/test split")
    print("-" * 60)
    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows:  {len(X_test)}")
    print(f"Training period: {df['timestamp'].iloc[0].date()} -> {df['timestamp'].iloc[split_index - 1].date()}")
    print(f"Testing period:  {df['timestamp'].iloc[split_index].date()} -> {df['timestamp'].iloc[-1].date()}")

    return X_train, X_test, y_train, y_test




def evaluate_model(model_name, y_true, y_pred):
    results = {"model": model_name, "horizons": {}}
    horizon_names = {"aqi_next_1d": "1_day", "aqi_next_2d": "2_day", "aqi_next_3d": "3_day"}

    all_true, all_pred = [], []
    for index, target in enumerate(TARGET_COLUMNS):
        true_values = y_true.iloc[:, index].values
        pred_values = y_pred[:, index]
        rmse = np.sqrt(mean_squared_error(true_values, pred_values))
        mae = mean_absolute_error(true_values, pred_values)
        r2 = r2_score(true_values, pred_values)
        results["horizons"][horizon_names[target]] = {"rmse": float(rmse), "mae": float(mae), "r2": float(r2)}
        all_true.extend(true_values)
        all_pred.extend(pred_values)

    results["overall"] = {
        "rmse": float(np.sqrt(mean_squared_error(all_true, all_pred))),
        "mae": float(mean_absolute_error(all_true, all_pred)),
        "r2": float(r2_score(all_true, all_pred)),
    }
    return results


def print_results(results):
    print("\n" + "=" * 70)
    print(f"{results['model']} RESULTS")
    print("=" * 70)
    print(f"{'Horizon':<12}{'RMSE':<15}{'MAE':<15}{'R²':<15}")
    print("-" * 70)
    for horizon, metrics in results["horizons"].items():
        print(f"{horizon:<12}{metrics['rmse']:<15.4f}{metrics['mae']:<15.4f}{metrics['r2']:<15.4f}")
    print("-" * 70)
    overall = results["overall"]
    print(f"{'OVERALL':<12}{overall['rmse']:<15.4f}{overall['mae']:<15.4f}{overall['r2']:<15.4f}")


def evaluate_persistence_baseline(X_test, y_test):
  
    print("\n" + "=" * 60)
    print("Evaluating naive persistence baseline (tomorrow = today)")
    print("=" * 60)
    today_aqi = X_test["aqi"].values
    predictions = np.column_stack([today_aqi, today_aqi, today_aqi])
    results = evaluate_model("Persistence Baseline (tomorrow=today)", y_test, predictions)
    print_results(results)
    return results


def train_ridge(X_train, X_test, y_train, y_test):
    print("\n" + "=" * 60 + "\nTraining Ridge Regression\n" + "=" * 60)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = Ridge(alpha=1.0)
    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)
    results = evaluate_model("Ridge Regression", y_test, predictions)
    print_results(results)
    joblib.dump({"model": model, "scaler": scaler, "features": FEATURE_COLUMNS, "targets": TARGET_COLUMNS},
                os.path.join(MODELS_DIR, "ridge_model.pkl"))
    return results

#random forest

def train_random_forest(X_train, X_test, y_train, y_test):
    print("\n" + "=" * 60 + "\nTraining Random Forest\n" + "=" * 60)
    model = RandomForestRegressor(n_estimators=300, max_depth=None, min_samples_split=2,
                                   min_samples_leaf=1, random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    results = evaluate_model("Random Forest", y_test, predictions)
    print_results(results)
    joblib.dump({"model": model, "features": FEATURE_COLUMNS, "targets": TARGET_COLUMNS},
                os.path.join(MODELS_DIR, "random_forest_model.pkl"))
    return results


# TensorFlow Neural Network


def build_tensorflow_model(input_dim):
    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(3, activation="linear"),
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse", metrics=["mae"])
    return model


def train_tensorflow(X_train, X_test, y_train, y_test):
    print("\n" + "=" * 60 + "\nTraining TensorFlow Neural Network\n" + "=" * 60)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    y_train_np = y_train.values.astype(np.float32)
    y_test_np = y_test.values.astype(np.float32)

    model = build_tensorflow_model(X_train_scaled.shape[1])
    early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)
    model.fit(X_train_scaled, y_train_np, validation_split=0.15, epochs=300, batch_size=32,
              callbacks=[early_stopping], verbose=1)

    predictions = model.predict(X_test_scaled, verbose=0)
    results = evaluate_model("TensorFlow", y_test, predictions)
    print_results(results)

    model.save(os.path.join(MODELS_DIR, "tensorflow_model.keras"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "tensorflow_scaler.pkl"))
    joblib.dump({"features": FEATURE_COLUMNS, "targets": TARGET_COLUMNS},
                os.path.join(MODELS_DIR, "tensorflow_metadata.pkl"))
    return results



# Model Registry


def get_current_registry_best_rmse(project):
    """
    We only ever register a new version when it BEATS the previous
    one, so the latest (highest) version already IS the current best
    by construction. Returns None if nothing has been registered yet.
    """
    mr = project.get_model_registry()
    try:
        models = mr.get_models("aqi_predictor")
    except Exception:
        models = []

    if not models:
        print("No model currently registered — this will be the first version.")
        return None

    latest = max(models, key=lambda m: m.version)
    rmse = latest.training_metrics.get("rmse")
    print(f"Current registry best (v{latest.version}): RMSE={rmse:.4f}")
    return rmse


def register_model(project, best_result):
    """Bundles whichever artifact files the winning model type produced
    and uploads them as a new Model Registry version."""
    mr = project.get_model_registry()

    files = ARTIFACT_MAP[best_result["model"]]
    bundle_dir = os.path.join(MODELS_DIR, "registry_bundle")
    if os.path.exists(bundle_dir):
        shutil.rmtree(bundle_dir)
    os.makedirs(bundle_dir, exist_ok=True)
    for fname in files:
        shutil.copy(os.path.join(MODELS_DIR, fname), os.path.join(bundle_dir, fname))

    model = mr.python.create_model(
        name="aqi_predictor",
        metrics=best_result["overall"],
        description=f"{best_result['model']} (Islamabad AQI 3-day forecast) — "
                     f"RMSE {best_result['overall']['rmse']:.4f}, "
                     f"MAE {best_result['overall']['mae']:.4f}, "
                     f"R2 {best_result['overall']['r2']:.4f}",
    )
    model.save(bundle_dir)
    print(f"\nRegistered '{best_result['model']}' as Model Registry version v{model.version}.")



def main():
    ensure_directories()

    project, raw_df = load_data_from_hopsworks()
    daily_df = aggregate_to_daily(raw_df)
    daily_df = create_lag_features(daily_df)
    daily_df = create_rolling_features(daily_df)
    df = create_targets(daily_df)

    validate_features(df)

    before = len(df)
    df = df.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
    print(f"\nRemoved {before - len(df)} rows containing missing feature values.")

    X_train, X_test, y_train, y_test = prepare_train_test_data(df)

    baseline_result = evaluate_persistence_baseline(X_test, y_test)

    all_results = [
        train_ridge(X_train, X_test, y_train, y_test),
        train_random_forest(X_train, X_test, y_train, y_test),
        train_tensorflow(X_train, X_test, y_train, y_test),
    ]

    print("\n" + "=" * 80 + "\nFINAL MODEL COMPARISON (incl. naive baseline)\n" + "=" * 80)
    print(f"{'Model':<32}{'Overall RMSE':<18}{'Overall MAE':<18}{'Overall R²':<18}")
    print("-" * 86)
    b = baseline_result["overall"]
    print(f"{baseline_result['model']:<32}{b['rmse']:<18.4f}{b['mae']:<18.4f}{b['r2']:<18.4f}")
    print("-" * 86)
    for result in all_results:
        overall = result["overall"]
        print(f"{result['model']:<32}{overall['rmse']:<18.4f}{overall['mae']:<18.4f}{overall['r2']:<18.4f}")

    best_result = min(all_results, key=lambda r: r["overall"]["rmse"])

    rmse_improvement_pct = (b["rmse"] - best_result["overall"]["rmse"]) / b["rmse"] * 100
    print(f"\nBest model RMSE improvement over naive baseline: {rmse_improvement_pct:.1f}%")

    print("\n" + "=" * 80)
    print(f"BEST MODEL TODAY: {best_result['model']}")
    print("=" * 80)
    print(f"Overall RMSE: {best_result['overall']['rmse']:.4f}")
    print(f"Overall MAE:  {best_result['overall']['mae']:.4f}")
    print(f"Overall R²:   {best_result['overall']['r2']:.4f}")


    #only overwrite if better
    
    current_best_rmse = get_current_registry_best_rmse(project)
    new_rmse = best_result["overall"]["rmse"]

    promoted = False
    if current_best_rmse is None or new_rmse < current_best_rmse:
        register_model(project, best_result)
        promoted = True
    else:
        print(
            f"\nNOT promoted — today's best RMSE ({new_rmse:.4f}) did not beat "
            f"the current registry best ({current_best_rmse:.4f}). "
            f"Dashboard keeps serving the existing model."
        )

    metrics_output = {
        "dataset": {
            "source": "hopsworks:aqi_features",
            "rows": len(df),
            "features": FEATURE_COLUMNS,
            "targets": TARGET_COLUMNS,
            "test_size": TEST_SIZE,
        },
        "baseline": baseline_result,
        "models": all_results,
        "best_model_today": best_result["model"],
        "rmse_improvement_over_baseline_pct": rmse_improvement_pct,
        "promoted_to_registry": promoted,
        "previous_registry_best_rmse": current_best_rmse,
    }
    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_output, f, indent=4)

    print(f"\nMetrics saved to: {os.path.join(RESULTS_DIR, 'metrics.json')}")
    print("\n" + "=" * 80 + "\nTRAINING COMPLETE\n" + "=" * 80)


if __name__ == "__main__":
    main()