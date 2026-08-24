
import pandas as pd

# AQICN's PM2.5 -> AQI breakpoints (2012 EPA standard, confirmed via
# aqicn.org/faq — this is what AQICN's own live feed uses, so historical
# data computed with this table stays on the same scale as the live 'aqi'
# field returned by the API).
PM25_BREAKPOINTS = [
    (0.0, 12.0, 0, 50),
    (12.1, 35.4, 51, 100),
    (35.5, 55.4, 101, 150),
    (55.5, 150.4, 151, 200),
    (150.5, 250.4, 201, 300),
    (250.5, 350.4, 301, 400),
    (350.5, 500.4, 401, 500),
]


def pm25_to_aqi(pm25):
    """Convert a raw PM2.5 concentration (µg/m3) to US AQI using AQICN's
    breakpoint table. Returns None if pm25 is missing."""
    if pm25 is None or pd.isna(pm25):
        return None
    pm25 = max(0.0, float(pm25))
    for bp_lo, bp_hi, i_lo, i_hi in PM25_BREAKPOINTS:
        if bp_lo <= pm25 <= bp_hi:
            return round((i_hi - i_lo) / (bp_hi - bp_lo) * (pm25 - bp_lo) + i_lo)
    # Beyond the top breakpoint ("Beyond the AQI"): extend using the last band
    bp_lo, bp_hi, i_lo, i_hi = PM25_BREAKPOINTS[-1]
    return round((i_hi - i_lo) / (bp_hi - bp_lo) * (pm25 - bp_lo) + i_lo)


def add_time_features(df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
    """Add hour/day/month/weekend features derived from a timestamp column."""
    df = df.copy()
    ts = pd.to_datetime(df[timestamp_col])
    df["hour_of_day"] = ts.dt.hour
    df["day_of_week"] = ts.dt.dayofweek
    df["month"] = ts.dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    return df


def add_rolling_features(df: pd.DataFrame, aqi_col: str = "aqi") -> pd.DataFrame:
    """Add change-rate and rolling-average features. Requires df sorted
    ascending by time already."""
    df = df.copy()
    df["aqi_change_rate"] = df[aqi_col].diff()
    df["rolling_avg_aqi_3d"] = df[aqi_col].rolling(3, min_periods=1).mean()
    df["rolling_avg_aqi_7d"] = df[aqi_col].rolling(7, min_periods=1).mean()
    return df