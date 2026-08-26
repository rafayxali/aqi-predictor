"""
Pearls AQI Predictor - Dashboard (Streamlit)

Run locally (with the backend already running in another terminal):
    pip install streamlit requests pandas plotly
    streamlit run dashboard.py

Environment variable:
    BACKEND_URL (default: http://127.0.0.1:8000)
"""

import os
import requests
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Pearls AQI Predictor", page_icon="🌫️", layout="centered")


@st.cache_data(ttl=300, show_spinner="Loading...")
def fetch_predict():
    resp = requests.get(f"{BACKEND_URL}/predict", timeout=120)
    resp.raise_for_status()
    return resp.json()


@st.cache_data(ttl=300, show_spinner="Loading...")
def fetch_history(days=30):
    resp = requests.get(f"{BACKEND_URL}/history", params={"days": days}, timeout=120)
    resp.raise_for_status()
    return resp.json()


st.title("🌫️ Pearls AQI Predictor")
st.caption("3-day Air Quality Index forecast for Islamabad")

try:
    data = fetch_predict()
except Exception as e:
    st.error(f"Could not reach the backend API at {BACKEND_URL}. Is it running?\n\n{e}")
    st.stop()

# --------------------------------------------------------
# Hazard alert banner
# --------------------------------------------------------
if data["hazard_alert"]:
    hazard_dates = ", ".join(data["hazard_days"])
    st.error(
        f"⚠️ **Hazard Alert:** AQI is forecast to reach Unhealthy levels or worse on: {hazard_dates}. "
        "Consider limiting outdoor activity on these days."
    )

# --------------------------------------------------------
# Current AQI
# --------------------------------------------------------
st.subheader(f"Current AQI — as of {data['as_of_date']}")
current_col1, current_col2 = st.columns([1, 2])
with current_col1:
    st.metric("AQI", data["current_aqi"])
with current_col2:
    st.markdown(
        f"<div style='background-color:{data['current_category']['color']}; "
        f"padding:10px; border-radius:8px; text-align:center; color:black; font-weight:bold;'>"
        f"{data['current_category']['category']}</div>",
        unsafe_allow_html=True,
    )

st.divider()

# --------------------------------------------------------
# 3-day AQI forecast cards
# --------------------------------------------------------
st.subheader("3-Day AQI Forecast")
forecast_cols = st.columns(3)
for col, day in zip(forecast_cols, data["forecast"]):
    with col:
        st.markdown(f"**{day['date']}**")
        st.metric("Predicted AQI", day["predicted_aqi"])
        st.markdown(
            f"<div style='background-color:{day['color']}; padding:8px; "
            f"border-radius:6px; text-align:center; color:black; font-size:0.85em;'>"
            f"{day['category']}</div>",
            unsafe_allow_html=True,
        )

st.divider()

# --------------------------------------------------------
# Current weather
# --------------------------------------------------------
st.subheader("Current Weather")
w = data["current_weather"]
wcol1, wcol2, wcol3, wcol4 = st.columns(4)
wcol1.metric("Temperature", f"{w['temperature']} °C")
wcol2.metric("Humidity", f"{w['humidity']} %")
wcol3.metric("Pressure", f"{w['pressure']} hPa")
wcol4.metric("Wind Speed", f"{w['wind_speed']} m/s")

st.divider()

# --------------------------------------------------------
# 3-day weather forecast
# --------------------------------------------------------
st.subheader("3-Day Weather Forecast")
weather_cols = st.columns(3)
for col, day in zip(weather_cols, data["weather_forecast"]):
    with col:
        st.markdown(f"**{day['date']}**")
        st.metric("Temp", f"{day['temperature']} °C")
        st.caption(f"💧 {day['humidity']}%  ·  🌬️ {day['wind_speed']} m/s")

st.divider()

# --------------------------------------------------------
# Historical trend chart
# --------------------------------------------------------
st.subheader("Historical AQI Trend")
days_option = st.slider("Days of history to show", min_value=7, max_value=180, value=30, step=7)

try:
    hist_data = fetch_history(days=days_option)
    hist_df = pd.DataFrame(hist_data["history"])
    if not hist_df.empty:
        hist_df["date"] = pd.to_datetime(hist_df["date"])
        st.line_chart(hist_df.set_index("date")["aqi"])
    else:
        st.info("No historical data available yet.")
except Exception as e:
    st.warning(f"Could not load historical trend: {e}")
    hist_df = pd.DataFrame()

st.divider()

# --------------------------------------------------------
# AQI calendar heatmap
# --------------------------------------------------------
st.subheader("AQI Calendar Heatmap")

if not hist_df.empty:
    heat_df = hist_df.copy()
    heat_df["weekday"] = heat_df["date"].dt.weekday  # 0=Mon
    heat_df["week"] = heat_df["date"].dt.isocalendar().week

    pivot = heat_df.pivot_table(index="weekday", columns="week", values="aqi", aggfunc="mean")
    weekday_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    pivot = pivot.reindex(range(7))

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=[str(w) for w in pivot.columns],
            y=weekday_labels,
            colorscale=[
                [0.0, "#00E400"],
                [0.20, "#FFFF00"],
                [0.40, "#FF7E00"],
                [0.55, "#FF0000"],
                [0.75, "#8F3F97"],
                [1.0, "#7E0023"],
            ],
            zmin=0,
            zmax=300,
            hovertemplate="Week %{x}, %{y}<br>AQI: %{z}<extra></extra>",
            colorbar=dict(title="AQI"),
        )
    )
    fig.update_layout(
        height=280,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="Week",
        yaxis_title="",
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No data available for heatmap yet.")

st.divider()

# --------------------------------------------------------
# Historical weather trend chart
# --------------------------------------------------------
st.subheader("Historical Weather Trend")
if not hist_df.empty:
    weather_metric = st.selectbox(
        "Weather variable", ["temperature", "humidity", "pressure", "wind_speed"]
    )
    st.line_chart(hist_df.set_index("date")[weather_metric])
else:
    st.info("No historical weather data available yet.")