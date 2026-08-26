"""
Pearls AQI Predictor - Dashboard (Streamlit)

Run locally (with the backend already running in another terminal):
    pip install streamlit requests pandas plotly
    streamlit run dashboard.py

Environment variable:
    BACKEND_URL (default: http://127.0.0.1:8000)

Theming: see .streamlit/config.toml (white background theme).
"""

import os
import requests
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Pearls AQI Predictor", page_icon="🌫️", layout="wide")


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


@st.cache_data(ttl=300, show_spinner="Loading...")
def fetch_explain(horizon="day_1"):
    resp = requests.get(f"{BACKEND_URL}/explain", params={"horizon": horizon}, timeout=120)
    resp.raise_for_status()
    return resp.json()


st.title("🌫️ Pearls AQI Predictor - ISLAMABAD")

st.write("")

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
# Row 1: Current conditions (AQI + Weather side by side)
# --------------------------------------------------------
with st.container(border=True):
    st.subheader(f"Current Conditions — {data['as_of_date']}")
    col_aqi, col_weather = st.columns([1, 2], gap="large")

    with col_aqi:
        st.metric("AQI", data["current_aqi"])
        st.markdown(
            f"<div style='background-color:{data['current_category']['color']}; "
            f"padding:10px; border-radius:8px; text-align:center; color:black; font-weight:600;'>"
            f"{data['current_category']['category']}</div>",
            unsafe_allow_html=True,
        )

    with col_weather:
        w = data["current_weather"]
        wcol1, wcol2, wcol3, wcol4 = st.columns(4)
        wcol1.metric("Temperature", f"{w['temperature']} °C")
        wcol2.metric("Humidity", f"{w['humidity']} %")
        wcol3.metric("Pressure", f"{w['pressure']} hPa")
        wcol4.metric("Wind Speed", f"{w['wind_speed']} m/s")

st.write("")

# --------------------------------------------------------
# Row 2: 3-day forecasts (AQI + Weather side by side)
# --------------------------------------------------------
col_aqi_forecast, col_weather_forecast = st.columns(2, gap="large")

with col_aqi_forecast:
    with st.container(border=True):
        st.subheader("3-Day AQI Forecast")
        for day in data["forecast"]:
            fcol1, fcol2 = st.columns([1, 2])
            fcol1.markdown(f"**{day['date']}**")
            fcol1.metric("AQI", day["predicted_aqi"], label_visibility="collapsed")
            fcol2.markdown(
                f"<div style='background-color:{day['color']}; padding:6px 10px; "
                f"border-radius:6px; text-align:center; color:black; font-size:0.85em; margin-top:8px;'>"
                f"{day['category']}</div>",
                unsafe_allow_html=True,
            )

with col_weather_forecast:
    with st.container(border=True):
        st.subheader("3-Day Weather Forecast")
        for day in data["weather_forecast"]:
            wfcol1, wfcol2 = st.columns([1, 2])
            wfcol1.markdown(f"**{day['date']}**")
            wfcol2.markdown(f"🌡️ {day['temperature']}°C · 💧 {day['humidity']}% · 🌬️ {day['wind_speed']} m/s")

st.write("")

# --------------------------------------------------------
# Row 3: Why this prediction (SHAP)
# --------------------------------------------------------
with st.container(border=True):
    st.subheader("Why This Prediction?")
    horizon_choice = st.radio(
        "Forecast day", ["day_1", "day_2", "day_3"], horizontal=True,
        format_func=lambda h: h.replace("_", " ").title(), label_visibility="collapsed"
    )

    try:
        explain_data = fetch_explain(horizon=horizon_choice)
        if explain_data["supported"]:
            top_contributions = explain_data["contributions"][:8]
            contrib_df = pd.DataFrame(top_contributions).sort_values("shap_value")

            fig_shap = go.Figure(
                go.Bar(
                    x=contrib_df["shap_value"],
                    y=contrib_df["feature"],
                    orientation="h",
                    marker_color=["#D64545" if v > 0 else "#3B7DD8" for v in contrib_df["shap_value"]],
                )
            )
            fig_shap.update_layout(
                height=320,
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis_title="Impact on predicted AQI",
                yaxis_title="",
                plot_bgcolor="white",
                paper_bgcolor="white",
            )
            st.plotly_chart(fig_shap, use_container_width=True)
            st.caption("Red pushes the prediction higher, blue pushes it lower.")
        else:
            st.info(explain_data["message"])
    except Exception as e:
        st.warning(f"Could not load explanation: {e}")

st.write("")

# --------------------------------------------------------
# Row 4: Historical trend + calendar heatmap side by side
# --------------------------------------------------------
col_trend, col_heatmap = st.columns(2, gap="large")

with col_trend:
    with st.container(border=True):
        st.subheader("Historical AQI Trend")
        days_option = st.slider("Days of history", min_value=7, max_value=180, value=30, step=7)

        try:
            hist_data = fetch_history(days=days_option)
            hist_df = pd.DataFrame(hist_data["history"])
            if not hist_df.empty:
                hist_df["date"] = pd.to_datetime(hist_df["date"])
                st.line_chart(hist_df.set_index("date")["aqi"], color="#2E7D5B")
            else:
                st.info("No historical data available yet.")
                hist_df = pd.DataFrame()
        except Exception as e:
            st.warning(f"Could not load historical trend: {e}")
            hist_df = pd.DataFrame()

with col_heatmap:
    with st.container(border=True):
        st.subheader("AQI Calendar Heatmap")
        if not hist_df.empty:
            heat_df = hist_df.copy()
            heat_df["weekday"] = heat_df["date"].dt.weekday
            heat_df["week"] = heat_df["date"].dt.isocalendar().week

            pivot = heat_df.pivot_table(index="weekday", columns="week", values="aqi", aggfunc="mean")
            weekday_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            pivot = pivot.reindex(range(7))

            fig_heat = go.Figure(
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
            fig_heat.update_layout(
                height=280,
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis_title="Week",
                yaxis_title="",
                plot_bgcolor="white",
                paper_bgcolor="white",
            )
            st.plotly_chart(fig_heat, use_container_width=True)
        else:
            st.info("No data available for heatmap yet.")

st.write("")

# --------------------------------------------------------
# Row 5: Historical weather trend
# --------------------------------------------------------
with st.container(border=True):
    st.subheader("Historical Weather Trend")
    if not hist_df.empty:
        weather_metric = st.selectbox(
            "Weather variable", ["temperature", "humidity", "pressure", "wind_speed"]
        )
        st.line_chart(hist_df.set_index("date")[weather_metric], color="#3B7DD8")
    else:
        st.info("No historical weather data available yet.")