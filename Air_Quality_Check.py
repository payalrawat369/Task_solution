# dashboard_app.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta

st.set_page_config(page_title="Air Quality Alert System", layout="wide")

# ---------------- Helpers ----------------
AQI_COLORS = {
    "Good": "#66bb6a",
    "Moderate": "#ffa726",
    "Unhealthy for Sensitive": "#ff7043",
    "Unhealthy": "#d32f2f",
    "Very Unhealthy": "#6a1b9a",
    "Hazardous": "#4a148c",
    np.nan: "#9e9e9e"
}

def load_data():
    f_forecast = "forecast_aqi.csv"
    f_raw = "air_quality_data.csv"

    if not os.path.exists(f_forecast):
        st.error(f"Missing {f_forecast}. Run train_models.py first to generate forecasts.")
        st.stop()

    if not os.path.exists(f_raw):
        st.warning(f"Missing {f_raw}. Pollutant trend charts will still work from forecast file if pollutant forecasts available.")

    df_forecast = pd.read_csv(f_forecast, parse_dates=["Date"])
    df_raw = pd.read_csv(f_raw, parse_dates=["Date"]) if os.path.exists(f_raw) else None
    return df_forecast, df_raw


def make_gauge(aqi_value, category):
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=float(aqi_value) if not np.isnan(aqi_value) else 0,
        number={'valueformat': '.0f'},
        title={'text': f"AQI<br><span style='font-size:0.6em'>{category}</span>"},
        gauge={
            'axis': {'range': [0, 500]},
            'bar': {'color': AQI_COLORS.get(category, "#ffa726")},
            'steps': [
                {'range': [0, 50], 'color': '#a8e6a1'},
                {'range': [51, 100], 'color': '#ffeb99'},
                {'range': [101, 200], 'color': '#ffc4a3'},
                {'range': [201, 300], 'color': '#f4a0b9'},
                {'range': [301, 400], 'color': '#d7a3ff'},
                {'range': [401, 500], 'color': '#d6b3ff'}
            ]
        }
    ))
    fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=300)
    return fig


def category_color(cat):
    return AQI_COLORS.get(cat, "#f0f0f0")


def format_forecast_boxes(df_city):
    cols = st.columns(7)
    for i, (_, row) in enumerate(df_city.sort_values("Date").iterrows()):
        with cols[i]:
            date_str = row["Date"].strftime("%a\n%d %b")
            cat = row["AQI_Category"]
            aqi = int(row["AQI"]) if not np.isnan(row["AQI"]) else "NA"
            st.markdown(
                f"""
                <div style="
                    background:{category_color(cat)};
                    padding:12px;
                    border-radius:8px;
                    text-align:center;
                    box-shadow: 0 1px 4px rgba(0,0,0,0.08);
                ">
                    <div style="font-weight:600">{date_str}</div>
                    <div style="font-size:22px; font-weight:700; margin-top:6px">{aqi}</div>
                    <div style="font-size:12px; margin-top:6px">{cat}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ---------------- Main ----------------
df_forecast, df_raw = load_data()

st.title("Air Quality Alert System")
st.markdown("Milestone 3: Working Application — 7-day forecasts, AQI, and pollutant trends")

# Sidebar city selector
cities = df_forecast["City"].unique().tolist()
city = st.sidebar.selectbox("City", cities, index=0)

df_city_fore = df_forecast[df_forecast["City"] == city].copy()
if df_city_fore.empty:
    st.error("No forecasts for selected city.")
    st.stop()

min_date = df_city_fore["Date"].min()
max_date = df_city_fore["Date"].max()
st.sidebar.write(f"Forecast range: {min_date.date()} → {max_date.date()}")

pollutants = [c for c in df_forecast.columns if c not in ["Date", "City", "AQI", "AQI_Category"]]
pollutant_choice = st.sidebar.selectbox("Pollutant for trend", pollutants if pollutants else ["PM2.5"], index=0)

# ---------------- Top row ----------------
g1, g2 = st.columns([1, 2])

with g1:
    today = pd.to_datetime(pd.Timestamp.now().floor("D"))
    if today in list(df_city_fore["Date"]):
        latest_row = df_city_fore[df_city_fore["Date"] == today].iloc[0]
    else:
        latest_row = df_city_fore.sort_values("Date").iloc[0]

    aqi_val = latest_row.get("AQI", np.nan)
    aqi_cat = latest_row.get("AQI_Category", "Unknown")

    st.subheader("Current Air Quality")
    st.plotly_chart(make_gauge(aqi_val, aqi_cat), use_container_width=True)

with g2:
    st.subheader("7-Day Forecast")
    df7 = df_city_fore.sort_values("Date").head(7)
    format_forecast_boxes(df7)

# ---------------- Pollutant Trends + Alerts (Side-by-Side) ----------------
st.markdown("---")
st.subheader("Pollutant Analysis & Alerts")

col_left, col_right = st.columns([2, 1])

# -------- Left Column: Pollutant Concentrations --------
with col_left:
    st.subheader("Pollutant Concentrations (48 hrs)")

    if df_raw is not None:
        recent = df_raw[df_raw["City"] == city].sort_values("Date").tail(48)

        if not recent.empty:
            present_pollutants = [
                p for p in ["PM2.5", "PM10", "NO2", "O3", "SO2"]
                if p in recent.columns
            ]

            if present_pollutants:
                df_m = (
                    recent.set_index("Date")[present_pollutants]
                    .resample("H")
                    .mean()
                    .interpolate()
                    .reset_index()
                )
                fig2 = px.line(
                    df_m,
                    x="Date",
                    y=present_pollutants,
                    title="Recent Pollutant Concentrations",
                    labels={"value": "Concentration", "variable": "Pollutant"},
                )
                st.plotly_chart(fig2, use_container_width=True)
        else:
            st.warning("Not enough raw data to display pollutant trends.")
    else:
        st.info("Raw pollutant history (air_quality_data.csv) not available.")


# -------- Right Column: Alerts --------
with col_right:
    st.subheader("Active Alerts (Next 7 Days)")

    alert_df = df_city_fore[
        df_city_fore["AQI_Category"].isin([
            "Unhealthy for Sensitive",
            "Unhealthy",
            "Very Unhealthy",
            "Hazardous"
        ])
    ]

    if alert_df.empty:
        st.info("No active alerts in the forecast.")
    else:
        for _, r in alert_df.sort_values("Date").iterrows():
            date_label = r["Date"].strftime("%a, %d %b")
            cat = r["AQI_Category"]
            aqi_v = int(r["AQI"]) if not np.isnan(r["AQI"]) else "NA"

            st.markdown(
                f"""
                <div style="border-left:4px solid {category_color(cat)};
                            padding:10px;
                            margin-bottom:8px;
                            border-radius:4px;
                            background:#fff;">
                    <div style="font-weight:600">{cat} — {date_label}</div>
                    <div style="font-size:13px">Predicted AQI: <strong>{aqi_v}</strong></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
