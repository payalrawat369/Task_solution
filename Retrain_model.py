
import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime

import code_run1   # your model training file

# -------------------------------------------------------
# App Config
# -------------------------------------------------------
st.set_page_config(page_title="Air Quality Prediction System", layout="wide")


# -------------------------------------------------------
# Load forecast file
# -------------------------------------------------------
@st.cache_data
def load_forecast():
    try:
        df = pd.read_csv("forecast_aqi.csv")
        return df
    except:
        return pd.DataFrame()


forecast_df = load_forecast()


# -------------------------------------------------------
# AQI Calculation Logic
# -------------------------------------------------------
def calculate_aqi(row):
    breakpoints = {
        'PM2.5': [(0,30,0,50),(31,60,51,100),(61,90,101,200),(91,120,201,300),(121,250,301,400),(251,500,401,500)],
        'PM10':  [(0,50,0,50),(51,100,51,100),(101,250,101,200),(251,350,201,300),(351,430,301,400),(431,600,401,500)],
        'NO2':   [(0,40,0,50),(41,80,51,100),(81,180,101,200),(181,280,201,300),(281,400,301,400),(401,1000,401,500)],
        'O3':    [(0,50,0,50),(51,100,51,100),(101,168,101,200),(169,208,201,300),(209,748,301,400),(749,1000,401,500)],
        'SO2':   [(0,40,0,50),(41,80,51,100),(81,380,101,200),(381,800,201,300),(801,1600,301,400),(1601,2000,401,500)]
    }

    pollutants = ['PM2.5','PM10','NO2','O3','SO2']
    sub_indices = []

    for p in pollutants:
        if p in row and not pd.isna(row[p]):
            for bp in breakpoints[p]:
                if bp[0] <= row[p] <= bp[1]:
                    aqi = ((bp[3]-bp[2])/(bp[1]-bp[0]))*(row[p]-bp[0])+bp[2]
                    sub_indices.append(aqi)
                    break

    return max(sub_indices) if sub_indices else np.nan


def aqi_category(aqi):
    if aqi <= 50:
        return "Good", "Air is clean and healthy."
    elif aqi <= 100:
        return "Moderate", "Acceptable; slight risk for sensitive people."
    elif aqi <= 200:
        return "Unhealthy for Sensitive Groups", "Sensitive groups should reduce outdoor time."
    elif aqi <= 300:
        return "Unhealthy", "Everyone may feel health effects."
    elif aqi <= 400:
        return "Very Unhealthy", "Serious health effects for all."
    else:
        return "Hazardous", "Health emergency; avoid outdoor activities."


# -------------------------------------------------------
# DATA UPLOAD & TRAINING (OPEN ACCESS)
# -------------------------------------------------------
def data_training_section():
    st.header("📂 Upload Data & Retrain Models (Open Access)")

    uploaded = st.file_uploader("Upload new air quality CSV", type=["csv"])

    if uploaded:
        new_df = pd.read_csv(uploaded)
        st.write("Uploaded Data Preview:", new_df.head())

        if st.button("Append to Dataset"):
            if os.path.exists("air_quality_data.csv"):
                old_df = pd.read_csv("air_quality_data.csv")
                full_df = pd.concat([old_df, new_df], ignore_index=True)
            else:
                full_df = new_df

            full_df.to_csv("air_quality_data.csv", index=False)
            st.success("✅ Data appended successfully!")

    if st.button("🔄 Retrain All Models"):
        with st.spinner("Training models... Please wait..."):
            try:
                code_run1.main()
                st.success("🎉 Training completed! Forecast updated.")
            except Exception as e:
                st.error(f"Training failed: {e}")

        st.cache_data.clear()
        st.experimental_rerun()


# -------------------------------------------------------
# MAIN USER INTERFACE
# -------------------------------------------------------
def user_interface():
    st.title("🌤️ Air Quality Prediction System")

    col1, col2 = st.columns(2)

    with col1:
        date = st.date_input("Date:")
        pm25 = st.number_input("PM2.5", min_value=0.0)
        co = st.number_input("CO", min_value=0.0)
        so2 = st.number_input("SO₂", min_value=0.0)
        temp = st.number_input("Temperature (°C)", min_value=-10.0)

    with col2:
        station = st.selectbox("Station:", forecast_df["City"].unique() if not forecast_df.empty else ["Delhi"])
        pm10 = st.number_input("PM10", min_value=0.0)
        no2 = st.number_input("NO₂", min_value=0.0)
        o3 = st.number_input("O₃", min_value=0.0)
        humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0)

    st.markdown("---")

    if st.button("🔮 Predict AQI"):
        row = {"PM2.5": pm25, "PM10": pm10, "NO2": no2, "O3": o3, "SO2": so2}
        aqi_val = calculate_aqi(row)

        if pd.isna(aqi_val):
            st.error("Invalid values. Cannot calculate AQI.")
        else:
            cat, msg = aqi_category(aqi_val)
            st.success(f"Predicted AQI: {aqi_val:.1f}")
            st.info(f"Category: {cat}")
            st.warning(f"Advice: {msg}")

    if st.button("🔍 Search 7-day Forecast"):
        if forecast_df.empty:
            st.error("No forecast available! Please retrain models.")
        else:
            city_df = forecast_df[forecast_df["City"] == station].sort_values("Date")
            st.subheader(f"📅 7-Day Forecast for {station}")
            st.dataframe(city_df[["Date", "AQI", "AQI_Category"]].tail(7))


# -------------------------------------------------------
# PAGE ROUTING
# -------------------------------------------------------
menu = st.sidebar.radio("Menu", ["User Panel", "Data Upload & Training"])

if menu == "User Panel":
    user_interface()
else:
    data_training_section()
