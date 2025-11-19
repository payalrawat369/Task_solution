'''# =========================================================
# 🌿 AIR QUALITY DATA EXPLORER + FORECAST DASHBOARD
# =========================================================

import streamlit as st

# Handle query params using new API
params = st.query_params

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os
import joblib
from datetime import timedelta

# ----------------------------
# SETUP
# ----------------------------
st.set_page_config(page_title="Air Quality Dashboard", layout="wide")

st.markdown("""
    <h1 style='color:#2e7d32;'>🌿 Air Quality Data Explorer</h1>
""", unsafe_allow_html=True)

# ----------------------------
# LOAD DATA
# ----------------------------
if os.path.exists("artifacts/air_quality_data_cleaned.csv"):
    DATA_PATH = "artifacts/air_quality_data_cleaned.csv"
elif os.path.exists("air_quality_data_cleaned.csv"):
    DATA_PATH = "air_quality_data_cleaned.csv"
else:
    st.error("❌ Cleaned dataset not found. Please run your training pipeline first (air_quality_pipeline.py).")
    st.stop()

df = pd.read_csv(DATA_PATH)
datetime_col = df.columns[0]
df[datetime_col] = pd.to_datetime(df[datetime_col])
df = df.sort_values(by=datetime_col)

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
city_cols = [c for c in df.columns if any(x in c.lower() for x in ['city', 'station', 'location'])]
group_col = city_cols[0] if city_cols else None

# ----------------------------
# SIDEBAR FILTERS
# ----------------------------
st.sidebar.header("🧭 Data Controls")

if group_col:
    locations = st.sidebar.multiselect("Location", df[group_col].dropna().unique(), default=df[group_col].dropna().unique()[:1])
else:
    locations = None

time_range = st.sidebar.selectbox("Time Range", ["All", "Last 30 Days", "Last 7 Days"])
pollutant = st.sidebar.selectbox("Pollutant", numeric_cols)
apply = st.sidebar.button("Apply Filters")

# ----------------------------
# FILTER DATA
# ----------------------------
filtered = df.copy()
if locations and group_col:
    filtered = filtered[filtered[group_col].isin(locations)]

if time_range == "Last 30 Days":
    filtered = filtered[filtered[datetime_col] >= (filtered[datetime_col].max() - pd.Timedelta(days=30))]
elif time_range == "Last 7 Days":
    filtered = filtered[filtered[datetime_col] >= (filtered[datetime_col].max() - pd.Timedelta(days=7))]

# ----------------------------
# TIME SERIES CHART
# ----------------------------
st.subheader(f"📈 {pollutant} Time Series")

filtered["Smoothed"] = filtered[pollutant].rolling(window=10, center=True).mean()

fig = px.line(
    filtered,
    x=datetime_col,
    y="Smoothed",
    title=f"{pollutant} Concentration Over Time",
    line_shape='spline',
    markers=True
)
fig.update_layout(
    template="plotly_white",
    xaxis_title="Date",
    yaxis_title=f"{pollutant} (µg/m³)",
    title_x=0.5,
    plot_bgcolor="rgba(240,248,255,0.6)",
    paper_bgcolor="rgba(240,248,255,0.6)",
    font=dict(size=14, color="#2E4A32"),
    margin=dict(l=40, r=40, t=60, b=40),
)
fig.update_traces(line=dict(color="#2E8B57", width=3))
st.plotly_chart(fig, use_container_width=True)

st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(to bottom right, #e8f5e9, #f1f8e9);
    }
    </style>
""", unsafe_allow_html=True)

# ----------------------------
# CORRELATION HEATMAP (as bubble chart)
# ----------------------------
st.subheader("🔗 Pollutant Correlations")
pollutant_cols = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']
data = df.dropna(subset=pollutant_cols)

corr = data[pollutant_cols].corr().round(2)
corr_long = corr.stack().reset_index()
corr_long.columns = ['Pollutant_X', 'Pollutant_Y', 'Correlation']
corr_long = corr_long[corr_long['Pollutant_X'] != corr_long['Pollutant_Y']]

fig = px.scatter(
    corr_long,
    x='Pollutant_X',
    y='Pollutant_Y',
    size=corr_long['Correlation'].abs() * 40,
    color='Correlation',
    color_continuous_scale='Greens',
    range_color=[-1, 1],
    hover_data={'Correlation': True},
    title='Pollutant Correlations',
)
fig.update_traces(marker=dict(opacity=0.9, line=dict(width=1.2, color='DarkGreen')))
fig.update_layout(
    title=dict(x=0.5, font=dict(size=20, color='#2E4A32', family="Arial Black")),
    xaxis=dict(title='', showgrid=True, gridcolor='rgba(200,200,200,0.2)'),
    yaxis=dict(title='', showgrid=True, gridcolor='rgba(200,200,200,0.2)'),
    plot_bgcolor='rgba(248, 250, 245, 0.9)',
    paper_bgcolor='rgba(248, 250, 245, 0.9)',
    font=dict(size=13, color='#2E4A32'),
    margin=dict(l=40, r=40, t=60, b=40)
)
st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# STATISTICAL SUMMARY
# ----------------------------
st.subheader("📊 Statistical Summary")
mean_val = filtered[pollutant].mean()
median_val = filtered[pollutant].median()
max_val = filtered[pollutant].max()
min_val = filtered[pollutant].min()
std_val = filtered[pollutant].std()
count_val = len(filtered)

col1, col2, col3 = st.columns(3)
col4, col5, col6 = st.columns(3)

col1.metric("Mean (µg/m³)", f"{mean_val:.1f}")
col2.metric("Median (µg/m³)", f"{median_val:.1f}")
col3.metric("Max (µg/m³)", f"{max_val:.1f}")
col4.metric("Min (µg/m³)", f"{min_val:.1f}")
col5.metric("Std Dev", f"{std_val:.1f}")
col6.metric("Data Points", f"{count_val}")

st.markdown("""
    <style>
    div[data-testid="metric-container"] {
        background-color: #e8f5e9;
        border: 1px solid #c8e6c9;
        padding: 10px 0;
        border-radius: 10px;
        text-align: center;
        box-shadow: 1px 1px 3px rgba(0,0,0,0.1);
    }
    div[data-testid="stMetricValue"] {
        color: #2e7d32;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------------------
# DISTRIBUTION CHART
# ----------------------------
st.subheader(f"📦 {pollutant} Distribution Analysis")
fig2 = px.histogram(filtered, x=pollutant, nbins=20, title=f"Distribution of {pollutant}", color_discrete_sequence=["#43a047"])
st.plotly_chart(fig2, use_container_width=True)

# ----------------------------
# DATA QUALITY
# ----------------------------
st.sidebar.header("📊 Data Quality")
completeness = 100 * (1 - df.isna().sum().sum() / (df.shape[0] * df.shape[1]))
validity = 100 - (df.duplicated().sum() / len(df) * 100)
st.sidebar.progress(int(completeness))
st.sidebar.text(f"Completeness: {completeness:.1f}%")
st.sidebar.progress(int(validity))
st.sidebar.text(f"Validity: {validity:.1f}%")
'''
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(page_title="Air Quality Data Explorer", layout="wide")

# Header
st.markdown("""
    <h1 style='text-align:center; color:#2e7d32;'>
        🌿 Air Quality Data Explorer
    </h1>
    
""", unsafe_allow_html=True)

# Background styling
st.markdown("""
<style>
.stApp {
    background: linear-gradient(to bottom right, #e3f2e1, #f1f8e9);
}
</style>
""", unsafe_allow_html=True)



# =========================================================
# LOAD DATA
# =========================================================
if os.path.exists("artifacts/air_quality_data_cleaned.csv"):
    DATA_PATH = "artifacts/air_quality_data_cleaned.csv"
elif os.path.exists("air_quality_data_cleaned.csv"):
    DATA_PATH = "air_quality_data_cleaned.csv"
else:
    st.error("❌ Dataset not found. Please run your pipeline first.")
    st.stop()

df = pd.read_csv(DATA_PATH)
datetime_col = df.columns[0]
df[datetime_col] = pd.to_datetime(df[datetime_col])
df = df.sort_values(by=datetime_col)



# =========================================================
# SIDEBAR — DATA CONTROLS
# =========================================================
st.sidebar.markdown("### 🧭 Data Controls")

city_cols = [c for c in df.columns if any(x in c.lower() for x in ['city', 'station', 'location'])]
group_col = city_cols[0] if city_cols else None

if group_col:
    locations = st.sidebar.selectbox("Location", df[group_col].dropna().unique())
else:
    locations = None

time_range = st.sidebar.selectbox("Time Range", ["All", "Last 24 Hours", "Last 7 Days", "Last 30 Days"])

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
pollutant = st.sidebar.selectbox("Pollutants", numeric_cols)

apply = st.sidebar.button("Apply Filters")

# Data Quality
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Data Quality")

completeness = 100 * (1 - df.isna().sum().sum() / (df.size))
validity = 100 - (df.duplicated().sum() / len(df) * 100)

st.sidebar.text(f"Completeness: {completeness:.1f}%")
st.sidebar.progress(int(completeness))
st.sidebar.text(f"Validity: {validity:.1f}%")
st.sidebar.progress(int(validity))



# =========================================================
# APPLY FILTERS
# =========================================================
filtered = df.copy()

if locations and group_col:
    filtered = filtered[filtered[group_col] == locations]

if time_range == "Last 24 Hours":
    filtered = filtered[filtered[datetime_col] >= (filtered[datetime_col].max() - pd.Timedelta(hours=24))]
elif time_range == "Last 7 Days":
    filtered = filtered[filtered[datetime_col] >= (filtered[datetime_col].max() - pd.Timedelta(days=7))]
elif time_range == "Last 30 Days":
    filtered = filtered[filtered[datetime_col] >= (filtered[datetime_col].max() - pd.Timedelta(days=30))]



# =========================================================
# 1️⃣ TIME SERIES + POLLUTANT CORRELATIONS (SIDE-BY-SIDE)
# =========================================================
ts_col, corr_col = st.columns(2)

# ---- LEFT: TIME SERIES ----
with ts_col:
    st.markdown("### PM2.5 Time Series")
    filtered["Smoothed"] = filtered[pollutant].rolling(8, center=True).mean()

    fig_ts = px.line(
        filtered,
        x=datetime_col,
        y="Smoothed",
        markers=True,
        title="",
    )
    fig_ts.update_traces(line=dict(color="#2e7d32", width=3))
    fig_ts.update_layout(
        template="simple_white",
        xaxis_title="",
        yaxis_title="Concentration (µg/m³)",
        height=350,
        plot_bgcolor="rgba(255,255,255,0.7)",
    )
    st.plotly_chart(fig_ts, use_container_width=True)

# ---- RIGHT: POLLUTANT CORRELATIONS ----
with corr_col:
    st.markdown("### Pollutant Correlations")

    pollutant_cols = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']
    data2 = df.dropna(subset=pollutant_cols)
    corr = data2[pollutant_cols].corr().round(2)

    corr_long = corr.stack().reset_index()
    corr_long.columns = ["Pollutant_X", "Pollutant_Y", "Correlation"]
    corr_long = corr_long[corr_long["Pollutant_X"] != corr_long["Pollutant_Y"]]

    fig_corr = px.scatter(
        corr_long,
        x="Pollutant_X",
        y="Pollutant_Y",
        size=corr_long["Correlation"].abs() * 40,
        color="Correlation",
        color_continuous_scale="Greens",
    )

    fig_corr.update_layout(
        height=350,
        xaxis_title="",
        yaxis_title="",
        plot_bgcolor="rgba(255,255,255,0.7)",
    )
    st.plotly_chart(fig_corr, use_container_width=True)



# =========================================================
# 2️⃣ STATISTICAL SUMMARY + DISTRIBUTION ANALYSIS
# =========================================================
left, right = st.columns(2)

with left:
    st.markdown("### Statistical Summary")
    mean_v = filtered[pollutant].mean()
    med_v = filtered[pollutant].median()
    max_v = filtered[pollutant].max()
    min_v = filtered[pollutant].min()
    sd_v  = filtered[pollutant].std()
    count_v = len(filtered)

    c1, c2 = st.columns(2)
    c1.metric("Mean (µg/m³)", f"{mean_v:.1f}")
    c2.metric("Median (µg/m³)", f"{med_v:.1f}")

    c3, c4 = st.columns(2)
    c3.metric("Max (µg/m³)", f"{max_v:.1f}")
    c4.metric("Min (µg/m³)", f"{min_v:.1f}")

    c5, c6 = st.columns(2)
    c5.metric("Std Dev", f"{sd_v:.1f}")
    c6.metric("Data Points", count_v)

with right:
    st.markdown(f"### Distribution Analysis")
    fig_hist = px.histogram(
        filtered,
        x=pollutant,
        nbins=20,
        color_discrete_sequence=["#2e7d32"],
    )
    fig_hist.update_layout(
        height=350,
        xaxis_title=f"{pollutant} Range (µg/m³)",
        yaxis_title="Frequency",
        plot_bgcolor="rgba(255,255,255,0.7)",
    )
    st.plotly_chart(fig_hist, use_container_width=True)

