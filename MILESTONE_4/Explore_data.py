
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os


st.set_page_config(page_title="Air Quality Data Explorer", layout="wide")

# Header
st.markdown("""
    <h1 style='text-align:center; color:#2e7d32;'>
        Air Quality Data Explorer
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




DATA_PATH = "cleaned_air_quality_hourly.csv"


df = pd.read_csv(DATA_PATH)
datetime_col = df.columns[0]
df[datetime_col] = pd.to_datetime(df[datetime_col])
df = df.sort_values(by=datetime_col)

# SIDEBAR 

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


# APPLY FILTERS

filtered = df.copy()

if locations and group_col:
    filtered = filtered[filtered[group_col] == locations]

if time_range == "Last 24 Hours":
    filtered = filtered[filtered[datetime_col] >= (filtered[datetime_col].max() - pd.Timedelta(hours=24))]
elif time_range == "Last 7 Days":
    filtered = filtered[filtered[datetime_col] >= (filtered[datetime_col].max() - pd.Timedelta(days=7))]
elif time_range == "Last 30 Days":
    filtered = filtered[filtered[datetime_col] >= (filtered[datetime_col].max() - pd.Timedelta(days=30))]


ts_col, corr_col = st.columns(2)

# TIME SERIES
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

#RIGHT: POLLUTANT CORRELATIONS
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


