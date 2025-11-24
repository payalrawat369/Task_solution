import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(page_title="Air Quality Forecast Engine", layout="wide")
st.title("🌤️ Air Quality Forecast Engine")

# ---- Load required files ----
REQUIRED = ["results_all.csv", "best_models.csv", "horizon_24h_accuracy.csv", "cleaned_air_quality_hourly.csv"]

missing = [f for f in REQUIRED if not os.path.exists(f)]
if missing:
    st.error("❌ Missing required file(s): " + ", ".join(missing) +
             "\n\nRun `train_and_save_models.py` first.")
    st.stop()

results = pd.read_csv("results_all.csv")
best = pd.read_csv("best_models.csv")
horizon_df = pd.read_csv("horizon_24h_accuracy.csv")
df_all = pd.read_csv("cleaned_air_quality_hourly.csv")

df_all["Date"] = pd.to_datetime(df_all["Date"], errors="coerce")
df_all = df_all.dropna(subset=["Date"]).sort_values("Date")

if os.path.exists("forecast_aqi.csv"):
    forecast_aqi = pd.read_csv("forecast_aqi.csv")
else:
    forecast_aqi = pd.DataFrame()

#Sidebar 
st.sidebar.header("⚙️ Filters")

selected_city = st.sidebar.selectbox("Select City", df_all["City"].unique())

df_city = df_all[df_all["City"] == selected_city]
available_pollutants = [c for c in ["PM2.5","PM10","NO2","SO2","CO","O3"] if c in df_city.columns]

selected_pollutant = st.sidebar.selectbox("Select Pollutant", available_pollutants)

metric = st.sidebar.radio("Metric", ["RMSE","MAE"], horizontal=True)

col1, col2 = st.columns(2)
col3, col4 = st.columns(2)

with col1:
    st.subheader(f"📊 Model Performance — {selected_city} (All Pollutants)")

    filtered_city = results[results["City"] == selected_city]

    if filtered_city.empty:
        st.info("No model results for this city.")
    else:
        pivot_df = filtered_city.pivot_table(
            index="Pollutant",
            columns="Model",
            values=metric,
            aggfunc="mean"
        ).reset_index()

        melted = pivot_df.melt(
            id_vars="Pollutant",
            value_vars=pivot_df.columns[1:],  # model names
            var_name="Model",
            value_name=metric
        )

        fig = px.bar(
            melted,
            x="Pollutant",
            y=metric,
            color="Model",
            barmode="group",
            text_auto=".2f",
            title=f"{metric} Comparison Across Pollutants — {selected_city}"
        )

        fig.update_layout(
            xaxis_title="Pollutant",
            yaxis_title=metric,
            template="plotly_white"
        )

        st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader(f"📈 Forecast vs Actual — {selected_pollutant} — {selected_city}")

    available_models = ["ARIMA", "Prophet", "XGBoost"]
    selected_model = st.selectbox("Select Forecast Model", available_models, key="model_fx")

    horizon_option = st.selectbox(
        "Select Forecast Horizon",
        ["30 days","All time","24 hours", "7 days"],
        key="horizon_fx"
    )

    df_plot = df_city[["Date", selected_pollutant]].dropna().sort_values("Date")

    if os.path.exists("forecast_aqi.csv"):
        fc = pd.read_csv("forecast_aqi.csv")
        fc["Date"] = pd.to_datetime(fc["Date"], errors="coerce")
        fc = fc[(fc["City"] == selected_city)]
    else:
        fc = pd.DataFrame()

    if horizon_option == "24 hours":
        h = 24
    elif horizon_option == "7 days":
        h = 7 * 24
    elif horizon_option == "30 days":
        h = 30 * 24
    else:
        h = len(df_plot)

    actual_slice = df_plot.iloc[-h:] if len(df_plot) >= h else df_plot

    if not fc.empty and selected_pollutant in fc.columns:
        forecast_slice = fc[["Date", selected_pollutant]].dropna()
    else:
        forecast_slice = pd.DataFrame()

    fig_fx = go.Figure()

    fig_fx.add_trace(go.Scatter(
        x=actual_slice["Date"],
        y=actual_slice[selected_pollutant],
        mode="lines",
        name="Actual",
        line=dict(width=3)
    ))

    if not forecast_slice.empty:
        fig_fx.add_trace(go.Scatter(
            x=forecast_slice["Date"],
            y=forecast_slice[selected_pollutant],
            mode="lines+markers",
            name=f"{selected_model} Forecast",
            line=dict(dash='dash')
        ))
    else:
        st.warning("⚠️ Forecast not available for this horizon. Only 24-hour forecast supported currently.")

    fig_fx.update_layout(
        title=f"{selected_pollutant}: Forecast vs Actual — {selected_model} — {horizon_option}",
        xaxis_title="Date",
        yaxis_title=f"{selected_pollutant} (µg/m³)",
        template="plotly_white"
    )

    st.plotly_chart(fig_fx, use_container_width=True)

with col3:
    st.subheader(f"🏆 Best Models — {selected_city}")

    best_city = best[best["City"] == selected_city]

    if best_city.empty:
        st.info("No best-model data available for this city.")
    else:
        st.dataframe(best_city, use_container_width=True)

with col4:
    st.subheader(f"📉 Forecast Accuracy — {selected_city} — {selected_pollutant}")

    metric_col = "RMSE" if metric == "RMSE" else "MAE"

    df_h = horizon_df[(horizon_df["City"] == selected_city) &
                      (horizon_df["Pollutant"] == selected_pollutant)]

    if df_h.empty:
        st.info("No accuracy data available.")
    else:
        horizons = ["1h", "3h", "6h", "12h", "24h"]
        horizon_scale = [0.65, 0.75, 0.85, 0.95, 1.0]

        fig_acc = go.Figure()

        for model in df_h["Model"].unique():
            base_val = df_h[df_h["Model"] == model][metric_col].values[0]
            y_vals = [round(base_val * s, 2) for s in horizon_scale]

            fig_acc.add_trace(go.Scatter(
                x=horizons,
                y=y_vals,
                mode="lines+markers",
                name=model,
                line=dict(width=3),
                marker=dict(size=8)
            ))

        fig_acc.update_layout(
            title=f"{metric} Across Forecast Horizons — Simulated Trend",
            xaxis_title="Forecast Horizon",
            yaxis_title=f"{metric} (lower is better)",
            template="plotly_white"
        )

        st.plotly_chart(fig_acc, use_container_width=True)

