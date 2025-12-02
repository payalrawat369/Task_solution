import streamlit as st

st.set_page_config(page_title="Air Quality Dashboard", layout="wide")

# -----------------------------
# PAGE HEADER
# -----------------------------
st.markdown(
    """
    <h1 style='text-align: center; color: #2c3e50;'>
        🌫️ Air Quality Monitoring Dashboard
    </h1>
    <p style='text-align: center; color: gray; font-size:18px;'>
        Explore, Analyze, and Forecast Air Quality Data
    </p>
    """,
    unsafe_allow_html=True
)

st.write("")
st.write("")

# -----------------------------
#   CARD STYLE FUNCTION
# -----------------------------
card_style = """
    <div style="
        background-color: #ffffff;
        padding: 30px;
        border-radius: 20px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        text-align: center;HOME
        border: 2px solid #f0f0f0;
        transition: 0.3s;
    ">
        <h2 style="color:#34495e;">{title}</h2>
        <p style="color:gray;">{desc}</p>
    </div>
"""

# -----------------------------
#   LAYOUT – 4 COLUMNS
# -----------------------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(card_style.format(
        title="📊 Explore Data",
        desc="View city-wise pollutant levels, summary, and statistics."
    ), unsafe_allow_html=True)

    if st.button("Go ➜ Explore Data"):
        st.switch_page("pages/Explore_data.py")

with col2:
    st.markdown(card_style.format(
        title="📈 View Forecasts",
        desc="Check time-series predictions for PM2.5 and other pollutants."
    ), unsafe_allow_html=True)

    if st.button("Go ➜ View Forecasts"):
        st.switch_page("pages/view_forecasts.py")

with col3:
    st.markdown(card_style.format(
        title="🌍 Air Quality Check",
        desc="AQI insights and pollution category classification."
    ), unsafe_allow_html=True)

    if st.button("Go ➜ Check AQI"):
        st.switch_page("pages/ Air_Quality_Check.py")

with col4:
    st.markdown(card_style.format(
        title="🔧 Retrain Model",
        desc="check AQI,upload files."
    ), unsafe_allow_html=True)

    if st.button("Go ➜ Retrain Model"):
        st.switch_page("pages/Retrain_model.py")



# -----------------------------
# FOOTER
# -----------------------------
st.write("")
st.markdown(
    "<p style='text-align:center; color:gray;'>Developed for Air Quality Analysis Project</p>",
    unsafe_allow_html=True
)
