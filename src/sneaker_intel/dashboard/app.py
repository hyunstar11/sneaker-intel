"""Streamlit dashboard entry point."""

from __future__ import annotations

import streamlit as st

from sneaker_intel import __version__

st.set_page_config(
    page_title="Footwear Launch Intelligence",
    page_icon="👟",
    layout="wide",
)

st.title("Footwear Launch Intelligence Platform")
st.markdown(f"*Version {__version__}*")
st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Models Trained", "4", help="Linear, RF, XGBoost, LightGBM + Ensemble")

with col2:
    st.metric("StockX Transactions", "99K+", help="StockX Data Contest 2019")

with col3:
    st.metric("Market Products", "2K+", help="Sneakers 2023 dataset")

st.markdown("---")
st.markdown(
    """
    ### Navigate to:
    - **Market Overview** — Brand breakdown, premium distribution, volatility
    - **Launch Forecaster** — Demand forecasting with strategic recommendations
    - **Demand Insights** — Demand tier segmentation and drivers
    - **Launch Strategy** — Timing analysis and pricing insights
    """
)
