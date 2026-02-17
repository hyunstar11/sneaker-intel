"""Release Strategy dashboard page."""

from __future__ import annotations

import streamlit as st

from sneaker_intel.analysis.release_strategy import ReleaseStrategyAnalyzer
from sneaker_intel.data import DatasetType, load_dataset
from sneaker_intel.visualization.style import apply_nike_style

apply_nike_style()

st.title("Release Strategy Analysis")

try:
    df = load_dataset(DatasetType.MARKET_2023)
except FileNotFoundError:
    st.error("Market 2023 dataset not found.")
    st.stop()

analyzer = ReleaseStrategyAnalyzer(df)

# Timing
st.subheader("Release Timing Analysis")
timing = analyzer.timing_analysis()
if not timing.empty:
    st.dataframe(timing, use_container_width=True)
    st.bar_chart(timing["avg_premium"])
else:
    st.info("No release date data available for timing analysis.")

# Pricing sensitivity
st.subheader("Pricing Sensitivity")
pricing = analyzer.pricing_sensitivity()
st.dataframe(pricing, use_container_width=True)

# Brand comparison
st.subheader("Brand Comparison")
brands = analyzer.brand_comparison()
st.dataframe(brands.head(20), use_container_width=True)
