"""Demand Insights dashboard page."""

from __future__ import annotations

import streamlit as st

from sneaker_intel.analysis.demand_forecast import DemandSegmenter, analyze_demand_drivers
from sneaker_intel.data import DatasetType, load_dataset
from sneaker_intel.visualization.style import apply_nike_style

apply_nike_style()

st.title("Demand Insights")

try:
    df = load_dataset(DatasetType.MARKET_2023)
except FileNotFoundError:
    st.error("Market 2023 dataset not found.")
    st.stop()

segmenter = DemandSegmenter()
df = segmenter.fit_predict(df)

# Tier distribution
st.subheader("Demand Tier Distribution")
tier_counts = df["demand_tier"].value_counts()
st.bar_chart(tier_counts)

# Tier summary
st.subheader("Tier Summary Statistics")
summary = segmenter.get_tier_summary(df)
st.dataframe(summary, use_container_width=True)

# Demand drivers
st.subheader("Demand Drivers")
drivers = analyze_demand_drivers(df)
st.dataframe(drivers, use_container_width=True)

# Scatter: premium vs sales by tier
st.subheader("Premium vs Sales by Demand Tier")
st.scatter_chart(
    df,
    x="pricePremium",
    y="salesThisPeriod",
    color="demand_tier",
)
