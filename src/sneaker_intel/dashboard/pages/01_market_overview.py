"""Market Overview dashboard page."""

from __future__ import annotations

import streamlit as st

from sneaker_intel.analysis.market_dynamics import MarketDynamicsAnalyzer
from sneaker_intel.data import DatasetType, load_dataset
from sneaker_intel.features.market import MarketDynamicsExtractor
from sneaker_intel.visualization.style import apply_nike_style

apply_nike_style()

st.title("Market Overview")

try:
    df = load_dataset(DatasetType.MARKET_2023)
except FileNotFoundError:
    st.error("Market 2023 dataset not found. Place sneakers2023.csv in data/raw/.")
    st.stop()

analyzer = MarketDynamicsAnalyzer(df)

# KPIs
overview = analyzer.overview()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Products", overview["total_products"])
c2.metric("Brands", overview["total_brands"])
c3.metric("Avg Premium", f"{overview['avg_premium']:.1%}")
c4.metric("Avg Volatility", f"{overview['avg_volatility']:.4f}")

st.markdown("---")

# Brand comparison
st.subheader("Brand Comparison")
brand_df = analyzer.liquidity_analysis()
st.dataframe(brand_df, use_container_width=True)

# Volatility drivers
st.subheader("High-Variance Products")
volatile_df = analyzer.volatility_drivers(top_n=15)
st.dataframe(volatile_df, use_container_width=True)

# Market derived features
st.subheader("Market Dynamics Features")
extractor = MarketDynamicsExtractor()
enriched = extractor.extract(df)
st.dataframe(
    enriched[["item", "brand"] + extractor.feature_names].head(20),
    use_container_width=True,
)
