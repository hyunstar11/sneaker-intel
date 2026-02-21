"""Launch Demand Forecaster dashboard page."""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd
import streamlit as st

from sneaker_intel.config import settings
from sneaker_intel.data import DatasetType, load_dataset
from sneaker_intel.data.transformers import clean_price_columns, parse_dates
from sneaker_intel.features.pipeline import FeaturePipeline
from sneaker_intel.features.stockx import (
    add_number_of_sales_feature,
    build_number_of_sales_reference,
)
from sneaker_intel.models.ensemble import EnsembleModel
from sneaker_intel.models.random_forest import RandomForestModel

_logger = logging.getLogger(__name__)

st.title("Launch Demand Forecaster")


@st.cache_resource
def load_model():
    """Train and cache the ensemble model."""
    df = load_dataset(DatasetType.STOCKX)
    df = clean_price_columns(df)
    df = parse_dates(df)

    pipeline = FeaturePipeline.stockx_default()
    df = pipeline.transform(df)

    sales_lookup, default_sales = build_number_of_sales_reference(df)
    df = add_number_of_sales_feature(
        df,
        sales_lookup=sales_lookup,
        default_sales=default_sales,
    )

    target = settings.features.target_column
    drop_cols = [
        "Brand",
        "Buyer Region",
        "Shoe Size",
        "Sneaker Name",
        "Order Date",
        "Release Date",
    ]
    features_df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    feature_cols = [c for c in features_df.columns if c != target]

    X = features_df[feature_cols]
    y = features_df[target]

    models = [RandomForestModel()]
    for module_path, class_name in [
        ("sneaker_intel.models.xgboost_model", "XGBoostModel"),
        ("sneaker_intel.models.lightgbm_model", "LightGBMModel"),
    ]:
        try:
            from importlib import import_module

            mod = import_module(module_path)
            models.append(getattr(mod, class_name)())
        except Exception as exc:
            _logger.warning("Skipping optional model %s: %s", class_name, exc)

    ensemble = EnsembleModel(models=models)
    ensemble.fit(X, y)
    return ensemble, pipeline, feature_cols, sales_lookup, default_sales


try:
    ensemble, pipeline, feature_cols, sales_lookup, default_sales = load_model()
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# Input form
with st.form("forecast_form"):
    sneaker_name = st.text_input("Product Name", "Adidas-Yeezy-Boost-350-V2-Cream-White")
    col1, col2 = st.columns(2)
    with col1:
        retail_price = st.number_input("Launch Retail Price ($)", value=220.0, min_value=1.0)
        shoe_size = st.number_input("Shoe Size", value=10.0, min_value=1.0, max_value=18.0)
    with col2:
        buyer_region = st.selectbox(
            "Target Market",
            ["California", "New York", "Oregon", "Florida", "Texas", "Other"],
        )
        order_date = st.date_input("Forecast Date", value=date.today())
        release_date = st.date_input("Release Date", value=date(2018, 1, 1))

    submitted = st.form_submit_button("Forecast Demand")

if submitted:
    input_df = pd.DataFrame(
        [
            {
                "Sneaker Name": sneaker_name,
                "Retail Price": retail_price,
                "Shoe Size": shoe_size,
                "Buyer Region": buyer_region if buyer_region != "Other" else "Illinois",
                "Order Date": pd.to_datetime(order_date),
                "Release Date": pd.to_datetime(release_date),
                "Sale Price": 0,
            }
        ]
    )

    transformed = pipeline.transform(input_df)
    transformed = add_number_of_sales_feature(
        transformed,
        sales_lookup=sales_lookup,
        default_sales=default_sales,
    )

    for col in feature_cols:
        if col not in transformed.columns:
            transformed[col] = 0

    prediction = float(ensemble.predict(transformed[feature_cols])[0])

    # Compute demand metrics
    demand_intensity = prediction / retail_price
    thresholds = settings.features
    if demand_intensity >= thresholds.demand_tier_high:
        tier = "High"
    elif demand_intensity >= thresholds.demand_tier_medium:
        tier = "Medium"
    else:
        tier = "Low"

    recommendations = {
        "High": (
            "Strong aftermarket signal suggests high launch demand. "
            "Consider expanded production run with multi-channel distribution (DTC + wholesale)."
        ),
        "Medium": (
            "Moderate demand signal indicates steady sell-through potential. "
            "Standard production volume with DTC-first allocation recommended."
        ),
        "Low": (
            "Conservative demand signal. "
            "Limited release with inventory risk mitigation recommended."
        ),
    }

    c1, c2, c3 = st.columns(3)
    c1.metric("Market Signal", f"${prediction:,.2f}")
    c2.metric("Demand Intensity", f"{demand_intensity:.2f}x")
    c3.metric("Demand Tier", tier)

    st.info(f"**Recommendation:** {recommendations[tier]}")
