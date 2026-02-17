"""Tests for the feature pipeline."""

from __future__ import annotations

import pandas as pd

from sneaker_intel.features.pipeline import FeaturePipeline


def test_stockx_default_pipeline(cleaned_stockx_df: pd.DataFrame) -> None:
    pipeline = FeaturePipeline.stockx_default()
    result = pipeline.transform(cleaned_stockx_df)

    # Should have all pipeline feature names as columns
    for name in pipeline.feature_names:
        assert name in result.columns, f"Missing feature: {name}"


def test_pipeline_preserves_rows(cleaned_stockx_df: pd.DataFrame) -> None:
    pipeline = FeaturePipeline.stockx_default()
    result = pipeline.transform(cleaned_stockx_df)
    assert len(result) == len(cleaned_stockx_df)


def test_pipeline_feature_names() -> None:
    pipeline = FeaturePipeline.stockx_default()
    names = pipeline.feature_names
    assert len(names) > 20  # colors + types + regions + temporal + size
    assert "Colorful" in names
    assert "yeezy350" in names
    assert "California" in names
    assert "Order Year" in names
    assert "size_freq" in names
