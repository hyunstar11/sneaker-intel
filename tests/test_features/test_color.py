"""Tests for color feature extraction."""

from __future__ import annotations

import pandas as pd

from sneaker_intel.features.color import ColorFeatureExtractor


def test_color_extraction(cleaned_stockx_df: pd.DataFrame) -> None:
    extractor = ColorFeatureExtractor()
    result = extractor.extract(cleaned_stockx_df)

    # "Blue-Tint" should trigger Blue
    blue_row = result[result["Sneaker Name"].str.contains("Blue-Tint")]
    assert blue_row["Blue"].iloc[0] == 1

    # "Black" in Off-White-Black
    black_row = result[result["Sneaker Name"].str.contains("Off-White-Black")]
    assert black_row["Black"].iloc[0] == 1


def test_colorful_composite(cleaned_stockx_df: pd.DataFrame) -> None:
    extractor = ColorFeatureExtractor()
    result = extractor.extract(cleaned_stockx_df)
    assert "Colorful" in result.columns
    assert result["Colorful"].dtype in (int, "int64")


def test_feature_names() -> None:
    extractor = ColorFeatureExtractor()
    names = extractor.feature_names
    assert "Colorful" in names
    assert "Black" in names
    assert len(names) == 11  # 10 colors + Colorful
