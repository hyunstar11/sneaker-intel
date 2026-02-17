"""Tests for sneaker type feature extraction."""

from __future__ import annotations

import pandas as pd

from sneaker_intel.features.sneaker_type import SneakerTypeExtractor


def test_sneaker_type_extraction(cleaned_stockx_df: pd.DataFrame) -> None:
    extractor = SneakerTypeExtractor()
    result = extractor.extract(cleaned_stockx_df)

    # Yeezy-Boost-350 should match
    yeezy_row = result[result["Sneaker Name"].str.contains("Yeezy-Boost-350")]
    assert yeezy_row["yeezy350"].iloc[0] == 1

    # Air-Jordan should match
    jordan_row = result[result["Sneaker Name"].str.contains("Air-Jordan")]
    assert jordan_row["airjordan"].iloc[0] == 1


def test_feature_names() -> None:
    extractor = SneakerTypeExtractor()
    names = extractor.feature_names
    assert "yeezy350" in names
    assert "airjordan" in names
    assert len(names) == 10
