"""Tests for data transformers."""

from __future__ import annotations

import pandas as pd

from sneaker_intel.data.transformers import clean_price_columns, parse_dates


def test_clean_price_columns(sample_stockx_df: pd.DataFrame) -> None:
    result = clean_price_columns(sample_stockx_df)
    assert result["Sale Price"].dtype == float
    assert result["Retail Price"].dtype == float
    assert result["Sale Price"].iloc[0] == 1097.0
    assert result["Retail Price"].iloc[0] == 220.0


def test_clean_price_preserves_other_columns(sample_stockx_df: pd.DataFrame) -> None:
    result = clean_price_columns(sample_stockx_df)
    assert "Sneaker Name" in result.columns
    assert len(result) == len(sample_stockx_df)


def test_parse_dates(sample_stockx_df: pd.DataFrame) -> None:
    cleaned = clean_price_columns(sample_stockx_df)
    result = parse_dates(cleaned)
    assert pd.api.types.is_datetime64_any_dtype(result["Order Date"])
    assert pd.api.types.is_datetime64_any_dtype(result["Release Date"])
