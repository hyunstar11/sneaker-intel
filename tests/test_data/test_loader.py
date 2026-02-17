"""Tests for data loading."""

from __future__ import annotations

import pandas as pd
import pytest

from sneaker_intel.data.loader import DatasetType, _validate_columns


def test_validate_columns_passes(sample_stockx_df: pd.DataFrame) -> None:
    required = {"Order Date", "Brand", "Sneaker Name", "Sale Price"}
    _validate_columns(sample_stockx_df, required, "test")


def test_validate_columns_fails_on_missing(sample_stockx_df: pd.DataFrame) -> None:
    required = {"Order Date", "NonExistentColumn"}
    with pytest.raises(ValueError, match="missing columns"):
        _validate_columns(sample_stockx_df, required, "test")


def test_dataset_type_enum() -> None:
    assert DatasetType.STOCKX.value == "stockx"
    assert DatasetType.MARKET_2023.value == "market_2023"
