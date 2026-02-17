"""Tests for shared StockX feature helpers."""

from __future__ import annotations

import pandas as pd
import pytest

from sneaker_intel.features.stockx import (
    add_number_of_sales_feature,
    build_number_of_sales_reference,
)


def test_build_number_of_sales_reference() -> None:
    df = pd.DataFrame(
        {
            "Order Date": [
                "2019-01-01",
                "2019-01-01",
                "2019-01-02",
                "2019-01-03",
                "2019-01-03",
                "2019-01-03",
            ]
        }
    )

    lookup, default_sales = build_number_of_sales_reference(df)

    assert lookup[pd.Timestamp("2019-01-01")] == 2.0
    assert lookup[pd.Timestamp("2019-01-02")] == 1.0
    assert lookup[pd.Timestamp("2019-01-03")] == 3.0
    assert default_sales == pytest.approx(2.0)


def test_add_number_of_sales_feature_uses_lookup_and_default() -> None:
    df = pd.DataFrame({"Order Date": ["2019-01-01", "2019-01-10"]})
    lookup = {pd.Timestamp("2019-01-01"): 4.0}

    result = add_number_of_sales_feature(df, sales_lookup=lookup, default_sales=1.5)

    assert result["Number of Sales"].tolist() == pytest.approx([4.0, 1.5])
