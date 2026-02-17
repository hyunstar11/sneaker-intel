"""Dataset loading with validation."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

import pandas as pd

from sneaker_intel.config import DATA_RAW_DIR, MARKET_2023_FILENAME, STOCKX_FILENAME

STOCKX_REQUIRED_COLUMNS = {
    "Order Date",
    "Brand",
    "Sneaker Name",
    "Sale Price",
    "Retail Price",
    "Release Date",
    "Shoe Size",
    "Buyer Region",
}

MARKET_2023_REQUIRED_COLUMNS = {
    "item",
    "brand",
    "retail",
    "lowestAsk",
    "highestBid",
    "numberOfBids",
    "numberOfAsks",
    "deadstockSold",
    "pricePremium",
}


class DatasetType(StrEnum):
    STOCKX = "stockx"
    MARKET_2023 = "market_2023"


def _validate_columns(df: pd.DataFrame, required: set[str], dataset_name: str) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{dataset_name} dataset missing columns: {missing}")


def load_dataset(
    dataset_type: DatasetType,
    path: str | Path | None = None,
) -> pd.DataFrame:
    """Load and validate a dataset.

    Args:
        dataset_type: Which dataset to load.
        path: Override path. Defaults to data/raw/ location.

    Returns:
        Raw DataFrame with validated columns.
    """
    if dataset_type == DatasetType.STOCKX:
        path = path or DATA_RAW_DIR / STOCKX_FILENAME
        df = pd.read_csv(path, encoding="utf-8-sig")
        _validate_columns(df, STOCKX_REQUIRED_COLUMNS, "StockX")
    elif dataset_type == DatasetType.MARKET_2023:
        path = path or DATA_RAW_DIR / MARKET_2023_FILENAME
        df = pd.read_csv(path, index_col=0)
        _validate_columns(df, MARKET_2023_REQUIRED_COLUMNS, "Market 2023")
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    return df
