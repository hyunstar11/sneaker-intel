"""Data transformation utilities for cleaning and parsing."""

from __future__ import annotations

import pandas as pd


def clean_price_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove $ and , from price columns, convert to float.

    Handles both 'Sale Price' and 'Retail Price' columns.
    """
    df = df.copy()
    for col in ("Sale Price", "Retail Price"):
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace("$", "", regex=False)
                .str.replace(",", "", regex=False)
                .astype(float)
            )
    return df


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Parse Order Date and Release Date to datetime."""
    df = df.copy()
    for col in ("Order Date", "Release Date"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format="mixed")
    return df
