"""Shared test fixtures."""

from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def sample_stockx_df() -> pd.DataFrame:
    """Minimal StockX-like DataFrame for testing."""
    return pd.DataFrame(
        {
            "Order Date": ["9/1/17", "10/15/17", "11/20/18", "1/5/19", "2/10/19"],
            "Brand": [" Yeezy", " Yeezy", "Nike", "Nike", " Yeezy"],
            "Sneaker Name": [
                "Adidas-Yeezy-Boost-350-V2-Beluga",
                "Adidas-Yeezy-Boost-350-V2-Blue-Tint",
                "Air-Jordan-1-Retro-High-Off-White-Black",
                "Nike-Air-Presto-Off-White-White-2018",
                "adidas-Yeezy-Boost-350-V2-Static",
            ],
            "Sale Price": ["$1,097", "$685", "$690", "$350", "$280"],
            "Retail Price": ["$220", "$220", "$190", "$120", "$220"],
            "Release Date": ["9/24/16", "12/16/17", "7/14/18", "8/3/18", "12/19/18"],
            "Shoe Size": [11.0, 10.0, 9.5, 8.0, 12.0],
            "Buyer Region": ["California", "New York", "Oregon", "Texas", "Illinois"],
        }
    )


@pytest.fixture
def sample_market_df() -> pd.DataFrame:
    """Minimal sneakers2023-like DataFrame for testing."""
    return pd.DataFrame(
        {
            "item": [
                "Jordan 4 Retro SB Pine Green",
                "Nike Dunk Low Panda",
                "Yeezy Boost 350 V2 Onyx",
                "New Balance 550 White Green",
                "Jordan 1 Low Fragment x Travis Scott",
            ],
            "brand": ["Jordan", "Nike", "Yeezy", "New Balance", "Jordan"],
            "retail": [225, 110, 230, 120, 150],
            "release": ["2023-03-21", "2023-01-10", "2023-05-15", "2023-02-01", "2023-07-20"],
            "lowestAsk": [325, 98, 200, 105, 1200],
            "numberOfAsks": [1995, 5000, 3000, 2000, 800],
            "salesThisPeriod": [2675, 15000, 5000, 3000, 500],
            "highestBid": [480, 90, 185, 95, 1100],
            "numberOfBids": [3697, 8000, 4000, 2500, 1200],
            "annualHigh": [952, 150, 300, 140, 2000],
            "annualLow": [280, 85, 180, 90, 900],
            "volatility": [0.061, 0.045, 0.055, 0.030, 0.080],
            "deadstockSold": [5408, 20000, 8000, 4000, 600],
            "pricePremium": [0.542, -0.1, -0.13, -0.125, 7.0],
            "averageDeadstockPrice": [388, 100, 210, 108, 1400],
            "lastSale": [347, 95, 195, 100, 1150],
            "changePercentage": [0.0, -0.05, 0.02, -0.01, 0.03],
        }
    )


@pytest.fixture
def cleaned_stockx_df(sample_stockx_df: pd.DataFrame) -> pd.DataFrame:
    """StockX DataFrame with prices cleaned and dates parsed."""
    from sneaker_intel.data.transformers import clean_price_columns, parse_dates

    df = clean_price_columns(sample_stockx_df)
    df = parse_dates(df)
    return df
