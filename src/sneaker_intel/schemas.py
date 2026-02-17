"""Pydantic data models for validation."""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field


class StockXRecord(BaseModel):
    """A single StockX transaction record."""

    order_date: date
    brand: str
    sneaker_name: str
    sale_price: float = Field(gt=0)
    retail_price: float = Field(gt=0)
    release_date: date
    shoe_size: float = Field(gt=0)
    buyer_region: str


class MarketRecord(BaseModel):
    """A single sneakers2023 product record."""

    item: str
    brand: str
    retail: float
    release: date | None = None
    lowest_ask: float = Field(alias="lowestAsk", default=0)
    number_of_asks: int = Field(alias="numberOfAsks", default=0)
    sales_this_period: int = Field(alias="salesThisPeriod", default=0)
    highest_bid: float = Field(alias="highestBid", default=0)
    number_of_bids: int = Field(alias="numberOfBids", default=0)
    annual_high: float = Field(alias="annualHigh", default=0)
    annual_low: float = Field(alias="annualLow", default=0)
    volatility: float = 0
    deadstock_sold: int = Field(alias="deadstockSold", default=0)
    price_premium: float = Field(alias="pricePremium", default=0)
    average_deadstock_price: float = Field(alias="averageDeadstockPrice", default=0)
    last_sale: float = Field(alias="lastSale", default=0)
    change_percentage: float = Field(alias="changePercentage", default=0)

    model_config = {"populate_by_name": True}
