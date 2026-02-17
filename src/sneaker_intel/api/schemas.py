"""API request and response models."""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field, model_validator


class PredictionRequest(BaseModel):
    """Request body for price prediction."""

    sneaker_name: str = Field(..., description="Full sneaker model name")
    retail_price: float = Field(gt=0, description="Retail price in USD")
    shoe_size: float = Field(gt=0, description="US shoe size")
    buyer_region: str = Field(..., description="US state of buyer")
    order_date: date = Field(..., description="Order date (YYYY-MM-DD)")
    release_date: date = Field(..., description="Release date (YYYY-MM-DD)")

    @model_validator(mode="after")
    def validate_date_order(self) -> PredictionRequest:
        if self.order_date < self.release_date:
            raise ValueError("order_date must be on or after release_date")
        return self


class PredictionResponse(BaseModel):
    """Response body for price prediction."""

    predicted_price: float
    model_used: str
    features_used: int


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "healthy"
    version: str
    details: str | None = None


class MarketOverviewResponse(BaseModel):
    """Market overview statistics."""

    total_products: int
    total_brands: int
    avg_premium: float
    median_premium: float
    avg_volatility: float
    total_deadstock_sold: int


class DemandTierResponse(BaseModel):
    """Demand tier information for a sneaker."""

    item: str
    brand: str
    demand_tier: str
    sales_this_period: int
    price_premium: float
