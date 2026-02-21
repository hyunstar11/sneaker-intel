"""API request and response models."""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field, model_validator


class LaunchForecastRequest(BaseModel):
    """Request body for launch demand forecasting."""

    sneaker_name: str = Field(..., description="Product name (full model identifier)")
    retail_price: float = Field(gt=0, description="Launch retail price in USD")
    shoe_size: float = Field(gt=0, description="US shoe size")
    buyer_region: str = Field(..., description="Target market (US state)")
    order_date: date = Field(..., description="Forecast date (YYYY-MM-DD)")
    release_date: date = Field(..., description="Launch date (YYYY-MM-DD)")

    @model_validator(mode="after")
    def validate_date_order(self) -> LaunchForecastRequest:
        if self.order_date < self.release_date:
            raise ValueError("order_date must be on or after release_date")
        return self


class LaunchForecastResponse(BaseModel):
    """Response body for launch demand forecasting."""

    market_signal: float
    demand_intensity: float
    demand_tier: str
    recommendation: str
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
