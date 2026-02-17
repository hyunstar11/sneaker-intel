"""Analytics endpoints for market insights."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from sneaker_intel.api.dependencies import get_state
from sneaker_intel.api.schemas import DemandTierResponse, MarketOverviewResponse

router = APIRouter(prefix="/api/v1/analytics")


@router.get("/market-overview", response_model=MarketOverviewResponse)
async def market_overview() -> MarketOverviewResponse:
    state = get_state()
    analyzer = state.get("market_analyzer")
    if analyzer is None:
        raise HTTPException(status_code=503, detail="Market data not loaded")

    overview = analyzer.overview()
    return MarketOverviewResponse(**overview)


@router.get("/demand-tiers", response_model=list[DemandTierResponse])
async def demand_tiers(limit: int = 50) -> list[DemandTierResponse]:
    state = get_state()
    market_df = state.get("market_df")
    if market_df is None:
        raise HTTPException(status_code=503, detail="Market data not loaded")

    rows = market_df.head(limit)
    return [
        DemandTierResponse(
            item=row["item"],
            brand=row["brand"],
            demand_tier=row.get("demand_tier", "Unknown"),
            sales_this_period=int(row.get("salesThisPeriod", 0)),
            price_premium=float(row.get("pricePremium", 0)),
        )
        for _, row in rows.iterrows()
    ]
