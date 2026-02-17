"""Tests for the analytics endpoints."""

from __future__ import annotations

from contextlib import asynccontextmanager

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from sneaker_intel.analysis.market_dynamics import MarketDynamicsAnalyzer
from sneaker_intel.api.dependencies import get_state
from sneaker_intel.api.routes import analytics


@pytest.fixture
def client() -> TestClient:
    """Create a test client with analytics routes."""

    @asynccontextmanager
    async def noop_lifespan(app: FastAPI):
        yield

    app = FastAPI(lifespan=noop_lifespan)
    app.include_router(analytics.router)
    return TestClient(app)


@pytest.fixture(autouse=True)
def clear_api_state():
    state = get_state()
    state.clear()
    yield
    state.clear()


@pytest.fixture
def market_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "item": ["Shoe A", "Shoe B"],
            "brand": ["Nike", "Adidas"],
            "retail": [120, 200],
            "lowestAsk": [150, 250],
            "numberOfAsks": [100, 200],
            "salesThisPeriod": [500, 300],
            "highestBid": [140, 230],
            "numberOfBids": [80, 150],
            "annualHigh": [200, 350],
            "annualLow": [110, 180],
            "volatility": [0.05, 0.08],
            "deadstockSold": [400, 250],
            "pricePremium": [0.25, 0.30],
            "averageDeadstockPrice": [160, 270],
            "lastSale": [155, 260],
            "changePercentage": [0.01, -0.02],
            "demand_tier": ["High", "Medium"],
        }
    )


# --- market-overview ---


def test_market_overview_returns_200(client: TestClient, market_df: pd.DataFrame) -> None:
    state = get_state()
    state["market_analyzer"] = MarketDynamicsAnalyzer(market_df)

    response = client.get("/api/v1/analytics/market-overview")
    assert response.status_code == 200
    data = response.json()
    assert data["total_products"] == 2
    assert data["total_brands"] == 2
    assert "avg_premium" in data
    assert "avg_volatility" in data


def test_market_overview_returns_503_when_not_loaded(client: TestClient) -> None:
    response = client.get("/api/v1/analytics/market-overview")
    assert response.status_code == 503


# --- demand-tiers ---


def test_demand_tiers_returns_200(client: TestClient, market_df: pd.DataFrame) -> None:
    state = get_state()
    state["market_df"] = market_df

    response = client.get("/api/v1/analytics/demand-tiers")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    assert data[0]["item"] == "Shoe A"
    assert data[0]["demand_tier"] == "High"
    assert data[0]["sales_this_period"] == 500


def test_demand_tiers_respects_limit(client: TestClient, market_df: pd.DataFrame) -> None:
    state = get_state()
    state["market_df"] = market_df

    response = client.get("/api/v1/analytics/demand-tiers?limit=1")
    assert response.status_code == 200
    assert len(response.json()) == 1


def test_demand_tiers_returns_503_when_not_loaded(client: TestClient) -> None:
    response = client.get("/api/v1/analytics/demand-tiers")
    assert response.status_code == 503
