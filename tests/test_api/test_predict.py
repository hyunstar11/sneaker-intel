"""Tests for the forecast endpoint."""

from __future__ import annotations

from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from sneaker_intel.api.dependencies import get_state
from sneaker_intel.api.routes import health, predict


@pytest.fixture
def client() -> TestClient:
    """Create a test client with mocked state."""

    @asynccontextmanager
    async def noop_lifespan(app: FastAPI):
        yield

    app = FastAPI(lifespan=noop_lifespan)
    app.include_router(health.router)
    app.include_router(predict.router)
    return TestClient(app)


@pytest.fixture(autouse=True)
def clear_api_state() -> None:
    """Isolate global API state between tests."""
    state = get_state()
    state.clear()
    yield
    state.clear()


class _DummyPipeline:
    def transform(self, df):  # noqa: ANN001
        transformed = df.copy()
        transformed["feature_a"] = 100.0
        return transformed


class _DummyEnsemble:
    name = "Ensemble"

    def predict(self, X):  # noqa: ANN001
        return np.array([float(X["feature_a"].iloc[0] + X["Number of Sales"].iloc[0])])


def test_forecast_returns_503_when_model_not_loaded(client: TestClient) -> None:
    """Forecast should 503 when no model is loaded."""
    response = client.post(
        "/api/v1/forecast",
        json={
            "sneaker_name": "Test-Shoe",
            "retail_price": 220.0,
            "shoe_size": 10.0,
            "buyer_region": "California",
            "order_date": "2019-01-01",
            "release_date": "2018-06-01",
        },
    )
    assert response.status_code == 503


def test_forecast_returns_200_with_loaded_state(client: TestClient) -> None:
    state = get_state()
    state["ensemble"] = _DummyEnsemble()
    state["pipeline"] = _DummyPipeline()
    state["feature_cols"] = ["feature_a", "Number of Sales"]
    state["order_date_sales_lookup"] = {pd.Timestamp("2019-01-01"): 7.0}
    state["default_number_of_sales"] = 3.0

    response = client.post(
        "/api/v1/forecast",
        json={
            "sneaker_name": "Test-Shoe",
            "retail_price": 220.0,
            "shoe_size": 10.0,
            "buyer_region": "California",
            "order_date": "2019-01-01",
            "release_date": "2018-06-01",
        },
    )

    assert response.status_code == 200
    data = response.json()
    # Dummy returns 107.0, retail 220.0 → intensity 107/220 ≈ 0.49 → Low
    assert data["market_signal"] == 107.0
    assert data["demand_intensity"] == 0.49
    assert data["demand_tier"] == "Low"
    assert "Conservative demand signal" in data["recommendation"]
    assert data["model_used"] == "Ensemble"
    assert data["features_used"] == 2


def test_forecast_returns_422_for_invalid_date_order(client: TestClient) -> None:
    response = client.post(
        "/api/v1/forecast",
        json={
            "sneaker_name": "Test-Shoe",
            "retail_price": 220.0,
            "shoe_size": 10.0,
            "buyer_region": "California",
            "order_date": "2018-01-01",
            "release_date": "2018-06-01",
        },
    )
    assert response.status_code == 422
