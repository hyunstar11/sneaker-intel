"""Tests for the health endpoint."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from sneaker_intel.api.dependencies import get_state


@pytest.fixture
def client() -> TestClient:
    """Create a test client with mocked lifespan."""
    from contextlib import asynccontextmanager

    from fastapi import FastAPI

    from sneaker_intel.api.routes import health

    @asynccontextmanager
    async def noop_lifespan(app: FastAPI):
        yield

    app = FastAPI(lifespan=noop_lifespan)
    app.include_router(health.router)
    return TestClient(app)


@pytest.fixture(autouse=True)
def clear_api_state() -> None:
    state = get_state()
    state.clear()
    yield
    state.clear()


def test_health_returns_200(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data


def test_health_returns_degraded_when_model_missing(client: TestClient) -> None:
    state = get_state()
    state["model_loaded"] = False
    state["market_data_error"] = None

    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "degraded"
    assert "details" in data
