"""FastAPI application factory."""

from __future__ import annotations

from fastapi import FastAPI

from sneaker_intel.api.dependencies import lifespan
from sneaker_intel.api.routes import analytics, health, predict


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Sneaker Demand Intelligence API",
        description="ML-powered sneaker resale price prediction and market analytics",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.include_router(health.router)
    app.include_router(predict.router)
    app.include_router(analytics.router)

    return app
