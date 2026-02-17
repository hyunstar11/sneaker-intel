"""Health check endpoint."""

from __future__ import annotations

from fastapi import APIRouter

from sneaker_intel import __version__
from sneaker_intel.api.dependencies import get_state
from sneaker_intel.api.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    state = get_state()
    if not state:
        return HealthResponse(status="healthy", version=__version__)

    if not state.get("model_loaded", False):
        stockx_error = state.get("stockx_error")
        detail_msg = "Prediction model is not loaded."
        if stockx_error:
            detail_msg += f" StockX data error: {stockx_error}"
        return HealthResponse(
            status="degraded",
            version=__version__,
            details=detail_msg,
        )

    market_error = state.get("market_data_error")
    if market_error:
        return HealthResponse(
            status="degraded",
            version=__version__,
            details=f"Market analytics unavailable: {market_error}",
        )

    return HealthResponse(status="healthy", version=__version__)
