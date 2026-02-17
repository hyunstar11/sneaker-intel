"""Price prediction endpoint."""

from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, HTTPException

from sneaker_intel.api.dependencies import get_state
from sneaker_intel.api.schemas import PredictionRequest, PredictionResponse
from sneaker_intel.features.stockx import add_number_of_sales_feature

router = APIRouter(prefix="/api/v1")


@router.post("/predict", response_model=PredictionResponse)
async def predict_price(request: PredictionRequest) -> PredictionResponse:
    state = get_state()
    ensemble = state.get("ensemble")
    pipeline = state.get("pipeline")
    feature_cols = state.get("feature_cols")
    sales_lookup = state.get("order_date_sales_lookup", {})
    default_sales = state.get("default_number_of_sales", 0.0)

    if ensemble is None or pipeline is None or feature_cols is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Build a single-row DataFrame matching the StockX schema
    input_data = pd.DataFrame(
        [
            {
                "Sneaker Name": request.sneaker_name,
                "Retail Price": request.retail_price,
                "Shoe Size": request.shoe_size,
                "Buyer Region": request.buyer_region,
                "Order Date": pd.Timestamp(request.order_date),
                "Release Date": pd.Timestamp(request.release_date),
                "Sale Price": 0,  # placeholder
            }
        ]
    )

    # Transform features
    transformed = pipeline.transform(input_data)
    transformed = add_number_of_sales_feature(
        transformed,
        sales_lookup=sales_lookup,
        default_sales=float(default_sales),
    )

    # Align columns
    for col in feature_cols:
        if col not in transformed.columns:
            transformed[col] = 0

    X_input = transformed[feature_cols]

    prediction = float(ensemble.predict(X_input)[0])

    return PredictionResponse(
        predicted_price=round(prediction, 2),
        model_used=ensemble.name,
        features_used=len(feature_cols),
    )
