"""FastAPI dependencies — model loading and pipeline setup."""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from importlib import import_module
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI

from sneaker_intel.analysis.demand_forecast import DemandSegmenter
from sneaker_intel.analysis.market_dynamics import MarketDynamicsAnalyzer
from sneaker_intel.config import settings
from sneaker_intel.data import DatasetType, load_dataset
from sneaker_intel.data.transformers import clean_price_columns, parse_dates
from sneaker_intel.features.pipeline import FeaturePipeline
from sneaker_intel.features.stockx import (
    add_number_of_sales_feature,
    build_number_of_sales_reference,
)

# Global state for loaded models/data
_state: dict = {}
_logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "models"
ARTIFACT_PATH = MODELS_DIR / "ensemble_v1.joblib"


def _build_base_models() -> list:
    """Build the default ensemble members, skipping unavailable optional models."""
    from sneaker_intel.models.random_forest import RandomForestModel

    models = [RandomForestModel()]

    optional_models = [
        ("sneaker_intel.models.xgboost_model", "XGBoostModel"),
        ("sneaker_intel.models.lightgbm_model", "LightGBMModel"),
    ]
    for module_path, class_name in optional_models:
        try:
            module = import_module(module_path)
            model_cls = getattr(module, class_name)
            models.append(model_cls())
        except Exception as exc:
            _logger.warning("Skipping optional model %s: %s", class_name, exc)

    return models


def _load_from_artifact() -> bool:
    """Try to load pre-trained model from disk. Returns True on success."""
    if not ARTIFACT_PATH.exists():
        return False

    try:
        artifact = joblib.load(ARTIFACT_PATH)
        _state["ensemble"] = artifact["ensemble"]
        _state["pipeline"] = artifact["pipeline"]
        _state["feature_cols"] = artifact["feature_cols"]
        _state["order_date_sales_lookup"] = artifact["sales_lookup"]
        _state["default_number_of_sales"] = artifact["default_sales"]
        _state["model_loaded"] = True
        _state["model_components"] = artifact.get("model_components", [])
        _state["loaded_from"] = "artifact"
        _logger.info("Loaded model from artifact: %s", ARTIFACT_PATH)
        return True
    except Exception as exc:
        _logger.warning("Failed to load artifact %s: %s", ARTIFACT_PATH, exc)
        return False


def _train_at_startup() -> None:
    """Fallback: train the model at startup if no artifact exists."""
    stockx_df = load_dataset(DatasetType.STOCKX)
    stockx_df = clean_price_columns(stockx_df)
    stockx_df = parse_dates(stockx_df)

    # Feature pipeline
    pipeline = FeaturePipeline.stockx_default()
    stockx_df = pipeline.transform(stockx_df)

    # Prepare training data
    target = settings.features.target_column
    drop_cols = [
        "Brand",
        "Buyer Region",
        "Shoe Size",
        "Sneaker Name",
        "Order Date",
        "Release Date",
    ]
    existing_drop = [c for c in drop_cols if c in stockx_df.columns]

    sales_lookup, default_sales = build_number_of_sales_reference(stockx_df)
    stockx_df = add_number_of_sales_feature(
        stockx_df,
        sales_lookup=sales_lookup,
        default_sales=default_sales,
    )

    features_df = stockx_df.drop(columns=existing_drop, errors="ignore")
    feature_cols = [c for c in features_df.columns if c != target]
    X = features_df[feature_cols]
    y = features_df[target]

    # Train ensemble
    from sneaker_intel.models.ensemble import EnsembleModel

    base_models = _build_base_models()
    ensemble = EnsembleModel(models=base_models)
    ensemble.fit(X, y)

    _state["ensemble"] = ensemble
    _state["pipeline"] = pipeline
    _state["feature_cols"] = feature_cols
    _state["stockx_df"] = stockx_df
    _state["order_date_sales_lookup"] = sales_lookup
    _state["default_number_of_sales"] = default_sales
    _state["model_loaded"] = True
    _state["model_components"] = [model.name for model in base_models]
    _state["loaded_from"] = "training"
    _logger.info("Trained model at startup with %d base models", len(base_models))


def get_state() -> dict:
    return _state


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Load models and data at startup."""
    _state.clear()
    _state["model_loaded"] = False
    _state["market_data_loaded"] = False
    _state["market_data_error"] = None

    # Load model: try artifact first, fall back to training
    try:
        if not _load_from_artifact():
            _logger.info("No artifact found at %s, training at startup...", ARTIFACT_PATH)
            _train_at_startup()
    except Exception as exc:
        _logger.exception("Failed to load StockX data or train model")
        _state["stockx_error"] = str(exc)

    # Load market data
    try:
        market_df = load_dataset(DatasetType.MARKET_2023)
        segmenter = DemandSegmenter()
        market_df = segmenter.fit_predict(market_df)
        _state["market_df"] = market_df
        _state["market_analyzer"] = MarketDynamicsAnalyzer(market_df)
        _state["market_data_loaded"] = True
    except (FileNotFoundError, pd.errors.ParserError, ValueError) as exc:
        _logger.warning("Market data unavailable: %s", exc)
        _state["market_data_error"] = str(exc)
        _state["market_df"] = None
        _state["market_analyzer"] = None
    except Exception as exc:
        _logger.exception("Unexpected error loading market data")
        _state["market_data_error"] = f"Unexpected market data error: {exc}"
        _state["market_df"] = None
        _state["market_analyzer"] = None

    yield

    _state.clear()
