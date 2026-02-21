"""Train models and save artifacts to disk.

Usage:
    uv run python -m sneaker_intel.training          # train and save
    uv run python -m sneaker_intel.training --help    # show options
"""

from __future__ import annotations

import argparse
import logging
import time
from importlib import import_module
from pathlib import Path

import joblib
import pandas as pd

from sneaker_intel.config import settings
from sneaker_intel.data import DatasetType, load_dataset
from sneaker_intel.data.transformers import clean_price_columns, parse_dates
from sneaker_intel.features.pipeline import FeaturePipeline
from sneaker_intel.features.stockx import (
    add_number_of_sales_feature,
    build_number_of_sales_reference,
)
from sneaker_intel.models.random_forest import RandomForestModel

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"

logger = logging.getLogger(__name__)


def _build_base_models() -> list:
    """Build ensemble members, skipping unavailable optional models."""
    models = [RandomForestModel()]

    optional = [
        ("sneaker_intel.models.xgboost_model", "XGBoostModel"),
        ("sneaker_intel.models.lightgbm_model", "LightGBMModel"),
    ]
    for module_path, class_name in optional:
        try:
            module = import_module(module_path)
            model_cls = getattr(module, class_name)
            models.append(model_cls())
        except Exception as exc:
            logger.warning("Skipping optional model %s: %s", class_name, exc)

    return models


def train_and_save(output_dir: Path | None = None) -> Path:
    """Run the full training pipeline and save artifacts.

    Returns the path to the saved artifact file.
    """
    output_dir = output_dir or MODELS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    logger.info("Loading StockX dataset...")
    stockx_df = load_dataset(DatasetType.STOCKX)
    stockx_df = clean_price_columns(stockx_df)
    stockx_df = parse_dates(stockx_df)

    logger.info("Running feature pipeline...")
    pipeline = FeaturePipeline.stockx_default()
    stockx_df = pipeline.transform(stockx_df)

    # Build sales lookup from training data
    sales_lookup, default_sales = build_number_of_sales_reference(stockx_df)
    stockx_df = add_number_of_sales_feature(stockx_df, sales_lookup, default_sales)

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

    features_df = stockx_df.drop(columns=existing_drop, errors="ignore")
    feature_cols = [c for c in features_df.columns if c != target]
    X = features_df[feature_cols]
    y = features_df[target]

    # Train ensemble
    from sneaker_intel.models.ensemble import EnsembleModel

    base_models = _build_base_models()
    logger.info(
        "Training ensemble with %d models: %s", len(base_models), [m.name for m in base_models]
    )

    start = time.time()
    ensemble = EnsembleModel(models=base_models)
    ensemble.fit(X, y)
    elapsed = time.time() - start
    logger.info("Training completed in %.1fs", elapsed)

    # Evaluate on full data (for quick sanity check)
    preds = ensemble.predict(X)
    mae = float((y - pd.Series(preds, index=y.index)).abs().mean())
    logger.info("Training MAE (full data): $%.2f", mae)

    # Save artifact bundle
    artifact = {
        "ensemble": ensemble,
        "pipeline": pipeline,
        "feature_cols": feature_cols,
        "sales_lookup": sales_lookup,
        "default_sales": default_sales,
        "model_components": [m.name for m in base_models],
    }

    artifact_path = output_dir / "ensemble_v1.joblib"
    joblib.dump(artifact, artifact_path)
    logger.info(
        "Artifact saved to %s (%.1f MB)", artifact_path, artifact_path.stat().st_size / 1e6
    )

    return artifact_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train footwear demand forecasting model")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=f"Directory to save model artifacts (default: {MODELS_DIR})",
    )
    args = parser.parse_args()
    train_and_save(output_dir=args.output_dir)


if __name__ == "__main__":
    main()
