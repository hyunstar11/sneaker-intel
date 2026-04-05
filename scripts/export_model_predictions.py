"""Export model predictions from the sneaker-intel ensemble to JSON.

Trains the ensemble on StockX 2017-2019 data, then generates demand scores
for every shoe model tracked in the reddit-sentiment MODEL_CATALOG.

Usage:
    uv run python scripts/export_model_predictions.py
    uv run python scripts/export_model_predictions.py --out path/to/output.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure the src/ layout is importable
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from sneaker_intel.config import settings  # noqa: E402
from sneaker_intel.data import DatasetType, load_dataset  # noqa: E402
from sneaker_intel.data.transformers import clean_price_columns, parse_dates  # noqa: E402
from sneaker_intel.features.pipeline import FeaturePipeline  # noqa: E402
from sneaker_intel.features.stockx import (  # noqa: E402
    add_number_of_sales_feature,
    build_number_of_sales_reference,
)
from sneaker_intel.models.ensemble import EnsembleModel  # noqa: E402
from sneaker_intel.models.random_forest import RandomForestModel  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shoe models to score — mirrors reddit-sentiment's MODEL_CATALOG
# ---------------------------------------------------------------------------

MODEL_CATALOG: dict[str, tuple[str, float]] = {
    # Nike
    "Air Jordan 1":    ("Nike",        180.0),
    "Air Jordan 3":    ("Nike",        200.0),
    "Air Jordan 4":    ("Nike",        210.0),
    "Air Jordan 5":    ("Nike",        210.0),
    "Air Jordan 11":   ("Nike",        220.0),
    "Air Jordan 12":   ("Nike",        200.0),
    "Dunk Low":        ("Nike",        110.0),
    "Dunk High":       ("Nike",        110.0),
    "Air Force 1":     ("Nike",        110.0),
    "Air Max 90":      ("Nike",        120.0),
    "Air Max 95":      ("Nike",        160.0),
    "Air Max 97":      ("Nike",        175.0),
    "Air Max 1":       ("Nike",        130.0),
    "Air Max Plus":    ("Nike",        165.0),
    "Pegasus":         ("Nike",        130.0),
    "Vaporfly":        ("Nike",        260.0),
    "Alphafly":        ("Nike",        285.0),
    # Adidas
    "Yeezy 350 V2":    ("Adidas",      230.0),
    "Yeezy 380":       ("Adidas",      230.0),
    "Yeezy 500":       ("Adidas",      200.0),
    "Yeezy 700":       ("Adidas",      300.0),
    "Adidas Samba":    ("Adidas",      100.0),
    "Adidas Gazelle":  ("Adidas",      100.0),
    "Campus 00s":      ("Adidas",      100.0),
    "NMD R1":          ("Adidas",      130.0),
    "Ultraboost":      ("Adidas",      190.0),
    "Stan Smith":      ("Adidas",      100.0),
    "Forum Low":       ("Adidas",      100.0),
    # New Balance
    "New Balance 550": ("New Balance", 110.0),
    "New Balance 990": ("New Balance", 185.0),
    "New Balance 2002R": ("New Balance", 150.0),
    "New Balance 1906R": ("New Balance", 150.0),
    "New Balance 574": ("New Balance",  90.0),
    # Other
    "Hoka Clifton":    ("Hoka",        145.0),
    "Hoka Bondi":      ("Hoka",        165.0),
    "Asics Gel-Kayano": ("Asics",      160.0),
    "Puma Speedcat":   ("Puma",        100.0),
}

# Feature columns the ensemble was trained on — must match training exactly
_RELEASE_DATE = pd.Timestamp("2022-01-01")   # representative release for new models
_ORDER_DATE   = pd.Timestamp(date.today())
_BUYER_REGION = "California"
_SHOE_SIZE    = 10.0


def _build_input_row(model_name: str, retail: float) -> dict:
    """Build a single-row input dict for the feature pipeline."""
    return {
        "Sneaker Name": model_name,
        "Retail Price": retail,
        "Shoe Size": _SHOE_SIZE,
        "Buyer Region": _BUYER_REGION,
        "Order Date": _ORDER_DATE,
        "Release Date": _RELEASE_DATE,
        "Sale Price": 0,
    }


def train_ensemble() -> tuple:
    """Train the ensemble on StockX data. Returns (model, pipeline, feature_cols, lookups)."""
    log.info("Loading StockX training data …")
    df = load_dataset(DatasetType.STOCKX)
    df = clean_price_columns(df)
    df = parse_dates(df)

    pipeline = FeaturePipeline.stockx_default()
    df = pipeline.transform(df)

    sales_lookup, default_sales = build_number_of_sales_reference(df)
    df = add_number_of_sales_feature(
        df, sales_lookup=sales_lookup, default_sales=default_sales
    )

    target = settings.features.target_column
    drop_cols = ["Brand", "Buyer Region", "Shoe Size", "Sneaker Name",
                 "Order Date", "Release Date"]
    feat_df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    feature_cols = [c for c in feat_df.columns if c != target]

    X, y = feat_df[feature_cols], feat_df[target]

    models = [RandomForestModel()]
    for module_path, class_name in [
        ("sneaker_intel.models.xgboost_model", "XGBoostModel"),
        ("sneaker_intel.models.lightgbm_model", "LightGBMModel"),
    ]:
        try:
            from importlib import import_module
            mod = import_module(module_path)
            models.append(getattr(mod, class_name)())
        except Exception as exc:
            log.warning("Skipping %s: %s", class_name, exc)

    ensemble = EnsembleModel(models=models)
    log.info("Training ensemble (%d models) on %d rows …", len(models), len(X))
    ensemble.fit(X, y)
    return ensemble, pipeline, feature_cols, sales_lookup, default_sales


def generate_predictions(
    ensemble, pipeline, feature_cols, sales_lookup, default_sales
) -> list[dict]:
    """Score every model in MODEL_CATALOG."""
    rows = []
    for model_name, (brand, retail) in MODEL_CATALOG.items():
        input_df = pd.DataFrame([_build_input_row(model_name, retail)])
        try:
            transformed = pipeline.transform(input_df)
            transformed = add_number_of_sales_feature(
                transformed, sales_lookup=sales_lookup, default_sales=default_sales
            )
            for col in feature_cols:
                if col not in transformed.columns:
                    transformed[col] = 0
            pred = float(ensemble.predict(transformed[feature_cols])[0])
        except Exception as exc:
            log.warning("Prediction failed for %s: %s", model_name, exc)
            pred = retail  # fallback: no premium

        rows.append({
            "model": model_name,
            "brand": brand,
            "retail_price": retail,
            "predicted_sale": round(pred, 2),
            "demand_multiplier": round(pred / retail, 4),
        })

    # Compute demand_score as global percentile rank
    multipliers = np.array([r["demand_multiplier"] for r in rows])
    pct_ranks = [
        round(float((multipliers < m).sum() / len(multipliers)), 4)
        for m in multipliers
    ]
    thresholds = settings.features
    for row, score in zip(rows, pct_ranks):
        row["demand_score"] = score
        dm = row["demand_multiplier"]
        if dm >= thresholds.demand_tier_high:
            row["demand_tier"] = "High"
        elif dm >= thresholds.demand_tier_medium:
            row["demand_tier"] = "Medium"
        else:
            row["demand_tier"] = "Low"

    return sorted(rows, key=lambda r: r["demand_score"], reverse=True)


def main(out_path: Path) -> None:
    ensemble, pipeline, feature_cols, sales_lookup, default_sales = train_ensemble()
    predictions = generate_predictions(
        ensemble, pipeline, feature_cols, sales_lookup, default_sales
    )

    payload = {
        "generated_at": date.today().isoformat(),
        "model_source": "sneaker-intel ensemble (RF+XGB+LGB) trained on StockX 2017-2019",
        "note": (
            "demand_score = global percentile rank by demand_multiplier "
            "(predicted_sale/retail). Higher = stronger historical demand signal."
        ),
        "models": predictions,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    log.info("Wrote %d model predictions → %s", len(predictions), out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export model predictions to JSON")
    parser.add_argument(
        "--out",
        type=Path,
        default=_ROOT.parent / "reddit-sentiment" / "data" / "processed" / "model_predictions.json",
        help="Output JSON path",
    )
    args = parser.parse_args()
    main(args.out)
