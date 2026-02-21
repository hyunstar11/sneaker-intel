"""Configuration constants and settings for the Sneaker Intel platform."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

STOCKX_FILENAME = "StockX-Data-Contest-2019-3 2.csv"
MARKET_2023_FILENAME = "sneakers2023.csv"


class FeatureConfig(BaseSettings):
    """Feature engineering configuration."""

    color_keywords: list[str] = [
        "Black",
        "White",
        "Grey",
        "Red",
        "Green",
        "Neon",
        "Orange",
        "Tan/Brown",
        "Pink",
        "Blue",
    ]

    sneaker_type_keywords: dict[str, str] = {
        "yeezy350": "Yeezy-Boost-350",
        "airjordan": "Air-Jordan",
        "airforce": "Air-Force",
        "airmax90": "Air-Max-90",
        "airmax97": "Air-Max-97",
        "presto": "Presto",
        "vapormax": "Vapormax",
        "blazer": "Blazer",
        "zoom": "Zoom",
        "react": "React",
    }

    region_states: list[str] = [
        "California",
        "New York",
        "Oregon",
        "Florida",
        "Texas",
    ]

    demand_tier_high: float = 1.5
    demand_tier_medium: float = 1.2

    target_column: str = "Sale Price"
    train_size: float = 0.80
    random_state: int = 42
    cv_folds: int = 10


class ModelConfig(BaseSettings):
    """Model hyperparameter configuration."""

    # Random Forest
    rf_n_estimators: int = 100
    rf_random_state: int = 42

    # XGBoost
    xgb_max_depth: int = 3
    xgb_learning_rate: float = 0.1
    xgb_n_estimators: int = 100
    xgb_objective: str = "reg:squarederror"

    # LightGBM
    lgbm_boosting_type: str = "gbdt"
    lgbm_num_leaves: int = 31
    lgbm_learning_rate: float = 0.05
    lgbm_n_estimators: int = 100
    lgbm_feature_fraction: float = 0.9
    lgbm_bagging_fraction: float = 0.8
    lgbm_bagging_freq: int = 5


class Settings(BaseSettings):
    """Combined application settings with env var override support."""

    features: FeatureConfig = Field(default_factory=FeatureConfig)
    models: ModelConfig = Field(default_factory=ModelConfig)

    data_raw_dir: Path = DATA_RAW_DIR
    data_processed_dir: Path = DATA_PROCESSED_DIR

    model_config = {"env_prefix": "SNEAKER_INTEL_"}


# Singleton settings instance
settings = Settings()
