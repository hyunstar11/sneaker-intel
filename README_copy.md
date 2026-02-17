# Sneaker Demand Intelligence Platform

An end-to-end ML platform that predicts sneaker resale prices and analyzes market dynamics in the secondary sneaker market. Built as a modular Python package with a REST API, interactive dashboard, and Docker deployment.

For an extensive codebase walkthrough and career strategy guide, see [CAREER_PLAYBOOK.md](CAREER_PLAYBOOK.md).

## Table of Contents

- [What This Project Does](#what-this-project-does)
- [Why This Project Exists](#why-this-project-exists)
- [Architecture Overview](#architecture-overview)
- [Datasets](#datasets)
- [Feature Engineering](#feature-engineering)
- [Models and Performance](#models-and-performance)
- [Analysis Modules](#analysis-modules)
- [API Reference](#api-reference)
- [Dashboard](#dashboard)
- [Expected Outputs](#expected-outputs)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Docker Deployment](#docker-deployment)
- [Testing](#testing)
- [Tech Stack](#tech-stack)
- [Future Work](#future-work)

---

## What This Project Does

This platform answers three core business questions relevant to the sneaker and footwear industry:

1. **What will a sneaker resell for?** — Given a sneaker's name, retail price, shoe size, buyer region, and release/order dates, an ensemble ML model (Random Forest + XGBoost + LightGBM) predicts the secondary-market resale price. Trained on 99K+ real StockX transactions.

2. **Which sneakers are in high demand, and why?** — KMeans clustering segments ~2,000 sneaker products into High / Medium / Low demand tiers based on sales volume, bid activity, and price premium. A Random Forest classifier then identifies which product characteristics (retail price, volatility, bid count) drive demand.

3. **How do release timing and pricing affect resale value?** — Analysis of release month, retail price bracket, and brand reveals which strategies correlate with higher premiums, faster sell-through, and greater market liquidity.

The platform surfaces these insights through three interfaces:
- A **FastAPI REST API** for programmatic access (price prediction + market analytics endpoints)
- A **Streamlit dashboard** with four interactive pages (Market Overview, Price Predictor, Demand Insights, Release Strategy)
- **Jupyter notebooks** for exploratory analysis and reproducible research

---

## Why This Project Exists

This project was originally a single 1.9MB Jupyter notebook built for a university course — a sneaker resale price prediction model running on Google Colab with hardcoded paths, no tests, no packaging, and all logic crammed into one file.

It was refactored into a modular, production-style platform to demonstrate:

- **ML engineering**: Feature pipelines with composable extractors, a model registry with a unified prediction interface, and an evaluation framework that standardizes metrics across models
- **Data product thinking**: Not just "I trained a model" but a deployable service with API endpoints, an interactive dashboard, and Docker packaging
- **Software engineering practices**: Type-checked configuration with Pydantic, abstract base classes for extensibility, 28 unit tests, linting with ruff, dependency management with uv

The restructuring targets roles in data science and ML engineering within the footwear industry (Nike, Adidas, StockX), where the ability to go from notebook prototype to deployable product is a key differentiator.

---

## Architecture Overview

```
                          ┌──────────────────────────────────────────┐
                          │           pyproject.toml                 │
                          │      (sneaker-intel package, uv/hatch)   │
                          └──────────────┬───────────────────────────┘
                                         │
               ┌─────────────────────────┼─────────────────────────┐
               │                         │                         │
        ┌──────▼──────┐          ┌───────▼───────┐         ┌──────▼──────┐
        │   data/     │          │  features/    │         │  models/    │
        │ loader.py   │──load──▶ │  pipeline.py  │──feat──▶│ registry.py │
        │ transform.  │  clean   │  5 extractors │  35     │ 4 wrappers  │
        │   py        │  parse   │  (chained)    │  cols   │ + ensemble  │
        └─────────────┘          └───────────────┘         └──────┬──────┘
                                                                  │
                                                           fit / predict
                                                                  │
               ┌──────────────────────────────────────────────────┤
               │                         │                        │
        ┌──────▼──────┐          ┌───────▼───────┐        ┌──────▼──────┐
        │ evaluation/ │          │  analysis/    │        │   api/      │
        │ metrics.py  │          │ demand_fore.  │        │   app.py    │
        │ comparison  │          │ market_dyn.   │        │   routes/   │
        │ interpret.  │          │ release_str.  │        │ dependencies│
        └─────────────┘          └───────────────┘        └──────┬──────┘
                                                                 │
                                        ┌────────────────────────┤
                                        │                        │
                                 ┌──────▼──────┐          ┌──────▼──────┐
                                 │ dashboard/  │          │   Docker    │
                                 │ Streamlit   │          │ Dockerfile  │
                                 │ 4 pages     │          │ compose.yml │
                                 └─────────────┘          └─────────────┘
```

**Data flow**: Raw CSV → `loader.py` validates columns → `transformers.py` cleans prices and parses dates → `FeaturePipeline` chains 5 extractors to produce 35 feature columns → Models train on features → `evaluation/` computes metrics → API/dashboard serve predictions and analytics.

---

## Datasets

### StockX Transactions (2017–2019)

**File**: `data/raw/StockX-Data-Contest-2019-3 2.csv` (99,956 rows)

Transaction-level records from the StockX resale marketplace. Each row is one completed sale.

| Column | Type | Example | Description |
|---|---|---|---|
| Order Date | date | `9/1/2017` | When the transaction occurred |
| Brand | string | `Yeezy`, `Nike` | Shoe brand |
| Sneaker Name | string | `Adidas-Yeezy-Boost-350-V2-Beluga` | Full model name (used for feature extraction) |
| Sale Price | string | `$1,097` | Actual resale price paid (prediction target) |
| Retail Price | string | `$220` | Original retail price |
| Release Date | date | `9/24/2016` | When the shoe was first released |
| Shoe Size | float | `11.0` | US shoe size |
| Buyer Region | string | `California` | US state of the buyer |

**Note**: Sale Price and Retail Price arrive as formatted strings (`"$1,097"`) and are cleaned to floats by `transformers.clean_price_columns()`. The CSV has a BOM character (`\ufeff`) in the first column header, handled transparently by pandas' UTF-8 reader.

### Market Snapshot (2023)

**File**: `data/raw/sneakers2023.csv` (~2,000 rows)

Product-level market data — one row per sneaker model, capturing live marketplace state.

| Column | Type | Description |
|---|---|---|
| item | string | Sneaker model name |
| brand | string | Brand name |
| retail | float | Retail price |
| release | date | Release date |
| lowestAsk | float | Current lowest asking price on the market |
| highestBid | float | Current highest bid price |
| numberOfAsks / numberOfBids | int | Active ask/bid count (market depth) |
| salesThisPeriod | int | Units sold in the current period |
| deadstockSold | int | Total deadstock (new, unworn) units ever sold |
| volatility | float | Price volatility metric |
| pricePremium | float | Premium over retail as a ratio (e.g., 0.54 = 54% above retail) |
| averageDeadstockPrice | float | Average sale price for deadstock units |
| lastSale | float | Most recent sale price |
| annualHigh / annualLow | float | 52-week price range |
| changePercentage | float | Recent price change percentage |

This dataset powers the market dynamics analysis, demand segmentation, and release strategy modules.

---

## Feature Engineering

Features are produced by a composable pipeline of extractors, each inheriting from `BaseFeatureExtractor` (an ABC requiring `extract(df)` → DataFrame and `feature_names` → list[str]).

`FeaturePipeline.stockx_default()` chains all 5 extractors and is used consistently across training, API prediction, and the dashboard — ensuring the same feature logic everywhere.

### StockX Feature Extractors (35 features)

| Extractor | What It Extracts | # Features | Logic |
|---|---|---|---|
| **ColorFeatureExtractor** | Binary flags for 10 colors + "Colorful" count | 11 | Searches `Sneaker Name` for color keywords (Black, White, Grey, Red, Green, Neon, Orange, Tan/Brown, Pink, Blue). "Colorful" = count of colors present. |
| **SneakerTypeExtractor** | Binary flags for 10 shoe model types | 10 | Matches keywords in `Sneaker Name`: Yeezy-Boost-350, Air-Jordan, Air-Force, Air-Max-90, Air-Max-97, Presto, Vapormax, Blazer, Zoom, React. |
| **RegionFeatureExtractor** | Binary flags for 5 states + "Other States" | 6 | One-hot encodes `Buyer Region` for California, New York, Oregon, Florida, Texas. All other states → "Other States". |
| **TemporalFeatureExtractor** | Date decomposition + days since release | 7 | Extracts year/month/day from both Order Date and Release Date. Computes `days_since_release = Order Date - Release Date` (captures hype decay). |
| **SizeNormalizer** | Frequency-normalized shoe size | 1 | Maps each `Shoe Size` to its normalized frequency in the dataset (common sizes like 10/11 get higher values). |

### Market Dynamics Features (6 features, sneakers2023 dataset)

| Feature | Formula | Business Meaning |
|---|---|---|
| `bid_ask_spread` | lowestAsk - highestBid | Market tightness; narrow spread = liquid market |
| `bid_ask_spread_pct` | spread / lowestAsk | Relative spread, comparable across price levels |
| `demand_supply_ratio` | numberOfBids / numberOfAsks | >1 means more demand than supply |
| `price_range_pct` | (annualHigh - annualLow) / retail | Annual price swing relative to retail |
| `sell_through_ratio` | salesThisPeriod / deadstockSold | Recent sales velocity vs total historical sales |
| `premium_sustainability` | lastSale / averageDeadstockPrice | Whether the most recent sale holds above the average |

---

## Models and Performance

All models implement a unified `PredictionModel` protocol: `.fit(X, y)`, `.predict(X)`, `.feature_importances`, `.get_params()`. This lets the ensemble, evaluation, and API code work identically regardless of which model is underneath.

| Model | Wrapper Class | Library | Key Hyperparameters |
|---|---|---|---|
| Linear Regression | `LinearRegressionModel` | scikit-learn | (defaults) |
| Random Forest | `RandomForestModel` | scikit-learn | 100 estimators, random_state=42 |
| XGBoost | `XGBoostModel` | xgboost | max_depth=3, lr=0.1, 100 rounds |
| LightGBM | `LightGBMModel` | lightgbm | 31 leaves, lr=0.05, 100 rounds, bagging_fraction=0.8 |
| **Ensemble** | `EnsembleModel` | (custom) | Averages predictions from RF + XGBoost + LightGBM |

### Performance on StockX Test Set (80/20 split, random_state=42)

| Model | MAE ($) | MSE | RMSE ($) | R² |
|---|---|---|---|---|
| Linear Regression | 86.93 | 20,126 | ~141.87 | 0.69 |
| Random Forest | 17.31 | 1,610 | ~40.12 | 0.975 |
| XGBoost | 51.95 | 8,508 | ~92.24 | 0.87 |
| LightGBM | 35.55 | 4,309 | ~65.64 | 0.93 |
| **Ensemble** | **31.90** | **3,281** | **~57.28** | **0.95** |

**Interpretation**: The ensemble model predicts sneaker resale prices within ~$32 of the actual transaction price on average, explaining 95% of the variance. Random Forest alone is the strongest individual model (R²=0.975), but the ensemble provides more robust generalization.

### Model Registry

Models are accessed through a factory pattern:

```python
from sneaker_intel.models import get_model

model = get_model("xgboost")  # or "linear", "random_forest", "lightgbm", "ensemble"
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

The registry (`models/registry.py`) uses lazy imports to avoid loading all ML libraries until a specific model is requested.

---

## Analysis Modules

### Demand Segmentation (`analysis/demand_forecast.py`)

**What it does**: Segments sneakers into **High / Medium / Low demand tiers** using KMeans clustering on four features: `salesThisPeriod`, `deadstockSold`, `numberOfBids`, and `pricePremium`.

**How it works**:
1. `DemandSegmenter.fit_predict(df)` scales the features with `StandardScaler`, runs KMeans (k=3), and maps cluster IDs to tier names by sorting clusters on average `salesThisPeriod`.
2. `get_tier_summary(df)` produces aggregate stats (mean, median, count) for each tier.
3. `analyze_demand_drivers(df)` trains a Random Forest classifier on the tier labels and returns feature importances — answering "what makes a sneaker high demand?"

**Expected output**: A DataFrame with a new `demand_tier` column. High-demand sneakers tend to have large bid counts, high sales volume, and moderate-to-high premiums.

### Market Dynamics (`analysis/market_dynamics.py`)

**What it does**: Provides brand-level and product-level market intelligence.

- `liquidity_analysis()` — Average bids, asks, deadstock sold, and sales per brand. Identifies which brands have the most active secondary markets.
- `volatility_drivers(top_n=20)` — The N most volatile sneakers with their retail, premium, and sales data. Useful for identifying speculative products.
- `market_inefficiencies()` — Detects products where `highestBid > lowestAsk` (theoretical arbitrage opportunities), sorted by the arbitrage percentage.
- `overview()` — High-level stats: total products, total brands, average premium, median premium, average volatility, total deadstock sold.

### Release Strategy (`analysis/release_strategy.py`)

**What it does**: Analyzes how release timing and pricing correlate with resale outcomes.

- `timing_analysis()` — Groups products by release month, shows average premium and average sales per month. Reveals seasonal patterns (e.g., holiday releases may carry higher premiums).
- `pricing_sensitivity()` — Bins retail prices into brackets (<$100, $100-150, ..., $500+) and shows how premium and sales vary across price tiers.
- `brand_comparison()` — Compares brands on average premium, average sales, average volatility, and product count.

---

## API Reference

The FastAPI service starts with `make run-api` and trains the ensemble model at startup (~60s on first launch). Auto-generates OpenAPI docs at `http://localhost:8000/docs`.

### Endpoints

#### `GET /health`
Health check. Returns service status and version.

```json
{"status": "healthy", "version": "0.1.0"}
```

#### `POST /api/v1/predict`
Predict resale price for a single sneaker.

**Request body**:
```json
{
  "sneaker_name": "Adidas-Yeezy-Boost-350-V2-Cream-White",
  "retail_price": 220.0,
  "shoe_size": 10.0,
  "buyer_region": "California",
  "order_date": "2019-06-15",
  "release_date": "2018-09-21"
}
```

**Response**:
```json
{
  "predicted_price": 347.52,
  "model_used": "Ensemble",
  "features_used": 37
}
```

The endpoint internally constructs a single-row DataFrame, runs it through the same `FeaturePipeline.stockx_default()` used in training, aligns columns, and calls `ensemble.predict()`.

#### `GET /api/v1/analytics/market-overview`
Returns aggregate market statistics from the sneakers2023 dataset.

```json
{
  "total_products": 2000,
  "total_brands": 45,
  "avg_premium": 0.342,
  "median_premium": 0.125,
  "avg_volatility": 0.0412,
  "total_deadstock_sold": 1250000
}
```

#### `GET /api/v1/analytics/demand-tiers?limit=50`
Returns demand tier classification for each sneaker (up to `limit` rows).

```json
[
  {
    "item": "Jordan 4 Retro SB Pine Green",
    "brand": "Jordan",
    "demand_tier": "High",
    "sales_this_period": 2675,
    "price_premium": 0.542
  }
]
```

---

## Dashboard

The Streamlit dashboard (`make run-dashboard`, port 8501) provides four interactive pages:

### Home
Summary KPIs: models trained (4), StockX transactions (99K+), market products (2K+). Navigation to all pages.

### 1. Market Overview
- KPI cards: total products, brands, average premium, average volatility
- Brand comparison table (liquidity metrics: avg bids, asks, deadstock, sales)
- Most volatile sneakers table (top 15)
- Market dynamics derived features table (bid-ask spread, demand/supply ratio, etc.)

### 2. Price Predictor
- Input form: sneaker name, retail price, shoe size, buyer region, order date, release date
- On submit: trains ensemble model (cached after first load), runs feature pipeline, returns predicted resale price
- Uses `@st.cache_resource` to avoid retraining on every interaction

### 3. Demand Insights
- Bar chart of demand tier distribution (High / Medium / Low)
- Tier summary statistics table (mean/median/count per tier for sales, premium, etc.)
- Demand driver feature importances table
- Scatter plot of price premium vs sales volume, colored by demand tier

### 4. Release Strategy
- Release timing table and bar chart (premium by month)
- Pricing sensitivity table (premium by retail price bracket)
- Brand comparison table (premium, sales, volatility, count by brand)

---

## Expected Outputs

### When you run the API (`make run-api`)
- The server trains the ensemble model on the full StockX dataset at startup (~60s)
- After startup, `GET /health` returns `{"status": "healthy", "version": "0.1.0"}`
- `POST /api/v1/predict` with sneaker attributes returns a dollar-amount prediction
- Analytics endpoints return JSON summaries of the 2023 market data

### When you run the dashboard (`make run-dashboard`)
- A browser opens at `localhost:8501` with the Streamlit app
- The Price Predictor page trains the model on first visit (cached afterwards)
- All market analysis pages load the sneakers2023.csv and compute analytics in real-time

### When you run tests (`make test`)
- 28 tests run across 7 test files, covering data loading, price cleaning, feature extraction, pipeline chaining, model registry, ensemble predictions, evaluation metrics, and API endpoints
- Expected: all 28 passing

### When you run via Docker (`docker-compose up --build`)
- Two containers start: `api` (port 8000) and `dashboard` (port 8501)
- Both share the `data/` directory via a volume mount
- Anyone with Docker installed can run the full platform without installing Python, uv, or any dependencies

---

## Project Structure

```
Portfolio/
├── pyproject.toml                          # Package metadata, dependencies, tool config
├── .python-version                         # Python 3.11
├── .gitignore
├── Makefile                                # install, lint, format, test, run-api, run-dashboard
├── Dockerfile                              # Multi-stage: base → api / dashboard targets
├── docker-compose.yml                      # Runs API + dashboard as two services
├── README.md
├── SUMMARY.md                              # Detailed codebase summary and change log
│
├── data/
│   ├── raw/
│   │   ├── StockX-Data-Contest-2019-3 2.csv    # 99K transactions
│   │   └── sneakers2023.csv                     # ~2K products
│   ├── processed/                               # (gitignored, for cached outputs)
│   └── README.md                                # Data dictionary
│
├── notebooks/
│   ├── 01_eda_stockx.ipynb                 # Exploratory analysis on StockX data
│   ├── 02_eda_market_dynamics.ipynb        # Market dynamics exploration
│   ├── 03_demand_forecasting.ipynb         # Demand tier analysis
│   ├── 04_release_strategy.ipynb           # Release timing/pricing analysis
│   └── archive/
│       └── Sneaker_ResellPred_Model_Edit.ipynb  # Original monolithic notebook
│
├── src/sneaker_intel/
│   ├── __init__.py                         # Package version
│   ├── config.py                           # FeatureConfig, ModelConfig, Settings (pydantic)
│   ├── schemas.py                          # StockXRecord, MarketRecord (pydantic validation)
│   │
│   ├── data/
│   │   ├── loader.py                       # DatasetType enum, load_dataset(), column validation
│   │   └── transformers.py                 # clean_price_columns(), parse_dates()
│   │
│   ├── features/
│   │   ├── base.py                         # BaseFeatureExtractor ABC
│   │   ├── color.py                        # ColorFeatureExtractor (11 features)
│   │   ├── sneaker_type.py                 # SneakerTypeExtractor (10 features)
│   │   ├── region.py                       # RegionFeatureExtractor (6 features)
│   │   ├── temporal.py                     # TemporalFeatureExtractor (7 features)
│   │   ├── size.py                         # SizeNormalizer (1 feature)
│   │   ├── market.py                       # MarketDynamicsExtractor (6 features)
│   │   └── pipeline.py                     # FeaturePipeline — chains extractors
│   │
│   ├── models/
│   │   ├── base.py                         # PredictionModel Protocol
│   │   ├── linear.py                       # LinearRegressionModel
│   │   ├── random_forest.py                # RandomForestModel
│   │   ├── xgboost_model.py                # XGBoostModel
│   │   ├── lightgbm_model.py               # LightGBMModel
│   │   ├── ensemble.py                     # EnsembleModel (weighted/unweighted averaging)
│   │   └── registry.py                     # get_model() factory, available_models()
│   │
│   ├── evaluation/
│   │   ├── metrics.py                      # ModelMetrics dataclass, evaluate_model()
│   │   ├── comparison.py                   # compare_models() → sorted DataFrame
│   │   └── interpretability.py             # compute_ensemble_importances(), get_top_features()
│   │
│   ├── analysis/
│   │   ├── demand_forecast.py              # DemandSegmenter, analyze_demand_drivers()
│   │   ├── market_dynamics.py              # MarketDynamicsAnalyzer (liquidity, volatility, etc.)
│   │   └── release_strategy.py             # ReleaseStrategyAnalyzer (timing, pricing, brands)
│   │
│   ├── visualization/
│   │   ├── style.py                        # Nike-inspired color palette, apply_nike_style()
│   │   └── plots.py                        # 5 reusable plot functions (actual vs predicted, etc.)
│   │
│   ├── api/
│   │   ├── app.py                          # create_app() FastAPI factory
│   │   ├── dependencies.py                 # lifespan handler (loads data, trains model at startup)
│   │   ├── schemas.py                      # Request/response Pydantic models
│   │   └── routes/
│   │       ├── health.py                   # GET /health
│   │       ├── predict.py                  # POST /api/v1/predict
│   │       └── analytics.py               # GET /api/v1/analytics/market-overview, demand-tiers
│   │
│   └── dashboard/
│       ├── app.py                          # Streamlit entry point (home page)
│       └── pages/
│           ├── 01_market_overview.py       # Brand comparison, volatility, market features
│           ├── 02_price_predictor.py       # Interactive prediction form
│           ├── 03_demand_insights.py       # Tier distribution, drivers, scatter plot
│           └── 04_release_strategy.py      # Timing, pricing sensitivity, brand comparison
│
└── tests/
    ├── conftest.py                         # Shared fixtures (sample DataFrames)
    ├── test_data/
    │   ├── test_loader.py                  # Dataset loading and column validation
    │   └── test_transformers.py            # Price cleaning and date parsing
    ├── test_features/
    │   ├── test_color.py                   # Color feature extraction
    │   ├── test_sneaker_type.py            # Sneaker type feature extraction
    │   └── test_pipeline.py                # Full pipeline produces 35 features
    ├── test_models/
    │   ├── test_registry.py                # get_model() and available_models()
    │   └── test_ensemble.py                # Ensemble fit/predict/weights/importances
    ├── test_evaluation/
    │   └── test_metrics.py                 # evaluate_model() and ModelMetrics
    └── test_api/
        ├── test_health.py                  # GET /health returns 200
        └── test_predict.py                 # POST /predict returns valid prediction
```

---

## Quick Start

```bash
# Prerequisites: Python 3.11+, uv (https://docs.astral.sh/uv/)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install
git clone <repo-url> && cd Portfolio
make install          # uv sync --all-extras

# Verify everything works
make test             # 28 tests, all should pass
make lint             # ruff check on src/ and tests/

# Run the API (trains model at startup, then serves on port 8000)
make run-api
# → http://localhost:8000/docs for interactive API docs

# Run the dashboard (separate terminal)
make run-dashboard
# → http://localhost:8501 for the Streamlit app
```

---

## Docker Deployment

Docker packages the entire application — Python runtime, all dependencies, trained models, and data — into self-contained images. Anyone with Docker can run the platform without installing Python, uv, or any ML libraries.

The Dockerfile uses **multi-stage builds**: a shared `base` layer (Python 3.11 + core dependencies), then separate `api` and `dashboard` targets that add only their specific extras. This keeps images lean.

```bash
# Build and start both services
docker-compose up --build

# API:       http://localhost:8000
# Dashboard: http://localhost:8501
```

`docker-compose.yml` mounts `./data` as a volume so both containers read the same CSV files without baking data into the image.

---

## Testing

```bash
make test
```

28 tests across 7 test modules:

| Module | Tests | What's Covered |
|---|---|---|
| `test_data/test_loader.py` | Dataset loading, column validation, error on missing columns |
| `test_data/test_transformers.py` | `"$1,097"` → `1097.0` conversion, date parsing |
| `test_features/test_color.py` | Color flags extracted correctly from sneaker names |
| `test_features/test_sneaker_type.py` | Shoe type binary features match expected keywords |
| `test_features/test_pipeline.py` | Full pipeline produces all 35 feature columns |
| `test_models/test_registry.py` | `get_model()` instantiates all 5 model types |
| `test_models/test_ensemble.py` | Ensemble fit/predict, weighted averaging, feature importances |
| `test_evaluation/test_metrics.py` | `evaluate_model()` returns correct metric types |
| `test_api/test_health.py` | Health endpoint returns 200 with version |
| `test_api/test_predict.py` | Predict endpoint returns valid price prediction |

All tests use synthetic fixtures (defined in `conftest.py`) — they don't require the real 99K-row dataset.

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.11 |
| Data | pandas, NumPy |
| ML | scikit-learn, XGBoost, LightGBM |
| Configuration | Pydantic, pydantic-settings |
| API | FastAPI, Uvicorn |
| Dashboard | Streamlit |
| Visualization | Matplotlib, Seaborn (Nike-themed palette) |
| Testing | pytest, pytest-cov |
| Linting | ruff (E, F, I, UP, B, SIM rules) |
| Packaging | hatchling, uv |
| Deployment | Docker, docker-compose |

---

## Future Work

### High Priority
- **Model serialization** — Save trained models to `.joblib` files so the API doesn't retrain on every startup (~60s cold start currently)
- **Git initial commit and CI/CD** — GitHub Actions for automated lint + test on push
- **Pre-commit hooks** — Enforce ruff + pytest before every commit

### Medium Priority
- **Temporal train/test split** — Current split is random, but this is time-series data. Using chronological splits would give more honest performance estimates
- **Hyperparameter tuning** — XGBoost params are from the original notebook and likely suboptimal. Optuna or GridSearchCV could improve results
- **Feature selection** — Many color and region features have near-zero importance. Dropping them would simplify the model with no accuracy loss
- **More test coverage** — Market dynamics extractor, demand segmenter, release strategy analyzer, and visualization functions are untested

### Nice to Have
- **MLflow experiment tracking** — Log runs, compare experiments, track model versions
- **API authentication** — JWT or API key auth for the prediction endpoint
- **Batch prediction endpoint** — Accept a CSV upload and return predictions for all rows
- **Live data ingestion** — Scheduled pipeline to pull fresh market data (even if simulated)
