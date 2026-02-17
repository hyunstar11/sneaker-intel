# Sneaker Demand Intelligence Platform — Codebase Summary

## What Changed: From Notebook to Platform

**Before**: A single 1.9MB Jupyter notebook (`Sneaker_ResellPred_Model_Edit.ipynb`) with everything inlined — data loading, feature engineering, model training, evaluation, and plots. All hardcoded paths (Google Colab), no tests, no packaging, no git. Two CSV files sitting loose in the root directory.

**After**: A modular, installable Python package (`sneaker-intel`) with 65 source files organized across 7 modules, 28 tests, 4 notebooks, a REST API, an interactive dashboard, and Docker deployment.

## Why This Restructuring

The original notebook is a common "data science portfolio anti-pattern" — it works, but it doesn't demonstrate software engineering skills. Hiring managers at companies like Nike want to see:

- **Can you write production code?** Not just notebook cells, but modules with clear interfaces, type hints, and tests.
- **Can you build a data product?** Not just train a model, but serve predictions and surface insights through an API and dashboard.
- **Do you follow engineering practices?** Dependency management, linting, configuration, packaging.

The restructuring transforms a school project into something that looks like a real internal tool a data team would build.

---

## Module-by-Module Breakdown

### `src/sneaker_intel/config.py`

All the magic numbers that were scattered across notebook cells (color lists, sneaker type keywords, region states, hyperparameters) are now centralized in Pydantic settings classes. This means:

- You change a hyperparameter in one place, not hunting through 60 cells
- Environment variables can override settings (e.g., `SNEAKER_INTEL_` prefix) for deployment
- The config is self-documenting and type-checked

### `src/sneaker_intel/data/` (loader + transformers)

The notebook had `pd.read_csv("/content/drive/MyDrive/...")` hardcoded to Google Colab. Now:

- `load_dataset(DatasetType.STOCKX)` loads with column validation — if a column is missing, you get a clear error, not a cryptic KeyError 50 cells later
- `clean_price_columns()` handles the `"$1,097"` → `1097.0` conversion that was copy-pasted in multiple notebook cells
- `parse_dates()` handles the mixed date format parsing

### `src/sneaker_intel/features/` (5 extractors + pipeline)

The notebook had one giant cell (cell 12-13) creating ~35 features with loops and inline logic. Now each feature type is its own class inheriting from `BaseFeatureExtractor`:

| Extractor | What it does | Features |
|---|---|---|
| `ColorFeatureExtractor` | Binary flags for 10 colors + "Colorful" composite | 11 |
| `SneakerTypeExtractor` | Binary flags for 10 shoe types (Yeezy 350, Air Jordan, etc.) | 10 |
| `RegionFeatureExtractor` | Binary flags for 5 states + "Other States" bucket | 6 |
| `TemporalFeatureExtractor` | Year/month/day decomposition of both dates + days_since_release | 7 |
| `SizeNormalizer` | Frequency-based shoe size normalization | 1 |

`FeaturePipeline.stockx_default()` chains all 5 together. You call `pipeline.transform(df)` once and get 35 new columns. This is testable, composable, and reusable — the API and dashboard both use the same pipeline.

### `src/sneaker_intel/models/` (4 wrappers + ensemble + registry)

The notebook trained each model with different APIs (sklearn's `.fit()/.predict()` vs XGBoost's `DMatrix` + `xgb.train()`). Now every model has the same interface:

```python
model.fit(X_train, y_train)
predictions = model.predict(X_test)
importances = model.feature_importances
```

The `EnsembleModel` averages predictions from RF + XGBoost + LightGBM (matching notebook cell 44). The `get_model("xgboost")` factory lets you swap models by name.

### `src/sneaker_intel/evaluation/`

The notebook had `mean_absolute_error()`, `mean_squared_error()`, `r2_score()` called separately in cells 20, 25, 31, 37, 41, and 44. Now `evaluate_model()` returns a `ModelMetrics` dataclass with all metrics, and `compare_models()` produces the comparison table in one call.

### `src/sneaker_intel/analysis/` (Phase 2 — new analysis)

This is entirely new work not in the original notebook. It uses the `sneakers2023.csv` dataset to answer Nike-relevant business questions:

- **`MarketDynamicsExtractor`**: Creates 6 derived features (bid-ask spread, demand/supply ratio, sell-through ratio, etc.) that translate raw market data into actionable metrics
- **`DemandSegmenter`**: KMeans clustering into High/Medium/Low demand tiers based on sales volume, bids, and premium
- **`ReleaseStrategyAnalyzer`**: Analyzes which release months, price points, and brands correlate with higher premiums
- **`MarketDynamicsAnalyzer`**: Detects market inefficiencies (where bids exceed asks), ranks volatility drivers

### `src/sneaker_intel/api/` (FastAPI)

REST API with 4 endpoints:

- `GET /health` — service health check
- `POST /api/v1/predict` — accepts sneaker attributes (name, retail price, size, region, dates), returns predicted resale price
- `GET /api/v1/analytics/market-overview` — aggregate market stats
- `GET /api/v1/analytics/demand-tiers` — demand tier for each product

The API loads and trains the ensemble model at startup via FastAPI's lifespan handler.

### `src/sneaker_intel/dashboard/` (Streamlit)

4-page interactive dashboard:

1. **Market Overview** — KPIs, brand comparison tables, volatility rankings
2. **Price Predictor** — fill out a form, get a predicted resale price
3. **Demand Insights** — demand tier scatter plots, tier summaries, driver analysis
4. **Release Strategy** — timing analysis charts, pricing sensitivity, brand comparison

### `tests/` (28 tests)

Coverage across all core modules:

- Data loading validation and price cleaning
- Color and sneaker type feature extraction
- Pipeline chaining (all 35 features present after transform)
- Model registry (all 5 models instantiate correctly)
- Ensemble fit/predict/weighted/feature importances
- Evaluation metrics computation
- API health and predict endpoints

---

## Docker: What It Does and Why

### The Problem Docker Solves

Right now, the project runs on the development Mac because it has Python 3.11, uv, and all the right packages installed. If someone else (a recruiter, a hiring manager, a teammate) clones this repo:

- They might have Python 3.9 or 3.13
- They might not have uv
- They might be on Windows or Linux
- Installing XGBoost/LightGBM can fail on certain systems due to C library dependencies
- They need to run `uv sync`, `uv pip install -e .`, know to run `uv run uvicorn ...` — lots of steps

Docker eliminates all of this. The `Dockerfile` packages the entire application — Python runtime, all dependencies, code, and data — into a self-contained image. Anyone with Docker installed runs one command and gets a working API + dashboard.

### How the Dockerfile Works

```dockerfile
FROM python:3.11-slim AS base    # Start from a minimal Python 3.11 image
# ... install uv, copy code, install deps ...

FROM base AS api                  # API target: adds FastAPI/uvicorn
CMD ["uv", "run", "uvicorn", ...] # Starts the API server on port 8000

FROM base AS dashboard            # Dashboard target: adds Streamlit
CMD ["uv", "run", "streamlit", ...] # Starts the dashboard on port 8501
```

It uses **multi-stage builds** — the `api` and `dashboard` targets share the same `base` layer (Python + core packages) but only install their specific extras. This keeps images smaller.

### What docker-compose.yml Does

```yaml
services:
  api:        # builds the "api" target, maps port 8000
  dashboard:  # builds the "dashboard" target, maps port 8501
```

`docker-compose up --build` starts both services simultaneously. The API serves JSON predictions at `localhost:8000`, the dashboard serves the interactive UI at `localhost:8501`. This mimics how a real data product would be deployed — API for programmatic access, dashboard for human exploration.

### Why Docker Matters for a Portfolio

Having Docker in the project signals:

- "I can take a model from notebook to deployment"
- "I understand how services are packaged and run in production"
- "A reviewer can try my project without installing anything"

It's the difference between "I trained a model" and "I built a product." For a Nike data science role, showing you can deploy an ML service is a significant differentiator.

---

## What Still Needs to Be Done

### Immediate / high-value

- **Run the EDA notebook end-to-end** and verify outputs look right — the notebook was written but not executed
- **Git initial commit** — the repo is initialized but nothing is committed yet
- **Pre-commit hooks** — ruff + pytest on commit to prevent regressions

### Medium-term improvements

- **Model serialization** — save trained models to `models/*.joblib` so the API doesn't retrain on every startup (currently trains from scratch each launch, which takes ~60s)
- **Data validation** — the `schemas.py` Pydantic models exist but aren't used for row-level validation during loading yet
- **More tests** — market dynamics extractor, demand segmenter, release strategy analyzer, and the visualization functions are untested
- **CI/CD** — GitHub Actions workflow for lint + test on push
- **StockX BOM issue** — the CSV has a `\ufeff` BOM character in the first column header; works now but could cause subtle issues

### Nice-to-haves

- **Hyperparameter tuning** — the XGBoost params (depth=3, lr=0.1, 100 rounds) are from the original notebook and likely suboptimal. GridSearchCV or Optuna would improve results
- **Feature selection** — the notebook showed many features have 0.0 importance (most colors, most regions). Dropping them would simplify the model
- **Time-series split** — the current train/test split is random, but this is time-series data. Using temporal splits would give more honest performance estimates
- **MLflow/experiment tracking** — log runs, compare experiments
- **Authentication** on the API
