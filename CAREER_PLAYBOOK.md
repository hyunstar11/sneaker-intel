# Sneaker Demand Intelligence: Codebase + Career Playbook

## Who this document is for
This guide is for an entry-level data scientist currently in manufacturing (for example, furniture) who wants to:
1. Understand exactly how this codebase works.
2. Improve it in a way that looks credible to hiring teams at Nike, Adidas, StockX, and similar companies.
3. Turn project work into an interview-ready story.

## 1) What this project is, in one sentence
This repo is a modular ML product that predicts sneaker resale prices and exposes market analytics through a FastAPI backend, Streamlit dashboard, and reusable Python package.

## 2) High-level architecture

```text
Raw CSV data
  -> data loader + cleaning
  -> feature pipeline
  -> model training/evaluation
  -> API endpoints + dashboard pages
  -> user-facing predictions and market intelligence
```

### Primary interfaces
- Python package (`src/sneaker_intel/`) for reusable logic.
- REST API (`/health`, `/api/v1/predict`, analytics routes).
- Streamlit dashboard (4 pages).
- Tests (`tests/`) for key modules.

## 3) Codebase map (what each area does)

### `src/sneaker_intel/config.py`
- Centralized configuration with Pydantic settings.
- Defines constants like data paths, target column, model hyperparameters, feature keyword lists.
- This is where environment overrides can be introduced for deployment.

### `src/sneaker_intel/data/`
- `loader.py`: loads StockX and market datasets with required-column validation.
- `transformers.py`: cleans price columns, parses dates.
- Purpose: avoid schema drift and repeated notebook-style cleaning code.

### `src/sneaker_intel/features/`
- `base.py`: abstract extractor interface.
- `color.py`, `sneaker_type.py`, `region.py`, `temporal.py`, `size.py`: feature extractors.
- `pipeline.py`: chains extractors in deterministic order.
- `market.py`: derived market-dynamics features.
- `stockx.py`: helper logic for `Number of Sales` feature using training lookup (shared between train + inference).

### `src/sneaker_intel/models/`
- Unified wrappers around Linear Regression, Random Forest, XGBoost, LightGBM, and Ensemble.
- `registry.py`: per-model lazy loading (so unavailable optional libs do not break unrelated models).
- `ensemble.py`: simple average/weighted averaging of base model predictions.

### `src/sneaker_intel/evaluation/`
- Standard metrics container + evaluator.
- Model comparison and feature-importance helper utilities.

### `src/sneaker_intel/analysis/`
- `demand_forecast.py`: demand-tier clustering and driver analysis.
- `market_dynamics.py`: liquidity, volatility, inefficiency analysis.
- `release_strategy.py`: month/price-bin/brand strategy summaries.

### `src/sneaker_intel/api/`
- FastAPI app factory.
- Lifespan startup loads data, trains model(s), builds shared runtime state.
- Predict route builds single-row input and applies same feature logic as training.
- Analytics routes return market-level summaries.
- Health route reports healthy/degraded status.

### `src/sneaker_intel/dashboard/`
- Landing page + 4 pages:
1. Market Overview
2. Price Predictor
3. Demand Insights
4. Release Strategy

### `tests/`
- Unit tests across data loading, feature extraction, pipeline behavior, API responses, model registry, ensemble behavior, evaluation metrics, and market inefficiency edge case.

## 4) End-to-end runtime flow

### Prediction path (API)
1. Startup loads StockX CSV.
2. Prices cleaned + dates parsed.
3. Feature pipeline runs.
4. `Number of Sales` reference is computed from training date frequencies.
5. Ensemble is trained and held in process memory.
6. Request comes in with input fields.
7. Single-row dataframe is built and transformed by the same pipeline.
8. `Number of Sales` is computed using training lookup + fallback default.
9. Feature alignment is enforced.
10. Ensemble predicts and response is returned.

### Analytics path
1. Startup loads 2023 market dataset.
2. Demand tiers are assigned.
3. Analyzer object is created.
4. Analytics endpoints and dashboard pages query analyzer methods.

## 5) What was recently patched and why it matters

### A) Train/serve feature consistency (critical)
- `size_freq` now reuses learned training frequencies rather than recomputing from one-row inference inputs.
- `Number of Sales` now uses a shared training-derived lookup in both API and dashboard inference.
- Why this matters: this removes train/serve skew and stabilizes predictions.

### B) API validation quality
- `order_date` and `release_date` are now typed as `date` in schema and validated (`order_date >= release_date`).
- Why this matters: invalid requests fail early with 422 instead of possible internal 500s.

### C) Health and startup transparency
- Market-data startup issues are logged and surfaced as degraded health details.
- Why this matters: easier operations/debugging and truthful service status.

### D) Optional model robustness
- Registry now uses per-model lazy imports.
- API startup now falls back gracefully when optional OpenMP-based libraries are unavailable.
- Why this matters: app remains usable on constrained dev environments.

### E) Reproducibility hardening
- Docker now copies `uv.lock` and runs `uv sync --frozen`.
- `uv.lock` is no longer ignored by `.gitignore`.
- Why this matters: deterministic builds for reviewers and deployment.

## 6) How this maps to a furniture manufacturing DS background
You already have highly transferable business problems:

### Comparable business questions
- Sneaker project question: “What resale price should we expect?”
- Furniture analog: “What sell-through price should we expect by SKU/channel/region?”

- Sneaker project question: “Which products are high demand?”
- Furniture analog: “Which collections/SKUs drive turnover and margin?”

- Sneaker project question: “How do timing and price impact outcomes?”
- Furniture analog: “How do launch windows and MSRP bands affect markdown risk and inventory aging?”

### Transferable technical skills
- Feature engineering on messy categorical text data.
- Model packaging and API serving.
- Dashboarding and stakeholder-facing analytics.
- Basic MLOps thinking (tests, linting, containerized runtime).

## 7) How to position this for Nike/footwear roles
Use this as evidence of product thinking, not just modeling.

### Your narrative should be
“I can convert notebook experiments into a deployable analytics product that supports merchandising and pricing decisions.”

### Nike-relevant framing
- Demand intelligence for drop strategy.
- Price premium and market liquidity monitoring.
- Brand/product-level portfolio analytics.
- API-ready output for downstream apps.

## 8) Highest-impact upgrades to become interview-strong

### Tier 1 (must-do)
1. Add model persistence
- Save fitted artifacts (`joblib`) and load on startup.
- Remove retraining-at-startup dependency.

2. Add proper train/validation/test protocol
- Use time-based split for transaction data.
- Report temporal generalization metrics.

3. Add CI pipeline
- Lint + tests + type checks on push.
- Optionally smoke-test API startup.

4. Add data contracts
- Enforce schema and range checks at ingest.
- Emit explicit errors for malformed rows.

### Tier 2 (strong differentiators)
1. Explainability layer
- SHAP/global importances per prediction cohort.
- Expose explainability endpoint.

2. Drift monitoring
- Track feature drift and prediction drift over time.
- Surface alerts in dashboard.

3. Offline experimentation
- Hyperparameter tuning framework (Optuna).
- Experiment tracking (MLflow or lightweight metadata store).

4. Better uncertainty
- Prediction intervals, not only point estimates.

### Tier 3 (portfolio polish)
1. Add business KPI simulation
- Margin impact, markdown risk, stockout risk scenarios.

2. Add role-based dashboard views
- Merchandising vs planning vs pricing personas.

3. Add architecture diagram + decision log
- Why each tradeoff was made and what comes next.

## 9) Concrete “manufacturing-to-footwear” extension ideas

### Extension A: Inventory Aging Risk Model
- Predict probability a SKU enters aged-inventory bucket.
- Inputs: lead time, seasonality, price tier, historical sell-through.
- Output: recommended markdown or allocation action.

### Extension B: Regional Demand Allocation
- Predict regional demand intensity by product family.
- Optimize allocation to reduce markdown and transfer costs.

### Extension C: Launch Calendar Optimizer
- Evaluate launch month/day patterns against premium and velocity.
- Suggest windows by product archetype.

### Extension D: Bid-Ask Style Proxy for DTC + Wholesale
- Create synthetic liquidity metrics using quote/discount spread analogs.
- Bring market microstructure ideas into retail planning context.

## 10) Suggested roadmap (8 weeks)

### Week 1-2
- Add artifact persistence and startup load path.
- Add time-based split and retrain evaluation report.
- Update README with model lifecycle section.

### Week 3-4
- Add CI workflow.
- Add schema validation and data-quality tests.
- Add regression tests for inference feature alignment.

### Week 5-6
- Add explainability module + endpoint.
- Add dashboard page for model diagnostics and drift.

### Week 7-8
- Add one manufacturing-to-footwear extension (A or B above).
- Write one polished case study with business recommendation and quantified impact.

## 11) Interview packaging: what to show

### In GitHub README
- Architecture diagram.
- “Why this matters for merchandising/pricing teams.”
- Real screenshot of dashboard and sample API payload/response.
- Explicit limitations + next steps.

### In your resume bullets
1. “Refactored 1.9MB notebook into modular ML platform (API + dashboard + tests + Docker).”
2. “Implemented train/serve feature consistency safeguards, reducing prediction skew risk.”
3. “Built market demand segmentation and release-strategy analytics for product planning use cases.”
4. “Containerized deterministic environment with lockfile-based reproducible builds.”

### In interview discussion
- Start with business problem, not algorithm.
- Explain one production bug you found (train/serve skew) and how you fixed it.
- Explain one tradeoff (startup training vs persisted model).
- Explain what you would productionize next and why.

## 12) Gaps you should be ready to discuss honestly
- Current service is not yet using persisted model artifacts by default.
- Optional GPU/OpenMP model dependencies can be environment-sensitive.
- No full monitoring pipeline yet.
- Limited integration/e2e tests vs broad unit coverage.

Being explicit about these gaps while presenting a clear mitigation roadmap is a positive signal.

## 13) “If I were hiring you” checklist
A hiring panel will be reassured if they see:
- Clear problem framing.
- Reproducible environment.
- Reliable API behavior and tests.
- Evidence of handling edge cases.
- Ability to connect modeling to operational decisions.

This project is now close enough to demonstrate all five if you execute the Tier 1 roadmap cleanly.

## 14) Next technical steps in this repo
1. Add persisted artifact path (`models/`) + startup loading fallback.
2. Add time-split evaluation script and publish benchmark table in README.
3. Add one CI workflow file for lint + tests.
4. Add one new domain-transfer page in Streamlit for manufacturing-style allocation or aging risk.

## 15) Final positioning statement
Use this project as proof that you can:
- Build a complete DS product, not only train a model.
- Diagnose and fix reliability issues in ML systems.
- Translate domain experience from manufacturing into footwear-relevant decision tooling.

That combination is exactly what helps entry-level candidates stand out for applied data science roles in athletic footwear companies.
