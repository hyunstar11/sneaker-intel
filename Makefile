.PHONY: install lint format test train run-api run-dashboard clean

install:
	uv sync --all-extras

lint:
	uv run ruff check src/ tests/

format:
	uv run ruff format src/ tests/

test:
	uv run pytest tests/ -v

train:
	uv run python -m sneaker_intel.training

run-api:
	uv run uvicorn sneaker_intel.api.app:create_app --factory --reload --port 8000

run-dashboard:
	uv run streamlit run src/sneaker_intel/dashboard/app.py --server.port 8501

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf dist/ build/ *.egg-info/
