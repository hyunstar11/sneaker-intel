FROM python:3.11-slim AS base

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy project files
COPY pyproject.toml uv.lock .python-version ./
COPY src/ src/
COPY data/ data/
COPY models/ models/

# Install dependencies
RUN uv sync --frozen --no-dev

# API target
FROM base AS api
RUN uv sync --frozen --extra api
EXPOSE 8000
CMD ["uv", "run", "uvicorn", "sneaker_intel.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]

# Dashboard target
FROM base AS dashboard
RUN uv sync --frozen --extra dashboard
EXPOSE 8501
CMD ["uv", "run", "streamlit", "run", "src/sneaker_intel/dashboard/app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
