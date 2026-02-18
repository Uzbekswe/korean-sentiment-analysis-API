# ==============================================================================
# Makefile — Korean Sentiment Analysis API
# Common operations for development and deployment
# ==============================================================================

.PHONY: install dev serve test lint format docker-build docker-up clean

# --- Setup ---
install:  ## Install production dependencies
	pip install --upgrade pip
	pip install "."

dev:  ## Install dev + production dependencies
	pip install --upgrade pip
	pip install ".[dev]"

# --- Run ---
serve:  ## Start the FastAPI server locally
	uvicorn src.serving.app:app --reload --host 0.0.0.0 --port 8000

streamlit:  ## Start the Streamlit UI
	streamlit run streamlit_app.py

# --- Quality ---
test:  ## Run all tests with pytest
	pytest tests/ -v --tb=short

lint:  ## Run linter
	ruff check src/ tests/

format:  ## Auto-format code
	ruff format src/ tests/

# --- Docker ---
docker-build:  ## Build the Docker image
	docker build -f docker/Dockerfile -t korean-sentiment-api:latest .

docker-up:  ## Start with docker compose
	docker compose -f docker/docker-compose.yml up --build

docker-down:  ## Stop docker compose
	docker compose -f docker/docker-compose.yml down

# --- Cleanup ---
clean:  ## Remove caches and build artifacts
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name *.egg-info -exec rm -rf {} +
	rm -rf dist/ build/ logs/

# --- Help ---
help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'
