# Makefile for ReleAF AI

.PHONY: help setup install clean test format lint docker-build docker-up docker-down train-all

help:
	@echo "ReleAF AI - Available commands:"
	@echo "  make setup          - Initial project setup"
	@echo "  make install        - Install dependencies"
	@echo "  make clean          - Clean generated files"
	@echo "  make test           - Run tests"
	@echo "  make format         - Format code"
	@echo "  make lint           - Lint code"
	@echo "  make docker-build   - Build Docker images"
	@echo "  make docker-up      - Start services"
	@echo "  make docker-down    - Stop services"
	@echo "  make train-all      - Train all models"

setup:
	@echo "Setting up ReleAF AI..."
	bash scripts/setup.sh

install:
	@echo "Installing dependencies..."
	pip install -e .

install-dev:
	@echo "Installing development dependencies..."
	pip install -e ".[dev]"

clean:
	@echo "Cleaning generated files..."
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .pytest_cache/ .mypy_cache/ htmlcov/

test:
	@echo "Running tests..."
	pytest tests/ -v --cov=services --cov=training

test-unit:
	@echo "Running unit tests..."
	pytest tests/unit/ -v

test-integration:
	@echo "Running integration tests..."
	pytest tests/integration/ -v

format:
	@echo "Formatting code..."
	black .
	isort .

lint:
	@echo "Linting code..."
	flake8 services/ training/
	mypy services/ training/

docker-build:
	@echo "Building Docker images..."
	docker-compose build

docker-up:
	@echo "Starting services..."
	docker-compose up -d

docker-down:
	@echo "Stopping services..."
	docker-compose down

docker-logs:
	@echo "Viewing logs..."
	docker-compose logs -f

train-vision-cls:
	@echo "Training vision classifier..."
	python training/vision/train_classifier.py

train-vision-det:
	@echo "Training object detector..."
	python training/vision/train_detector.py

train-llm:
	@echo "Training LLM..."
	python training/llm/train_sft.py

train-all: train-vision-cls train-vision-det train-llm
	@echo "All models trained!"

start-services:
	@echo "Starting all services..."
	bash scripts/start_all_services.sh

stop-services:
	@echo "Stopping all services..."
	bash scripts/stop_all_services.sh

build-rag-index:
	@echo "Building RAG index..."
	bash scripts/build_rag_index.sh

init-db:
	@echo "Initializing databases..."
	python scripts/init_databases.py

check:
	@echo "Running all checks..."
	make format
	make lint
	make test

