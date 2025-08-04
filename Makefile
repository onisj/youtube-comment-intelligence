# YouTube Comment Intelligence MLOps Project Makefile

.PHONY: help setup install install-dev clean test lint format docker-build docker-up docker-down jupyter mlflow dvc-repro streamlit

# Default target
help:
	@echo "YouTube Comment Intelligence MLOps Project"
	@echo "========================================"
	@echo "Available commands:"
	@echo "  setup          - Set up the development environment"
	@echo "  install        - Install production dependencies"
	@echo "  install-dev    - Install all dependencies (production + development)"
	@echo "  install-deps   - Install all dependencies using installer script"
	@echo "  clean          - Clean up temporary files and caches"
	@echo "  test           - Run tests"
	@echo "  lint           - Run code linting"
	@echo "  format         - Format code with black and isort"
	@echo "  docker-build   - Build Docker images"
	@echo "  docker-up      - Start all services with Docker Compose"
	@echo "  docker-down    - Stop all Docker services"
	@echo "  jupyter        - Start Jupyter Lab"
	@echo "  mlflow         - Start MLflow UI"
	@echo "  dvc-repro      - Reproduce DVC pipeline"
	@echo "  streamlit      - Start Streamlit demo app"
	@echo "  train          - Train the model"
	@echo "  predict        - Run predictions"

# Environment setup
setup:
	@echo "🚀 Setting up development environment..."
	@chmod +x setup_environment.sh
	@./setup_environment.sh

# Install dependencies
install:
	@echo "📦 Installing production dependencies..."
	python install_dependencies.py --type production

install-dev:
	@echo "🛠️ Installing all dependencies..."
	python install_dependencies.py --type all

install-deps:
	@echo "📦 Installing all dependencies..."
	python install_dependencies.py

# Clean up
clean:
	@echo "🧹 Cleaning up..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name ".coverage" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +

# Testing
test:
	@echo "🧪 Running tests..."
	pytest tests/ -v --cov=app --cov=streamlit_app --cov=src --cov-report=html --cov-report=term

test-fast:
	@echo "⚡ Running fast tests..."
	pytest tests/ -v -x

test-unit:
	@echo "🧪 Running unit tests..."
	pytest tests/ -v -m unit

test-api:
	@echo "🌐 Running API tests..."
	pytest tests/test_api.py -v

test-model:
	@echo "🤖 Running model tests..."
	pytest tests/test_model.py -v

test-coverage:
	@echo "📊 Running tests with coverage..."
	pytest tests/ --cov=app --cov=streamlit_app --cov=src --cov-report=html --cov-report=term-missing

test-parallel:
	@echo "⚡ Running tests in parallel..."
	pytest tests/ -n auto

# Code quality
lint:
	@echo "🔍 Running linting..."
	flake8 src/ tests/ --max-line-length=88 --ignore=E203,W503
	mypy src/

format:
	@echo "✨ Formatting code..."
	black src/ tests/ notebooks/
	isort src/ tests/ notebooks/

# Docker commands
docker-build:
	@echo "🐳 Building Docker images..."
	docker-compose build

docker-up:
	@echo "🚀 Starting all services..."
	docker-compose up -d
	@echo "Services available at:"
	@echo "  - Main App: http://localhost:5000"
	@echo "  - MLflow: http://localhost:5001"
	@echo "  - Jupyter: http://localhost:8888 (token: youtube-sentiment)"
	@echo "  - Streamlit: http://localhost:8501"
	@echo "  - Grafana: http://localhost:3000 (admin/admin)"

docker-down:
	@echo "🛑 Stopping all services..."
	docker-compose down

docker-logs:
	@echo "📋 Showing Docker logs..."
	docker-compose logs -f

# Development servers
jupyter:
	@echo "📓 Starting Jupyter Lab..."
	jupyter lab --ip=0.0.0.0 --port=8888 --no-browser

mlflow:
	@echo "📊 Starting MLflow UI..."
	mlflow ui --host 0.0.0.0 --port 5001

streamlit:
	@echo "🎯 Starting Streamlit demo..."
	streamlit run streamlit_app.py --server.port=8501

# DVC and ML Pipeline
dvc-repro:
	@echo "🔄 Reproducing DVC pipeline..."
	dvc repro

dvc-dag:
	@echo "📊 Showing DVC pipeline DAG..."
	dvc dag

dvc-metrics:
	@echo "📈 Showing DVC metrics..."
	dvc metrics show

# ML Operations
train:
	@echo "🤖 Training model..."
	python src/model/model_building.py

evaluate:
	@echo "📊 Evaluating model..."
	python src/model/model_evaluation.py

predict:
	@echo "🔮 Running predictions..."
	python app.py

# Data operations
data-ingestion:
	@echo "📥 Running data ingestion..."
	python src/data/data_ingestion.py

data-preprocessing:
	@echo "🔧 Running data preprocessing..."
	python src/data/data_preprocessing.py

# Environment info
info:
	@echo "ℹ️ Environment Information:"
	@echo "Python version: $$(python --version)"
	@echo "Pip version: $$(pip --version)"
	@echo "Virtual environment: $$VIRTUAL_ENV"
	@echo "Current directory: $$(pwd)"
	@echo "Git branch: $$(git branch --show-current 2>/dev/null || echo 'Not a git repo')"

# Install and setup everything
bootstrap: setup install-dev
	@echo "🎉 Bootstrap completed! Environment is ready for development."