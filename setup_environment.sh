#!/bin/bash

# YouTube Sentiment Analysis MLOps Environment Setup
# This script sets up the optimal development environment for the project

set -e  # Exit on any error

echo "🚀 Setting up MLOps Environment for YouTube Sentiment Analysis"
echo "================================================="

# Check if Python 3.8+ is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "✅ Python version: $PYTHON_VERSION"

# Create virtual environment
echo "📦 Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install core dependencies
echo "📚 Installing core dependencies..."
pip install -r requirements.txt

# Install development dependencies
echo "🛠️ Installing development dependencies..."
pip install \
    jupyter \
    jupyterlab \
    black \
    flake8 \
    pytest \
    pytest-cov \
    pre-commit \
    ipykernel \
    notebook \
    streamlit \
    fastapi \
    uvicorn

# Install MLOps tools
echo "🤖 Installing additional MLOps tools..."
pip install \
    wandb \
    optuna \
    evidently \
    great-expectations \
    bentoml

# Setup pre-commit hooks
echo "🔒 Setting up pre-commit hooks..."
if [ ! -f ".pre-commit-config.yaml" ]; then
    cat > .pre-commit-config.yaml << EOF
repos:
  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black
        language_version: python3
        
  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        args: [--max-line-length=88, --ignore=E203,W503]
        
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
EOF
fi

pre-commit install
echo "✅ Pre-commit hooks installed"

# Create Jupyter kernel
echo "📓 Setting up Jupyter kernel..."
python -m ipykernel install --user --name=youtube-sentiment --display-name="YouTube Sentiment Analysis"

# Initialize DVC if not already done
echo "📊 Checking DVC setup..."
if [ ! -d ".dvc" ]; then
    dvc init
    echo "✅ DVC initialized"
else
    echo "✅ DVC already initialized"
fi

# Create data directories
echo "📁 Creating data directories..."
mkdir -p data/{raw,interim,processed,external}
mkdir -p models
mkdir -p reports/{figures,metrics}
mkdir -p logs

echo "✅ Data directories created"

# Create environment configuration file
echo "⚙️ Creating environment configuration..."
cat > .env << EOF
# Environment Configuration
PYTHON_ENV=development
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=youtube-sentiment-analysis
DVC_REMOTE=local
LOG_LEVEL=INFO
EOF

echo "✅ Environment configuration created"

echo ""
echo "🎉 Environment setup completed successfully!"
echo "================================================="
echo "Next steps:"
echo "1. Activate the environment: source venv/bin/activate"
echo "2. Start Jupyter Lab: jupyter lab"
echo "3. Run MLflow UI: mlflow ui"
echo "4. Execute DVC pipeline: dvc repro"
echo "5. Start development server: python app.py"
echo ""
echo "📝 Available commands:"
echo "  - Format code: black ."
echo "  - Lint code: flake8 ."
echo "  - Run tests: pytest"
echo "  - DVC pipeline: dvc repro"
echo "  - MLflow UI: mlflow ui"
echo "================================================="