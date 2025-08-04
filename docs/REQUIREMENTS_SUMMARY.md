# 📦 Dependencies Overview

Comprehensive guide to all project dependencies, their purposes, and installation strategies.

## 📋 Overview

This project uses a carefully curated set of Python packages to provide a complete machine learning pipeline for YouTube comment intelligence. The dependencies are organized into logical categories for easy management and deployment.

## 🎯 Dependency Categories

### Core Application Dependencies
Essential packages for the main application functionality.

| Package | Version | Purpose | Critical |
|---------|---------|---------|----------|
| **Flask** | 3.0.3 | Web framework for REST API | ✅ |
| **Flask-Cors** | 5.0.0 | Cross-origin resource sharing | ✅ |
| **Streamlit** | 1.28.1 | Interactive web dashboard | ✅ |
| **Joblib** | 1.4.2 | Model serialization | ✅ |

### Machine Learning & Data Science
Core ML libraries for sentiment analysis and data processing.

| Package | Version | Purpose | Critical |
|---------|---------|---------|----------|
| **LightGBM** | 4.5.0 | Gradient boosting classifier | ✅ |
| **Scikit-learn** | 1.6.1 | Machine learning utilities | ✅ |
| **NumPy** | 1.24.3 | Numerical computing | ✅ |
| **Pandas** | 2.2.3 | Data manipulation | ✅ |
| **NLTK** | 3.9.1 | Natural language processing | ✅ |
| **MLflow** | 2.17.0 | ML experiment tracking | ⚠️ |

### Data Validation & Monitoring
Packages for data validation and system monitoring.

| Package | Version | Purpose | Critical |
|---------|---------|---------|----------|
| **Pydantic** | 1.10.13 | Data validation | ✅ |
| **Cerberus** | 1.3.5 | Schema validation | ⚠️ |
| **Psutil** | 5.9.6 | System monitoring | ⚠️ |

### Visualization & Analytics
Libraries for creating charts and visualizations.

| Package | Version | Purpose | Critical |
|---------|---------|---------|----------|
| **Matplotlib** | 3.8.4 | Plotting library | ✅ |
| **Seaborn** | 0.13.2 | Statistical visualization | ✅ |
| **Plotly** | 5.18.0 | Interactive charts | ✅ |
| **WordCloud** | 1.9.3 | Text visualization | ✅ |

### Development & Testing
Tools for code quality, testing, and development.

| Package | Version | Purpose | Critical |
|---------|---------|---------|----------|
| **Pytest** | 8.4.1 | Testing framework | ✅ |
| **Pytest-cov** | 6.2.1 | Coverage reporting | ✅ |
| **Pytest-mock** | 3.14.1 | Mocking utilities | ✅ |
| **Pytest-xdist** | 3.8.0 | Parallel testing | ⚠️ |

### Code Quality & Formatting
Tools for maintaining code quality and consistency.

| Package | Version | Purpose | Critical |
|---------|---------|---------|----------|
| **Black** | 24.1.1 | Code formatting | ✅ |
| **Isort** | 5.13.2 | Import sorting | ✅ |
| **Flake8** | 7.0.0 | Linting | ✅ |
| **MyPy** | 1.8.0 | Type checking | ⚠️ |

### Security & Analysis
Security scanning and code analysis tools.

| Package | Version | Purpose | Critical |
|---------|---------|---------|----------|
| **Bandit** | 1.7.7 | Security scanning | ⚠️ |
| **Pre-commit** | 3.6.2 | Git hooks | ⚠️ |

### MLOps & Monitoring
Tools for ML operations and monitoring.

| Package | Version | Purpose | Critical |
|---------|---------|---------|----------|
| **Evidently** | 0.4.8 | ML monitoring | ⚠️ |
| **Great Expectations** | 0.18.15 | Data validation | ⚠️ |

## 🚀 Installation Strategies

### Option 1: Complete Installation (Recommended)
```bash
# Install all dependencies
python install_dependencies.py

# Verify installation
python -c "import flask, streamlit, lightgbm; print('✅ All packages installed!')"
```

### Option 2: Production Installation
```bash
# Install only production dependencies
python install_dependencies.py --type production

# Core packages only
pip install flask flask-cors streamlit lightgbm scikit-learn numpy pandas nltk
```

### Option 3: Development Installation
```bash
# Install development tools
python install_dependencies.py --type development

# Testing and quality tools
pip install pytest pytest-cov black isort flake8 mypy
```

### Option 4: Manual Installation
```bash
# Create environment
conda create -n youtube python=3.11 -y
conda activate youtube

# Install from requirements
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

## 📊 Dependency Analysis

### Critical Dependencies
These packages are essential for core functionality:
- **Flask**: Web API framework
- **Streamlit**: Web dashboard
- **LightGBM**: ML classifier
- **Scikit-learn**: ML utilities
- **NLTK**: Text processing
- **Pandas/NumPy**: Data handling

### Optional Dependencies
These packages enhance functionality but aren't critical:
- **MLflow**: Experiment tracking
- **Evidently**: ML monitoring
- **Great Expectations**: Data validation
- **MyPy**: Type checking
- **Bandit**: Security scanning

### Development Dependencies
These packages are only needed for development:
- **Pytest**: Testing framework
- **Black**: Code formatting
- **Isort**: Import sorting
- **Flake8**: Linting
- **Pre-commit**: Git hooks

## 🔧 Version Compatibility

### Python Version Requirements
- **Minimum**: Python 3.11
- **Recommended**: Python 3.11+
- **Tested**: Python 3.11.8

### Key Version Constraints
```python
# Core ML packages
scikit-learn>=1.6.1,<2.0.0
lightgbm>=4.5.0,<5.0.0
numpy>=1.24.3,<2.0.0

# Web frameworks
flask>=3.0.3,<4.0.0
streamlit>=1.28.1,<2.0.0

# Data processing
pandas>=2.2.3,<3.0.0
nltk>=3.9.1,<4.0.0
```

## 🐳 Docker Dependencies

### Base Image
```dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*
```

### Python Dependencies
```dockerfile
# Copy requirements
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt
```

## 🔍 Dependency Conflicts

### Known Issues
1. **NumPy Version**: Some packages require numpy<2, others need numpy>=2
   - **Solution**: Using numpy==1.24.3 for compatibility

2. **Pydantic Version**: Evidently requires pydantic<2, others need pydantic>=2
   - **Solution**: Using pydantic==1.10.13 for compatibility

3. **Scikit-learn Version**: Model was trained with 1.6.1, but some packages require 1.5.x
   - **Solution**: Using scikit-learn==1.6.1 for model compatibility

### Resolution Strategy
```bash
# Check for conflicts
pip check

# Resolve conflicts
pip install --upgrade pip
pip install -r requirements.txt --no-deps
pip install -r requirements.txt
```

## 📈 Performance Impact

### Memory Usage
- **LightGBM**: ~200MB for model loading
- **NLTK**: ~50MB for language data
- **Streamlit**: ~100MB for web interface
- **Flask**: ~50MB for API server

### Startup Time
- **Cold Start**: ~5-10 seconds
- **Warm Start**: ~2-3 seconds
- **Model Loading**: ~1-2 seconds

### Package Sizes
- **Total Installation**: ~2GB
- **Core Packages**: ~500MB
- **Development Tools**: ~1.5GB

## 🔒 Security Considerations

### Security Scanning
```bash
# Run security scan
bandit -r . -f json -o bandit-report.json

# Check for vulnerabilities
safety check
```

### Vulnerable Packages
- **None currently identified**
- **Regular updates**: Monthly security updates
- **Dependency monitoring**: Automated vulnerability scanning

## 🚀 Optimization Strategies

### Minimal Installation
```bash
# Core packages only
pip install flask streamlit lightgbm scikit-learn numpy pandas nltk
```

### Development Installation
```bash
# Full development environment
pip install -r requirements.txt
pre-commit install
```

### Production Installation
```bash
# Production-optimized
pip install flask flask-cors streamlit lightgbm scikit-learn numpy pandas nltk
```

## 📋 Maintenance

### Regular Updates
```bash
# Update all packages
pip install --upgrade -r requirements.txt

# Update specific packages
pip install --upgrade flask streamlit lightgbm
```

### Dependency Monitoring
- **Monthly**: Check for security updates
- **Quarterly**: Review for performance improvements
- **Annually**: Major version compatibility review

### Version Pinning
```bash
# Generate requirements with exact versions
pip freeze > requirements.txt

# Install exact versions
pip install -r requirements.txt
```

## 🎯 Best Practices

### Installation Order
1. **System Dependencies**: Python, conda
2. **Core ML Packages**: numpy, pandas, scikit-learn
3. **ML Framework**: lightgbm
4. **Web Frameworks**: flask, streamlit
5. **NLP Libraries**: nltk
6. **Development Tools**: pytest, black, flake8

### Environment Management
```bash
# Use virtual environments
conda create -n youtube python=3.11
conda activate youtube

# Pin versions for reproducibility
pip freeze > requirements.txt
```

### Testing Dependencies
```bash
# Test core functionality
python -c "import flask, streamlit, lightgbm; print('✅ Core packages working')"

# Test ML functionality
python -c "import sklearn, nltk; print('✅ ML packages working')"

# Test development tools
python -c "import pytest, black; print('✅ Dev tools working')"
```

---

**📦 Ready to install? Run `python install_dependencies.py` for the complete setup!** 