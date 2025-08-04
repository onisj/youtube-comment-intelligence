# 🛠️ Environment Setup Guide

Complete guide for setting up the YouTube Comment Intelligence development environment.

## 📋 Prerequisites

### System Requirements
- **OS**: macOS, Linux, or Windows
- **Python**: 3.11 or higher
- **Memory**: 4GB RAM minimum (8GB recommended)
- **Storage**: 2GB free space
- **Network**: Internet connection for package installation

### Required Software
- **Python 3.11+**: [Download from python.org](https://www.python.org/downloads/)
- **Conda/Miniconda**: [Download from conda.io](https://docs.conda.io/en/latest/miniconda.html)
- **Git**: [Download from git-scm.com](https://git-scm.com/downloads)

## 🚀 Quick Environment Setup

### 1. Clone the Repository
```bash
git clone <repository-url>
cd End-to-end-Youtube-Sentiment
```

### 2. Automated Setup (Recommended)
```bash
# Install all dependencies automatically
python install_dependencies.py

# Verify installation
python -c "import flask, streamlit, lightgbm; print('✅ All packages installed successfully!')"
```

### 3. Manual Setup (Advanced Users)
```bash
# Create conda environment
conda create -n youtube python=3.11 -y
conda activate youtube

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

## 🔧 Environment Configuration

### Environment Variables
```bash
# Optional: Set custom ports
export FLASK_PORT=8080
export STREAMLIT_PORT=8501

# Optional: Set log level
export LOG_LEVEL=INFO
```

### Virtual Environment Management
```bash
# Activate environment
conda activate youtube

# Deactivate environment
conda deactivate

# List installed packages
pip list

# Update packages
pip install --upgrade -r requirements.txt
```

## 📦 Package Management

### Core Dependencies
- **Flask 3.0.3**: Web framework for API
- **Streamlit 1.28.1**: Interactive web interface
- **LightGBM 4.5.0**: Gradient boosting for ML
- **Scikit-learn 1.6.1**: Machine learning utilities
- **NLTK 3.9.1**: Natural language processing
- **Pandas 2.2.3**: Data manipulation
- **NumPy 1.24.3**: Numerical computing

### Development Dependencies
- **Pytest 8.4.1**: Testing framework
- **Black**: Code formatting
- **Flake8**: Linting
- **MyPy**: Type checking
- **Pre-commit**: Git hooks

### Visualization Dependencies
- **Matplotlib**: Plotting
- **Seaborn**: Statistical visualization
- **Plotly**: Interactive charts
- **WordCloud**: Text visualization

## 🧪 Testing Environment

### Running Tests
```bash
# Run all tests
python run_tests.py

# Run specific test categories
python -m pytest tests/test_api.py -v
python -m pytest tests/test_model.py -v
python -m pytest tests/test_preprocessing.py -v
python -m pytest tests/test_streamlit.py -v

# Run with coverage
python -m pytest tests/ --cov=app --cov=streamlit_app --cov=src
```

### Test Results
- **70 tests passing** (98.6% success rate)
- **64% code coverage**
- **Comprehensive error handling**
- **Input validation testing**

## 🐳 Docker Environment

### Building Docker Image
```bash
# Build the image
docker build -t youtube-sentiment .

# Run the container
docker run -p 8080:8080 -p 8501:8501 youtube-sentiment
```

### Docker Compose
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## 🔍 Troubleshooting

### Common Issues

#### 1. Package Installation Errors
```bash
# Clear pip cache
pip cache purge

# Upgrade pip
pip install --upgrade pip

# Install with verbose output
pip install -r requirements.txt -v
```

#### 2. Model Loading Errors
```bash
# Check if model files exist
ls -la models/

# Verify file permissions
chmod 644 models/*.pkl
```

#### 3. Port Conflicts
```bash
# Check if ports are in use
lsof -i :8080
lsof -i :8501

# Kill processes using ports
kill -9 <PID>
```

#### 4. Memory Issues
```bash
# Monitor memory usage
top -p $(pgrep python)

# Increase swap space (Linux)
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### Environment Verification
```bash
# Check Python version
python --version

# Check conda environment
conda info --envs

# Verify key packages
python -c "
import flask
import streamlit
import lightgbm
import sklearn
import nltk
import pandas
import numpy
print('✅ All packages installed successfully!')
"
```

## 📊 Performance Optimization

### System Tuning
```bash
# Increase file descriptor limit (Linux)
ulimit -n 4096

# Optimize Python performance
export PYTHONOPTIMIZE=1
export PYTHONUNBUFFERED=1
```

### Memory Optimization
- Use batch processing for large datasets
- Implement garbage collection for long-running processes
- Monitor memory usage with `psutil`

## 🔐 Security Considerations

### API Security
- Use HTTPS in production
- Implement rate limiting
- Add authentication for sensitive endpoints
- Validate all input data

### Data Privacy
- Anonymize user data
- Implement data retention policies
- Use secure storage for API keys
- Regular security audits

## 📈 Monitoring and Logging

### Application Logs
```bash
# View application logs
tail -f logs/app.log

# Monitor system resources
htop
```

### Performance Monitoring
- CPU usage monitoring
- Memory usage tracking
- API response time metrics
- Error rate monitoring

## 🚀 Production Deployment

### Environment Checklist
- [ ] All tests passing
- [ ] Code coverage > 60%
- [ ] Security audit completed
- [ ] Performance testing done
- [ ] Documentation updated
- [ ] Monitoring configured
- [ ] Backup strategy in place

### Deployment Commands
```bash
# Production setup
python install_dependencies.py --type production

# Start applications
python start_apps.py

# Monitor logs
tail -f logs/app.log
```

---

**🎯 Ready to start developing? Run `python install_dependencies.py` and then `python start_apps.py`!**