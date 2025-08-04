# 🎬 YouTube Comment intelligence

A comprehensive machine learning pipeline for analyzing sentiment in YouTube comments using advanced NLP techniques and LightGBM classification.

## 📋 Overview

This project provides a complete solution for YouTube comment sentiment analysis, featuring a robust machine learning pipeline, interactive web interface, and RESTful API. Built with modern Python technologies and best practices in MLOps.

## 🚀 Key Features

- **🤖 Advanced ML Pipeline**: Complete workflow from data preprocessing to model deployment
- **🌐 Interactive Web Interface**: Streamlit-based dashboard for real-time analysis
- **🔌 RESTful API**: Production-ready Flask API with comprehensive endpoints
- **📊 Real-time Analytics**: Live sentiment analysis with confidence scores
- **🎯 YouTube Integration**: Direct comment fetching from YouTube videos
- **🐳 Docker Support**: Containerized deployment for easy scaling
- **🧪 Comprehensive Testing**: 70+ tests with 64% code coverage
- **📈 Visualization Tools**: Word clouds, trend graphs, and sentiment charts

## 📁 Project Architecture

```
End-to-end-Youtube-Sentiment/
├── app.py                    # Flask REST API
├── streamlit_app.py          # Interactive web dashboard
├── start_apps.py            # Application launcher
├── install_dependencies.py   # Automated setup script
├── run_tests.py             # Test suite runner
├── models/                  # Trained ML models
│   ├── lgbm_model.pkl      # LightGBM classifier
│   └── tfidf_vectorizer.pkl # TF-IDF vectorizer
├── logs/                    # Application logs
├── docs/                    # Documentation
├── assets/                  # Visualizations & images
├── tests/                   # Comprehensive test suite
├── src/                     # Core ML pipeline
├── notebooks/               # Jupyter notebooks
└── requirements.txt         # Dependencies
```

## 🛠️ Quick Start

### Prerequisites
- Python 3.11+
- Conda (recommended)

### Installation

**Option 1: Automated Setup (Recommended)**
```bash
# Install all dependencies
python install_dependencies.py

# Install only production dependencies
python install_dependencies.py --type production

# Install only development tools
python install_dependencies.py --type development
```

**Option 2: Manual Setup**
```bash
# Create and activate environment
conda create -n youtube python=3.11 -y
conda activate youtube

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

### Launch Applications

**Quick Launch (Both Apps)**
```bash
python start_apps.py
```

**Manual Launch**
```bash
# Terminal 1: Start Flask API
conda activate youtube && python app.py

# Terminal 2: Start Streamlit Dashboard
conda activate youtube && streamlit run streamlit_app.py --server.port=8501
```

## 🌐 Access Points

| Service | URL | Description |
|---------|-----|-------------|
| **Flask API** | http://localhost:8080 | RESTful sentiment analysis API |
| **Streamlit Dashboard** | http://localhost:8501 | Interactive web interface |
| **API Documentation** | http://localhost:8080/docs | Complete API reference |

## 🔌 API Usage

### Basic Sentiment Analysis
```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "comments": [
      "This video is absolutely amazing!",
      "Terrible explanation, very confusing",
      "Pretty good content overall"
    ]
  }'
```

### Response Format
```json
[
  {
    "comment": "This video is absolutely amazing!",
    "sentiment": 1
  },
  {
    "comment": "Terrible explanation, very confusing",
    "sentiment": 0
  },
  {
    "comment": "Pretty good content overall",
    "sentiment": 1
  }
]
```

## 🧪 Testing

Run the comprehensive test suite:
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

**Test Results:**
- ✅ **70 tests passing** (98.6% success rate)
- ✅ **64% code coverage**
- ✅ **Comprehensive error handling**
- ✅ **Input validation testing**

## 🎯 Use Cases

### For Data Scientists
- Complete ML pipeline with preprocessing, training, and evaluation
- Jupyter notebooks for experimentation
- Model performance metrics and visualization

### For Developers
- RESTful API for integration
- Comprehensive test suite
- Docker containerization

### For End Users
- Interactive web dashboard
- Real-time sentiment analysis
- Beautiful visualizations and charts

## 🔧 Configuration

### Environment Variables
```bash
# Optional: Set custom ports
export FLASK_PORT=8080
export STREAMLIT_PORT=8501
```

### Model Parameters
The trained model uses:
- **Algorithm**: LightGBM
- **Vectorization**: TF-IDF
- **Features**: Text preprocessing with lemmatization
- **Classes**: Positive (1), Negative (0)

## 📊 Performance

- **Accuracy**: High-performance sentiment classification
- **Speed**: Real-time processing (< 1 second per comment)
- **Scalability**: Handles batch processing of thousands of comments
- **Reliability**: Comprehensive error handling and validation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- YouTube Data API for comment access
- NLTK for natural language processing
- LightGBM for efficient gradient boosting
- Streamlit for interactive web interface
- Flask for RESTful API framework

---

**🎉 Ready to analyze YouTube sentiment? Start with `python install_dependencies.py` and then `python start_apps.py`!**
