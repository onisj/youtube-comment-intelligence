import pytest
import tempfile
import os
import sys
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

@pytest.fixture
def sample_comments():
    """Sample comments for testing."""
    return [
        "This video is awesome! I loved it a lot",
        "Very bad explanation. Poor video quality",
        "It's okay, nothing special",
        "Great content, very informative",
        "Terrible video, waste of time"
    ]

@pytest.fixture
def sample_sentiment_data():
    """Sample sentiment data with timestamps."""
    return [
        {"text": "Great video!", "timestamp": "2024-01-01T10:00:00Z"},
        {"text": "Bad content", "timestamp": "2024-01-01T11:00:00Z"},
        {"text": "Okay video", "timestamp": "2024-01-01T12:00:00Z"}
    ]

@pytest.fixture
def mock_model():
    """Mock model for testing."""
    mock = Mock()
    mock.predict.return_value = np.array([1, -1, 0, 1, -1])
    mock.predict_proba.return_value = np.array([
        [0.1, 0.2, 0.7],
        [0.8, 0.1, 0.1],
        [0.2, 0.6, 0.2],
        [0.1, 0.1, 0.8],
        [0.9, 0.05, 0.05]
    ])
    return mock

@pytest.fixture
def mock_vectorizer():
    """Mock vectorizer for testing."""
    mock = Mock()
    mock.transform.return_value = np.array([[0.1, 0.2, 0.3, 0.4, 0.5]])
    return mock

@pytest.fixture
def temp_model_files():
    """Create temporary model files for testing."""
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as model_file:
        model_file.write(b'mock_model_data')
        model_path = model_file.name
    
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as vectorizer_file:
        vectorizer_file.write(b'mock_vectorizer_data')
        vectorizer_path = vectorizer_file.name
    
    yield model_path, vectorizer_path
    
    # Cleanup
    os.unlink(model_path)
    os.unlink(vectorizer_path)

@pytest.fixture
def sample_csv_data():
    """Sample CSV data for testing."""
    return pd.DataFrame({
        'text': [
            "This is a positive comment",
            "This is a negative comment", 
            "This is a neutral comment"
        ],
        'sentiment': [1, -1, 0]
    })

@pytest.fixture
def app_client():
    """Flask test client."""
    from app import app
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client 