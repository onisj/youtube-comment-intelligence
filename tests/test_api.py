import pytest
import json
import io
from unittest.mock import patch, Mock
import numpy as np
from app import app

@pytest.fixture
def app_client():
    """Create a test client for the Flask app."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

@pytest.fixture
def sample_comments():
    """Sample comments for testing."""
    return [
        'This video is awesome! I loved it a lot',
        'Very bad explanation. Poor video quality',
        "It's okay, nothing special",
        'Great content, very informative',
        'Terrible video, waste of time'
    ]

class TestAPIEndpoints:
    """Test API endpoints."""
    
    def test_home_endpoint(self, app_client):
        """Test home endpoint."""
        response = app_client.get('/')
        # The home endpoint should return 404 since it doesn't exist
        assert response.status_code == 404

    def test_predict_endpoint_success(self, app_client, sample_comments):
        """Test successful prediction."""
        data = {"comments": sample_comments}
        response = app_client.post('/predict',
                                data=json.dumps(data),
                                content_type='application/json',
                                headers={'X-API-Key': 'testkey123456789'})
        assert response.status_code == 200
        result = json.loads(response.data)
        assert 'predictions' in result
        assert 'sentiments' in result
        assert len(result['predictions']) == len(sample_comments)

    def test_predict_endpoint_no_comments(self, app_client):
        """Test prediction with no comments."""
        data = {"comments": []}
        response = app_client.post('/predict',
                                data=json.dumps(data),
                                content_type='application/json',
                                headers={'X-API-Key': 'testkey123456789'})
        assert response.status_code == 400

    def test_predict_endpoint_missing_comments(self, app_client):
        """Test prediction with missing comments field."""
        data = {"wrong_field": ["test"]}
        response = app_client.post('/predict',
                                data=json.dumps(data),
                                content_type='application/json',
                                headers={'X-API-Key': 'testkey123456789'})
        assert response.status_code == 400

    def test_predict_endpoint_invalid_json(self, app_client):
        """Test prediction with invalid JSON."""
        response = app_client.post('/predict',
                                data='invalid json',
                                content_type='application/json',
                                headers={'X-API-Key': 'testkey123456789'})
        assert response.status_code == 400

    def test_predict_endpoint_large_input(self, app_client):
        """Test prediction with large input."""
        large_comments = ["test"] * 1000
        data = {"comments": large_comments}
        response = app_client.post('/predict',
                                data=json.dumps(data),
                                content_type='application/json',
                                headers={'X-API-Key': 'testkey123456789'})
        assert response.status_code == 200

    def test_predict_with_timestamps_success(self, app_client):
        """Test successful prediction with timestamps."""
        data = {
            "comments": ["Great video", "Bad video"],
            "timestamps": ["2024-01-01T10:00:00Z", "2024-01-01T11:00:00Z"]
        }
        response = app_client.post('/predict_with_timestamps',
                                data=json.dumps(data),
                                content_type='application/json',
                                headers={'X-API-Key': 'testkey123456789'})
        assert response.status_code == 200
        result = json.loads(response.data)
        assert 'predictions' in result
        assert 'timestamps' in result

    def test_predict_with_timestamps_missing_data(self, app_client):
        """Test prediction with timestamps but missing data."""
        data = {"comments": ["test"]}  # Missing timestamps
        response = app_client.post('/predict_with_timestamps',
                                data=json.dumps(data),
                                content_type='application/json',
                                headers={'X-API-Key': 'testkey123456789'})
        assert response.status_code == 200  # Should still work with default timestamps

    def test_generate_chart_success(self, app_client):
        """Test successful chart generation."""
        data = {
            "sentiment_counts": {
                "1": 10,
                "0": 5,
                "-1": 3
            }
        }
        response = app_client.post('/generate_chart',
                                data=json.dumps(data),
                                content_type='application/json',
                                headers={'X-API-Key': 'testkey123456789'})
        assert response.status_code == 200

    def test_generate_chart_no_data(self, app_client):
        """Test chart generation with no data."""
        data = {"sentiment_counts": {}}
        response = app_client.post('/generate_chart',
                                data=json.dumps(data),
                                content_type='application/json',
                                headers={'X-API-Key': 'testkey123456789'})
        assert response.status_code == 400

    def test_generate_chart_zero_counts(self, app_client):
        """Test chart generation with zero counts."""
        data = {"sentiment_counts": {"1": 0, "0": 0, "-1": 0}}
        response = app_client.post('/generate_chart',
                                data=json.dumps(data),
                                content_type='application/json',
                                headers={'X-API-Key': 'testkey123456789'})
        assert response.status_code == 400

    def test_generate_wordcloud_success(self, app_client, sample_comments):
        """Test successful wordcloud generation."""
        data = {"comments": sample_comments}
        response = app_client.post('/generate_wordcloud',
                                data=json.dumps(data),
                                content_type='application/json',
                                headers={'X-API-Key': 'testkey123456789'})
        assert response.status_code == 200

    def test_generate_wordcloud_no_comments(self, app_client):
        """Test wordcloud generation with no comments."""
        data = {"comments": []}
        response = app_client.post('/generate_wordcloud',
                                data=json.dumps(data),
                                content_type='application/json',
                                headers={'X-API-Key': 'testkey123456789'})
        assert response.status_code == 400

    def test_generate_trend_graph_success(self, app_client):
        """Test successful trend graph generation."""
        data = {
            "sentiment_data": [
                {"text": "Great video", "sentiment": 1, "timestamp": "2024-01-01T10:00:00Z"},
                {"text": "Bad video", "sentiment": -1, "timestamp": "2024-01-01T11:00:00Z"},
                {"text": "Okay video", "sentiment": 0, "timestamp": "2024-01-01T12:00:00Z"}
            ]
        }
        response = app_client.post('/generate_trend_graph',
                                data=json.dumps(data),
                                content_type='application/json',
                                headers={'X-API-Key': 'testkey123456789'})
        assert response.status_code == 200

    def test_generate_trend_graph_no_data(self, app_client):
        """Test trend graph generation with no data."""
        data = {"sentiment_data": []}
        response = app_client.post('/generate_trend_graph',
                                data=json.dumps(data),
                                content_type='application/json',
                                headers={'X-API-Key': 'testkey123456789'})
        assert response.status_code == 400

    def test_invalid_methods(self, app_client):
        """Test endpoints with invalid HTTP methods."""
        # Test GET on POST-only endpoints
        response = app_client.get('/predict')
        assert response.status_code == 405  # Method Not Allowed

    @patch('app.load_model')
    def test_model_loading_error(self, mock_load_model, app_client):
        """Test behavior when model loading fails."""
        mock_load_model.side_effect = Exception("Model loading failed")
        
        data = {"comments": ["test"]}
        response = app_client.post('/predict',
                                data=json.dumps(data),
                                content_type='application/json',
                                headers={'X-API-Key': 'testkey123456789'})
        assert response.status_code == 500

    def test_input_validation_empty_strings(self, app_client):
        """Test input validation with empty strings."""
        data = {"comments": ["", "   ", "\n\t"]}
        response = app_client.post('/predict',
                                data=json.dumps(data),
                                content_type='application/json',
                                headers={'X-API-Key': 'testkey123456789'})
        assert response.status_code == 200

    def test_input_validation_very_long_comment(self, app_client):
        """Test input validation with very long comment."""
        long_comment = "a" * 10000  # 10k character comment
        data = {"comments": [long_comment]}
        response = app_client.post('/predict',
                                data=json.dumps(data),
                                content_type='application/json',
                                headers={'X-API-Key': 'testkey123456789'})
        assert response.status_code == 200

    def test_input_validation_special_characters(self, app_client):
        """Test input validation with special characters."""
        special_chars = ["!@#$%^&*()", "🎬📱💻", "测试文本", "مرحبا بالعالم"]
        data = {"comments": special_chars}
        response = app_client.post('/predict',
                                data=json.dumps(data),
                                content_type='application/json',
                                headers={'X-API-Key': 'testkey123456789'})
        assert response.status_code == 200

    def test_content_type_validation(self, app_client):
        """Test that endpoints require JSON content type."""
        data = {"comments": ["test"]}
        response = app_client.post('/predict',
                                data=data,
                                content_type='text/plain',
                                headers={'X-API-Key': 'testkey123456789'})
        assert response.status_code == 400

    def test_request_size_limit(self, app_client):
        """Test behavior with very large request."""
        # Create a very large request
        large_data = {"comments": ["test"] * 10000}
        response = app_client.post('/predict',
                                data=json.dumps(large_data),
                                content_type='application/json',
                                headers={'X-API-Key': 'testkey123456789'})
        
        # Should either succeed or fail gracefully
        assert response.status_code in [200, 400, 413, 500] 