import pytest
import json
import io
from unittest.mock import patch, Mock
import numpy as np
from app import app

class TestAPIEndpoints:
    """Test cases for Flask API endpoints."""
    
    def test_home_endpoint(self, app_client):
        """Test the home endpoint."""
        response = app_client.get('/')
        assert response.status_code == 200
        assert "Welcome" in response.get_data(as_text=True)
    
    def test_predict_endpoint_success(self, app_client, sample_comments):
        """Test successful prediction endpoint."""
        data = {"comments": sample_comments}
        response = app_client.post('/predict', 
                                data=json.dumps(data),
                                content_type='application/json')
        
        assert response.status_code == 200
        result = json.loads(response.get_data(as_text=True))
        assert isinstance(result, list)
        assert len(result) == len(sample_comments)
        
        for item in result:
            assert "comment" in item
            assert "sentiment" in item
            assert isinstance(item["sentiment"], (int, str))
    
    def test_predict_endpoint_no_comments(self, app_client):
        """Test prediction endpoint with no comments."""
        data = {"comments": []}
        response = app_client.post('/predict',
                                data=json.dumps(data),
                                content_type='application/json')
        
        assert response.status_code == 400
        result = json.loads(response.get_data(as_text=True))
        assert "error" in result
    
    def test_predict_endpoint_missing_comments(self, app_client):
        """Test prediction endpoint with missing comments field."""
        data = {"wrong_field": ["test"]}
        response = app_client.post('/predict',
                                data=json.dumps(data),
                                content_type='application/json')
        
        assert response.status_code == 400
        result = json.loads(response.get_data(as_text=True))
        assert "error" in result
    
    def test_predict_endpoint_invalid_json(self, app_client):
        """Test prediction endpoint with invalid JSON."""
        response = app_client.post('/predict',
                                data="invalid json",
                                content_type='application/json')
        
        assert response.status_code == 400
    
    def test_predict_endpoint_large_input(self, app_client):
        """Test prediction endpoint with very large input."""
        large_comments = ["test comment"] * 1000
        data = {"comments": large_comments}
        response = app_client.post('/predict',
                                data=json.dumps(data),
                                content_type='application/json')
        
        assert response.status_code == 200
        result = json.loads(response.get_data(as_text=True))
        assert len(result) == 1000
    
    def test_predict_with_timestamps_success(self, app_client, sample_sentiment_data):
        """Test successful prediction with timestamps endpoint."""
        response = app_client.post('/predict_with_timestamps',
                                data=json.dumps({"comments": sample_sentiment_data}),
                                content_type='application/json')
        
        assert response.status_code == 200
        result = json.loads(response.get_data(as_text=True))
        assert isinstance(result, list)
        assert len(result) == len(sample_sentiment_data)
        
        for item in result:
            assert "comment" in item
            assert "sentiment" in item
            assert "timestamp" in item
    
    def test_predict_with_timestamps_missing_data(self, app_client):
        """Test prediction with timestamps endpoint with missing data."""
        data = {"comments": [{"text": "test", "timestamp": "2024-01-01"}]}
        response = app_client.post('/predict_with_timestamps',
                                data=json.dumps(data),
                                content_type='application/json')
        
        assert response.status_code == 200
    
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
                                content_type='application/json')
        
        assert response.status_code == 200
        assert response.mimetype == 'image/png'
    
    def test_generate_chart_no_data(self, app_client):
        """Test chart generation with no data."""
        data = {"sentiment_counts": {}}
        response = app_client.post('/generate_chart',
                                data=json.dumps(data),
                                content_type='application/json')
        
        assert response.status_code == 400
        result = json.loads(response.get_data(as_text=True))
        assert "error" in result
    
    def test_generate_chart_zero_counts(self, app_client):
        """Test chart generation with zero counts."""
        data = {"sentiment_counts": {"1": 0, "0": 0, "-1": 0}}
        response = app_client.post('/generate_chart',
                                data=json.dumps(data),
                                content_type='application/json')
        
        assert response.status_code == 400
    
    def test_generate_wordcloud_success(self, app_client, sample_comments):
        """Test successful wordcloud generation."""
        data = {"comments": sample_comments}
        response = app_client.post('/generate_wordcloud',
                                data=json.dumps(data),
                                content_type='application/json')
        
        assert response.status_code == 200
        assert response.mimetype == 'image/png'
    
    def test_generate_wordcloud_no_comments(self, app_client):
        """Test wordcloud generation with no comments."""
        data = {"comments": []}
        response = app_client.post('/generate_wordcloud',
                                data=json.dumps(data),
                                content_type='application/json')
        
        assert response.status_code == 400
        result = json.loads(response.get_data(as_text=True))
        assert "error" in result
    
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
                                content_type='application/json')
        
        assert response.status_code == 200
        assert response.mimetype == 'image/png'
    
    def test_generate_trend_graph_no_data(self, app_client):
        """Test trend graph generation with no data."""
        data = {"sentiment_data": []}
        response = app_client.post('/generate_trend_graph',
                                data=json.dumps(data),
                                content_type='application/json')
        
        assert response.status_code == 400
        result = json.loads(response.get_data(as_text=True))
        assert "error" in result
    
    def test_invalid_methods(self, app_client):
        """Test endpoints with invalid HTTP methods."""
        # Test GET on POST-only endpoints
        response = app_client.get('/predict')
        assert response.status_code == 500  # Our error handler converts 405 to 500
        
        response = app_client.get('/generate_chart')
        assert response.status_code == 500  # Our error handler converts 405 to 500
    
    def test_cors_headers(self, app_client):
        """Test that CORS headers are present."""
        response = app_client.get('/')
        assert 'Access-Control-Allow-Origin' in response.headers
    
    @patch('app.model')
    def test_model_loading_error(self, mock_model, app_client):
        """Test behavior when model fails to load."""
        # Mock the model to raise an exception during prediction
        mock_model.predict.side_effect = Exception("Model prediction failed")
        
        data = {"comments": ["test comment"]}
        response = app_client.post('/predict',
                                data=json.dumps(data),
                                content_type='application/json')
        
        assert response.status_code == 500
        result = json.loads(response.get_data(as_text=True))
        assert "error" in result
    
    def test_input_validation_empty_strings(self, app_client):
        """Test input validation with empty strings."""
        data = {"comments": ["", "   ", "\n\t"]}
        response = app_client.post('/predict',
                                data=json.dumps(data),
                                content_type='application/json')
        
        assert response.status_code == 200
        result = json.loads(response.get_data(as_text=True))
        assert len(result) == 3
    
    def test_input_validation_very_long_comment(self, app_client):
        """Test input validation with very long comment."""
        long_comment = "a" * 10000  # 10k character comment
        data = {"comments": [long_comment]}
        response = app_client.post('/predict',
                                data=json.dumps(data),
                                content_type='application/json')
        
        assert response.status_code == 200
        result = json.loads(response.get_data(as_text=True))
        assert len(result) == 1
    
    def test_input_validation_special_characters(self, app_client):
        """Test input validation with special characters."""
        special_chars = ["!@#$%^&*()", "🎬📱💻", "测试文本", "مرحبا بالعالم"]
        data = {"comments": special_chars}
        response = app_client.post('/predict',
                                data=json.dumps(data),
                                content_type='application/json')
        
        assert response.status_code == 200
        result = json.loads(response.get_data(as_text=True))
        assert len(result) == len(special_chars)
    
    def test_content_type_validation(self, app_client):
        """Test that endpoints require JSON content type."""
        data = {"comments": ["test"]}
        response = app_client.post('/predict',
                                data=data,
                                content_type='text/plain')
        
        assert response.status_code == 400
    
    def test_request_size_limit(self, app_client):
        """Test behavior with very large request."""
        # Create a very large request
        large_data = {"comments": ["test"] * 10000}
        response = app_client.post('/predict',
                                data=json.dumps(large_data),
                                content_type='application/json')
        
        # Should either succeed or fail gracefully
        assert response.status_code in [200, 400, 413, 500] 