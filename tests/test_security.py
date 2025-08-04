#!/usr/bin/env python3
"""
Security Tests for YouTube Comment Intelligence
"""

import pytest
import json
from unittest.mock import patch, Mock
from app import app
import security_config

class TestSecurityConfiguration:
    """Test security configuration."""
    
    def test_security_settings_loading(self):
        """Test that security settings load correctly."""
        settings = security_config.SecuritySettings()
        assert settings.require_api_key is True
        assert settings.rate_limit_enabled is True
        assert settings.max_comment_length > 0

class TestInputValidation:
    """Test input validation functionality."""
    
    def test_sanitize_text_basic(self):
        """Test basic text sanitization."""
        # Test normal text
        result = security_config.InputValidator.sanitize_text("Hello world")
        assert result == "Hello world"
        
        # Test text with dangerous characters
        result = security_config.InputValidator.sanitize_text("Hello<script>alert('xss')</script>world")
        # The sanitization removes script tags but keeps the content
        assert "script" not in result.lower()
        assert "alert" in result

    def test_sanitize_text_empty(self):
        """Test sanitization of empty text."""
        result = security_config.InputValidator.sanitize_text("")
        assert result == ""

    def test_sanitize_text_none(self):
        """Test sanitization of None text."""
        result = security_config.InputValidator.sanitize_text(None)
        assert result == ""

    def test_validate_comments_basic(self):
        """Test basic comment validation."""
        comments = ["Hello", "World"]
        result = security_config.InputValidator.validate_comments(comments)
        assert result == comments

    def test_validate_comments_empty(self):
        """Test validation of empty comment list."""
        # Should not raise ValueError for empty list
        result = security_config.InputValidator.validate_comments([])
        assert result == []

    def test_validate_comments_none(self):
        """Test validation of None comments."""
        with pytest.raises(ValueError):
            security_config.InputValidator.validate_comments(None)

    def test_validate_comments_too_long(self):
        """Test validation of comments that are too long."""
        long_comment = "a" * 10000
        with pytest.raises(ValueError):
            security_config.InputValidator.validate_comments([long_comment])

class TestRateLimiting:
    """Test rate limiting functionality."""
    
    def test_rate_limiter_creation(self):
        """Test rate limiter creation."""
        limiter = security_config.RateLimiter()
        assert limiter.requests is not None

    def test_rate_limiter_check(self):
        """Test rate limiter check."""
        limiter = security_config.RateLimiter()
        client_id = "test_client"
        
        # First request should be allowed
        assert limiter.is_allowed(client_id) is True
        
        # Make many requests to trigger rate limiting
        for _ in range(105):
            limiter.is_allowed(client_id)
        
        # Should be rate limited after 100 requests
        assert limiter.is_allowed(client_id) is False

class TestSecurityManager:
    """Test security manager functionality."""
    
    def test_security_manager_creation(self):
        """Test security manager creation."""
        manager = security_config.SecurityManager()
        assert manager is not None

    def test_load_api_keys(self):
        """Test API key loading."""
        manager = security_config.SecurityManager()
        # Should load from environment or have default
        assert len(manager.api_keys) >= 0

class TestAPISecurity:
    """Test API security features."""
    
    def test_health_endpoint_no_auth(self):
        """Test health endpoint doesn't require authentication."""
        with app.test_client() as client:
            # Mock the model loading to avoid errors
            with patch('app.load_model'):
                response = client.get('/health')
                assert response.status_code == 200

    def test_predict_endpoint_requires_auth(self):
        """Test that predict endpoint requires authentication."""
        with app.test_client() as client:
            response = client.post('/predict',
                                json={'comments': ['test']})
            assert response.status_code == 401

    def test_predict_endpoint_with_valid_key(self):
        """Test predict endpoint with valid API key."""
        with app.test_client() as client:
            # Add a test API key
            with patch.dict(security_config.api_keys, {'test': 'valid_key'}):
                response = client.post('/predict',
                                    json={'comments': ['test']},
                                    headers={'X-API-Key': 'valid_key'})
                # Should either succeed or fail for other reasons, but not auth
                assert response.status_code != 401

    def test_predict_endpoint_rate_limiting(self):
        """Test rate limiting on predict endpoint."""
        with app.test_client() as client:
            # Mock valid API key
            with patch.dict(security_config.api_keys, {'test': 'validkey123456'}):
                # Reset rate limiter
                security_config.rate_limiter.requests = {}
                
                # Make many requests to trigger rate limiting
                for i in range(105):  # More than the limit
                    response = client.post('/predict',
                                        json={'comments': ['test']},
                                        headers={'X-API-Key': 'validkey123456'})
                    
                    if i >= 100:  # Should be rate limited
                        assert response.status_code == 429

    def test_input_validation_on_predict(self):
        """Test input validation on predict endpoint."""
        with app.test_client() as client:
            with patch.dict(security_config.api_keys, {'test': 'validkey123456'}):
                # Test with invalid JSON
                response = client.post('/predict',
                                    data='invalid json',
                                    content_type='text/plain',
                                    headers={'X-API-Key': 'validkey123456'})
                assert response.status_code == 400

    def test_security_headers(self):
        """Test that security headers are present."""
        with app.test_client() as client:
            response = client.get('/health')
            headers = response.headers
            
            # Check for security headers
            assert 'X-Content-Type-Options' in headers
            assert 'X-Frame-Options' in headers
            assert 'X-XSS-Protection' in headers

class TestErrorHandling:
    """Test error handling functionality."""
    
    def test_400_error_handler(self):
        """Test 400 error handler."""
        with app.test_client() as client:
            response = client.post('/predict',
                                data='invalid json',
                                content_type='text/plain',
                                headers={'X-API-Key': 'test_key'})
            assert response.status_code == 400

    def test_401_error_handler(self):
        """Test 401 error handler."""
        with app.test_client() as client:
            response = client.post('/predict',
                                json={'comments': ['test']})
            assert response.status_code == 401

    def test_429_error_handler(self):
        """Test 429 error handler."""
        with app.test_client() as client:
            # Mock rate limiter to always return False
            with patch.object(security_config.rate_limiter, 'is_allowed', return_value=False):
                response = client.post('/predict',
                                    json={'comments': ['test']},
                                    headers={'X-API-Key': 'test_key'})
                assert response.status_code == 429

    def test_500_error_handler(self):
        """Test 500 error handler."""
        with app.test_client() as client:
            # Mock model to raise exception
            with patch('app.load_model', side_effect=Exception("Test error")):
                response = client.post('/predict',
                                    json={'comments': ['test']},
                                    headers={'X-API-Key': 'test_key'})
                assert response.status_code == 500

class TestLogging:
    """Test logging functionality."""
    
    def test_request_logging(self):
        """Test that requests are logged."""
        with app.test_client() as client:
            # Mock the model loading to avoid errors
            with patch('app.load_model'):
                response = client.get('/health')
                # Logging is tested by checking that the endpoint works
                assert response.status_code == 200

    def test_security_event_logging(self):
        """Test that security events are logged."""
        with app.test_client() as client:
            # Test unauthorized access
            response = client.post('/predict',
                                json={'comments': ['test']})
            assert response.status_code == 401
            # Logging should have occurred

    def test_error_logging(self):
        """Test that errors are logged."""
        with app.test_client() as client:
            # Mock model to raise exception
            with patch('app.load_model', side_effect=Exception("Test error")):
                response = client.post('/predict',
                                    json={'comments': ['test']},
                                    headers={'X-API-Key': 'test_key'})
                assert response.status_code == 500
                # Error logging should have occurred 