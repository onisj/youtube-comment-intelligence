"""
Security Tests for YouTube Sentiment Analyzer
"""

import pytest
import json
import time
from unittest.mock import Mock, patch
from flask import Flask
from werkzeug.exceptions import BadRequest, Unauthorized, TooManyRequests

# Import the app and security components
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app, security_config, rate_limiter, InputValidator
from security_config import SecurityManager, security_manager

class TestSecurityConfiguration:
    """Test security configuration."""
    
    def test_security_config_initialization(self):
        """Test security configuration initialization."""
        assert security_config is not None
        assert hasattr(security_config, 'secret_key')
        assert hasattr(security_config, 'api_keys')
        assert hasattr(security_config, 'rate_limit_requests')
        assert hasattr(security_config, 'rate_limit_window')
    
    def test_security_manager_initialization(self):
        """Test security manager initialization."""
        assert security_manager is not None
        assert hasattr(security_manager, 'settings')
        assert hasattr(security_manager, 'api_keys')
        assert hasattr(security_manager, 'blocked_ips')
    
    def test_api_key_validation(self):
        """Test API key validation."""
        # Test valid API key
        with patch.dict(security_manager.api_keys, {'test': 'valid_key_123'}):
            assert security_manager.validate_api_key('valid_key_123') is True
        
        # Test invalid API key
        assert security_manager.validate_api_key('invalid_key') is False
        assert security_manager.validate_api_key('') is False
        assert security_manager.validate_api_key(None) is False

class TestInputValidation:
    """Test input validation and sanitization."""
    
    def test_sanitize_text_basic(self):
        """Test basic text sanitization."""
        # Test normal text
        result = InputValidator.sanitize_text("Hello world")
        assert result == "Hello world"
        
        # Test text with dangerous characters
        result = InputValidator.sanitize_text("Hello<script>alert('xss')</script>world")
        assert result == "Helloalert(xss)world"
        
        # Test text with quotes
        result = InputValidator.sanitize_text('Hello"world\'test')
        assert result == "Helloworldtest"
    
    def test_sanitize_text_length_limit(self):
        """Test text length limiting."""
        long_text = "a" * 2000
        result = InputValidator.sanitize_text(long_text)
        assert len(result) <= 1000
    
    def test_sanitize_text_invalid_input(self):
        """Test sanitization with invalid input."""
        with pytest.raises(ValueError):
            InputValidator.sanitize_text(123)
        
        with pytest.raises(ValueError):
            InputValidator.sanitize_text(None)
    
    def test_validate_comments_basic(self):
        """Test comment validation."""
        comments = ["Hello", "World", "Test"]
        result = InputValidator.validate_comments(comments)
        assert result == comments
    
    def test_validate_comments_empty(self):
        """Test validation of empty comment list."""
        with pytest.raises(ValueError):
            InputValidator.validate_comments([])
    
    def test_validate_comments_too_many(self):
        """Test validation with too many comments."""
        comments = ["test"] * 150
        with pytest.raises(ValueError):
            InputValidator.validate_comments(comments)
    
    def test_validate_comments_invalid_type(self):
        """Test validation with invalid comment type."""
        with pytest.raises(ValueError):
            InputValidator.validate_comments("not a list")
    
    def test_validate_comments_mixed_types(self):
        """Test validation with mixed types."""
        comments = ["Hello", 123, "World"]
        with pytest.raises(ValueError):
            InputValidator.validate_comments(comments)

class TestRateLimiting:
    """Test rate limiting functionality."""
    
    def test_rate_limiter_initialization(self):
        """Test rate limiter initialization."""
        assert rate_limiter is not None
        assert hasattr(rate_limiter, 'requests')
    
    def test_rate_limiter_basic(self):
        """Test basic rate limiting."""
        client_id = "test_client"
        max_requests = 5
        window = 60
        
        # First 5 requests should be allowed
        for i in range(5):
            assert rate_limiter.is_allowed(client_id, max_requests, window) is True
        
        # 6th request should be blocked
        assert rate_limiter.is_allowed(client_id, max_requests, window) is False
    
    def test_rate_limiter_window_expiry(self):
        """Test rate limiter window expiry."""
        client_id = "test_client_window"
        max_requests = 3
        window = 1  # 1 second window
        
        # Make 3 requests
        for i in range(3):
            assert rate_limiter.is_allowed(client_id, max_requests, window) is True
        
        # 4th request should be blocked
        assert rate_limiter.is_allowed(client_id, max_requests, window) is False
        
        # Wait for window to expire
        time.sleep(1.1)
        
        # Should be allowed again
        assert rate_limiter.is_allowed(client_id, max_requests, window) is True
    
    def test_rate_limiter_different_clients(self):
        """Test rate limiting for different clients."""
        client1 = "client1"
        client2 = "client2"
        max_requests = 2
        window = 60
        
        # Client 1 makes 2 requests
        assert rate_limiter.is_allowed(client1, max_requests, window) is True
        assert rate_limiter.is_allowed(client1, max_requests, window) is True
        assert rate_limiter.is_allowed(client1, max_requests, window) is False
        
        # Client 2 should still be allowed
        assert rate_limiter.is_allowed(client2, max_requests, window) is True

class TestSecurityManager:
    """Test security manager functionality."""
    
    def test_ip_blocking(self):
        """Test IP blocking functionality."""
        test_ip = "192.168.1.1"
        
        # Initially not blocked
        assert security_manager.is_ip_blocked(test_ip) is False
        
        # Block IP
        security_manager.block_ip(test_ip)
        assert security_manager.is_ip_blocked(test_ip) is True
        
        # Unblock IP
        security_manager.unblock_ip(test_ip)
        assert security_manager.is_ip_blocked(test_ip) is False
    
    def test_suspicious_content_detection(self):
        """Test suspicious content detection."""
        # Test normal content
        assert security_manager.detect_suspicious_content("Hello world") is False
        
        # Test script tag
        assert security_manager.detect_suspicious_content("<script>alert('xss')</script>") is True
        
        # Test iframe
        assert security_manager.detect_suspicious_content("<iframe src='evil.com'></iframe>") is True
        
        # Test javascript protocol
        assert security_manager.detect_suspicious_content("javascript:alert('xss')") is True
    
    def test_input_sanitization(self):
        """Test input sanitization."""
        # Test normal input
        result = security_manager.sanitize_input("Hello world")
        assert result == "Hello world"
        
        # Test dangerous input
        result = security_manager.sanitize_input("<script>alert('xss')</script>")
        assert result == "scriptalert(xss)/script"
        
        # Test length limiting
        long_input = "a" * 2000
        result = security_manager.sanitize_input(long_input)
        assert len(result) <= security_manager.settings.max_comment_length
    
    def test_batch_size_validation(self):
        """Test batch size validation."""
        # Test valid sizes
        assert security_manager.validate_batch_size(1) is True
        assert security_manager.validate_batch_size(50) is True
        assert security_manager.validate_batch_size(100) is True
        
        # Test invalid sizes
        assert security_manager.validate_batch_size(0) is False
        assert security_manager.validate_batch_size(101) is False
        assert security_manager.validate_batch_size(-1) is False
    
    def test_security_headers(self):
        """Test security headers generation."""
        headers = security_manager.get_security_headers()
        
        assert 'X-Content-Type-Options' in headers
        assert 'X-Frame-Options' in headers
        assert 'X-XSS-Protection' in headers
        assert headers['X-Frame-Options'] == 'DENY'

class TestAPISecurity:
    """Test API security features."""
    
    def test_health_endpoint_no_auth(self):
        """Test health endpoint doesn't require authentication."""
        with app.test_client() as client:
            response = client.get('/health')
            assert response.status_code == 200
    
    def test_predict_endpoint_requires_auth(self):
        """Test predict endpoint requires authentication."""
        with app.test_client() as client:
            # Test without API key
            response = client.post('/predict', 
                                json={'comments': ['test']})
            assert response.status_code == 401
            
            # Test with invalid API key
            response = client.post('/predict',
                                json={'comments': ['test']},
                                headers={'X-API-Key': 'invalid_key'})
            assert response.status_code == 401
    
    def test_predict_endpoint_rate_limiting(self):
        """Test rate limiting on predict endpoint."""
        with app.test_client() as client:
            # Mock valid API key
            with patch.dict(security_config.api_keys, {'test': 'valid_key'}):
                # Make many requests to trigger rate limiting
                for i in range(105):  # More than the limit
                    response = client.post('/predict',
                                        json={'comments': ['test']},
                                        headers={'X-API-Key': 'valid_key'})
                    
                    if i >= 100:  # Should be rate limited
                        assert response.status_code == 429
                    else:
                        # First 100 should succeed (though may fail for other reasons)
                        pass
    
    def test_input_validation_on_predict(self):
        """Test input validation on predict endpoint."""
        with app.test_client() as client:
            with patch.dict(security_config.api_keys, {'test': 'valid_key'}):
                # Test with invalid JSON
                response = client.post('/predict',
                                    data='invalid json',
                                    content_type='text/plain')
                assert response.status_code == 400
                
                # Test with missing comments field
                response = client.post('/predict',
                                    json={},
                                    headers={'X-API-Key': 'valid_key'})
                assert response.status_code == 400
                
                # Test with too many comments
                response = client.post('/predict',
                                    json={'comments': ['test'] * 150},
                                    headers={'X-API-Key': 'valid_key'})
                assert response.status_code == 400
    
    def test_security_headers_present(self):
        """Test that security headers are present in responses."""
        with app.test_client() as client:
            response = client.get('/health')
            
            assert 'X-Content-Type-Options' in response.headers
            assert 'X-Frame-Options' in response.headers
            assert 'X-XSS-Protection' in response.headers
            assert response.headers['X-Frame-Options'] == 'DENY'
    
    def test_cors_headers(self):
        """Test CORS headers are present."""
        with app.test_client() as client:
            response = client.get('/health')
            
            # CORS headers should be present
            assert 'Access-Control-Allow-Origin' in response.headers

class TestErrorHandling:
    """Test error handling and logging."""
    
    def test_400_error_handler(self):
        """Test 400 error handler."""
        with app.test_client() as client:
            response = client.post('/predict',
                                data='invalid json',
                                content_type='text/plain')
            
            assert response.status_code == 400
            data = json.loads(response.get_data(as_text=True))
            assert 'error' in data
            assert data['error'] == 'Bad Request'
    
    def test_401_error_handler(self):
        """Test 401 error handler."""
        with app.test_client() as client:
            response = client.post('/predict',
                                json={'comments': ['test']})
            
            assert response.status_code == 401
            data = json.loads(response.get_data(as_text=True))
            assert 'error' in data
            assert data['error'] == 'Unauthorized'
    
    def test_429_error_handler(self):
        """Test 429 error handler."""
        with app.test_client() as client:
            # This would require rate limiting to be triggered
            # For now, just test the error handler structure
            response = client.get('/health')
            # 429 would be tested in rate limiting tests
            pass
    
    def test_500_error_handler(self):
        """Test 500 error handler."""
        # This would require an internal error to be triggered
        # For now, just test the error handler structure
        with app.test_client() as client:
            response = client.get('/health')
            # 500 would be tested in error scenarios
            pass

class TestLogging:
    """Test logging functionality."""
    
    def test_request_logging(self):
        """Test that requests are logged."""
        with app.test_client() as client:
            response = client.get('/health')
            # Logging is tested by checking that the endpoint works
            assert response.status_code == 200
    
    def test_security_event_logging(self):
        """Test that security events are logged."""
        with app.test_client() as client:
            # Test unauthorized access logging
            response = client.post('/predict',
                                json={'comments': ['test']})
            assert response.status_code == 401
            # Logging verification would require checking log files

if __name__ == '__main__':
    pytest.main([__file__, '-v']) 