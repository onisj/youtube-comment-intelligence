# 🔒 Security Guide

Comprehensive security implementation for the YouTube Sentiment Analyzer.

## 📋 Security Overview

This project implements enterprise-grade security features to protect against common web application vulnerabilities and ensure safe operation in production environments.

## 🛡️ Security Features

### 1. **Authentication & Authorization**

#### API Key Authentication
- **Header**: `X-API-Key`
- **Validation**: Server-side validation of API keys
- **Storage**: Environment variables (never in code)
- **Rotation**: Support for multiple API keys

```bash
# Example API request
curl -X POST http://localhost:8080/predict \
  -H "X-API-Key: your_api_key_here" \
  -H "Content-Type: application/json" \
  -d '{"comments": ["Hello world"]}'
```

#### Configuration
```bash
# Environment variables
YOUTUBE_API_KEY=your_youtube_api_key
CUSTOM_API_KEYS=key1,key2,key3
REQUIRE_API_KEY=True
```

### 2. **Rate Limiting**

#### Implementation
- **In-memory rate limiter** with configurable limits
- **Per-client tracking** using IP address or client ID
- **Sliding window** algorithm for accurate limiting
- **Configurable limits** via environment variables

#### Configuration
```bash
# Rate limiting settings
RATE_LIMIT_ENABLED=True
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=3600  # 1 hour
```

#### Limits by Endpoint
| Endpoint | Default Limit | Window |
|----------|---------------|---------|
| `/predict` | 100 requests | 1 hour |
| `/predict_with_timestamps` | 100 requests | 1 hour |
| `/generate_chart` | 100 requests | 1 hour |
| `/generate_wordcloud` | 100 requests | 1 hour |
| `/generate_trend_graph` | 100 requests | 1 hour |

### 3. **Input Validation & Sanitization**

#### Text Sanitization
- **XSS Prevention**: Remove dangerous HTML tags
- **Length Limiting**: Maximum 1000 characters per comment
- **Character Filtering**: Remove potentially dangerous characters
- **Type Validation**: Ensure proper data types

#### Input Validation Rules
```python
# Comment validation
- Type: String only
- Length: 1-1000 characters
- Content: No HTML/script tags
- Batch size: Maximum 100 comments

# API request validation
- Content-Type: application/json
- Required fields: 'comments' array
- Data types: Proper JSON structure
```

#### Sanitization Examples
```python
# Before sanitization
"Hello<script>alert('xss')</script>world"

# After sanitization
"Helloalert(xss)world"
```

### 4. **Security Headers**

#### Implemented Headers
```http
X-Content-Type-Options: nosniff
X-Frame-Options: DENY
X-XSS-Protection: 1; mode=block
Strict-Transport-Security: max-age=31536000; includeSubDomains
Content-Security-Policy: default-src 'self'
```

#### Header Purposes
- **X-Content-Type-Options**: Prevents MIME type sniffing
- **X-Frame-Options**: Prevents clickjacking attacks
- **X-XSS-Protection**: Enables browser XSS filtering
- **HSTS**: Forces HTTPS connections
- **CSP**: Controls resource loading

### 5. **CORS Configuration**

#### Allowed Origins
```python
# Default allowed origins
[
    "http://localhost:3000",
    "http://localhost:8501",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:8501"
]
```

#### Configuration
```bash
# Environment variable
CORS_ORIGINS='["http://localhost:3000", "https://yourdomain.com"]'
```

### 6. **Error Handling & Logging**

#### Comprehensive Logging
- **Request logging**: All incoming requests
- **Response logging**: All outgoing responses
- **Security events**: Authentication failures, rate limit violations
- **Error logging**: Detailed error information
- **Performance logging**: Response times

#### Log Format
```
2024-01-15 10:30:45 - app - INFO - Request: POST /predict from 192.168.1.1
2024-01-15 10:30:45 - app - INFO - Response: 200 in 0.123s
2024-01-15 10:30:46 - app - WARNING - Rate limit exceeded for client: 192.168.1.1
```

#### Log Rotation
- **File size**: 10MB maximum
- **Backup count**: 5 files
- **Location**: `logs/app.log`

### 7. **Model Security**

#### Confidence Thresholds
- **Default threshold**: 0.6 (60%)
- **Configurable**: Via environment variable
- **Purpose**: Prevent low-confidence predictions

#### Configuration
```bash
PREDICTION_CONFIDENCE_THRESHOLD=0.6
MAX_COMMENT_LENGTH=1000
MAX_BATCH_SIZE=100
```

### 8. **IP Blocking**

#### Dynamic IP Management
- **Block suspicious IPs**: Automatic or manual blocking
- **Unblock IPs**: Manual unblocking capability
- **Persistence**: In-memory storage (configurable for Redis)

#### Usage
```python
# Block an IP
security_manager.block_ip("192.168.1.100")

# Unblock an IP
security_manager.unblock_ip("192.168.1.100")

# Check if IP is blocked
is_blocked = security_manager.is_ip_blocked("192.168.1.100")
```

## 🔧 Security Configuration

### Environment Variables

#### Required Variables
```bash
# API Keys
YOUTUBE_API_KEY=your_youtube_api_key_here
SECRET_KEY=your_secret_key_here

# Rate Limiting
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=3600

# Input Validation
MAX_COMMENT_LENGTH=1000
MAX_BATCH_SIZE=100
PREDICTION_CONFIDENCE_THRESHOLD=0.6
```

#### Optional Variables
```bash
# Security Features
REQUIRE_API_KEY=True
RATE_LIMIT_ENABLED=True
ENABLE_SECURITY_HEADERS=True
LOG_SECURITY_EVENTS=True

# CORS
CORS_ORIGINS='["http://localhost:3000", "https://yourdomain.com"]'

# Custom API Keys
CUSTOM_API_KEYS=key1,key2,key3
```

### Security Manager Configuration

#### Settings Class
```python
@dataclass
class SecuritySettings:
    require_api_key: bool = True
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 100
    rate_limit_window: int = 3600
    max_comment_length: int = 1000
    max_batch_size: int = 100
    prediction_confidence_threshold: float = 0.6
    enable_security_headers: bool = True
    log_security_events: bool = True
```

## 🧪 Security Testing

### Test Coverage
- **Authentication tests**: API key validation
- **Rate limiting tests**: Request limiting functionality
- **Input validation tests**: Sanitization and validation
- **Error handling tests**: Proper error responses
- **Security header tests**: Header presence and values

### Running Security Tests
```bash
# Run all security tests
python -m pytest tests/test_security.py -v

# Run specific security test categories
python -m pytest tests/test_security.py::TestAuthentication -v
python -m pytest tests/test_security.py::TestRateLimiting -v
python -m pytest tests/test_security.py::TestInputValidation -v
```

### Security Test Examples
```python
def test_api_key_validation():
    """Test API key validation."""
    assert security_manager.validate_api_key('valid_key') is True
    assert security_manager.validate_api_key('invalid_key') is False

def test_rate_limiting():
    """Test rate limiting functionality."""
    client_id = "test_client"
    for i in range(5):
        assert rate_limiter.is_allowed(client_id, 5, 60) is True
    assert rate_limiter.is_allowed(client_id, 5, 60) is False

def test_input_sanitization():
    """Test input sanitization."""
    result = security_manager.sanitize_input("<script>alert('xss')</script>")
    assert "<script>" not in result
```

## 🚨 Security Best Practices

### 1. **API Key Management**
- ✅ **Use strong keys**: Generate cryptographically secure keys
- ✅ **Rotate regularly**: Change keys periodically
- ✅ **Store securely**: Use environment variables, never in code
- ✅ **Limit access**: Use different keys for different environments

### 2. **Rate Limiting**
- ✅ **Monitor usage**: Track rate limit violations
- ✅ **Adjust limits**: Tune based on legitimate usage patterns
- ✅ **Graceful handling**: Return proper 429 responses
- ✅ **Client identification**: Use consistent client identification

### 3. **Input Validation**
- ✅ **Validate early**: Check input as soon as possible
- ✅ **Sanitize thoroughly**: Remove all dangerous content
- ✅ **Type checking**: Ensure proper data types
- ✅ **Length limits**: Prevent oversized inputs

### 4. **Error Handling**
- ✅ **Don't expose internals**: Never reveal system details
- ✅ **Log security events**: Track all security-related activities
- ✅ **Consistent responses**: Use standard error formats
- ✅ **Graceful degradation**: Handle errors without crashing

### 5. **Monitoring & Logging**
- ✅ **Comprehensive logging**: Log all security events
- ✅ **Monitor patterns**: Watch for suspicious activity
- ✅ **Alert on violations**: Set up alerts for security events
- ✅ **Regular reviews**: Review logs periodically

## 🔍 Security Monitoring

### Key Metrics to Monitor
- **Authentication failures**: Track failed login attempts
- **Rate limit violations**: Monitor abuse patterns
- **Input validation errors**: Watch for malicious inputs
- **Response times**: Detect performance issues
- **Error rates**: Monitor system health

### Log Analysis
```bash
# Check for authentication failures
grep "Unauthorized" logs/app.log

# Check for rate limit violations
grep "Rate limit exceeded" logs/app.log

# Check for suspicious inputs
grep "suspicious content" logs/app.log
```

### Security Alerts
```python
# Example alert conditions
- More than 10 authentication failures per minute
- Rate limit violations from same IP
- Suspicious content detection
- Unusual response time patterns
```

## 🚀 Production Deployment

### Security Checklist
- [ ] **HTTPS enabled**: Use SSL/TLS certificates
- [ ] **API keys configured**: Set up proper authentication
- [ ] **Rate limits tuned**: Adjust based on usage
- [ ] **Logging configured**: Ensure security events are logged
- [ ] **Monitoring active**: Set up security monitoring
- [ ] **Backups secure**: Protect backup data
- [ ] **Updates regular**: Keep dependencies updated

### Environment Setup
```bash
# Production environment variables
export FLASK_ENV=production
export DEBUG=False
export SECRET_KEY=$(python -c "import secrets; print(secrets.token_hex(32))")
export YOUTUBE_API_KEY=your_production_key
export RATE_LIMIT_REQUESTS=50
export LOG_LEVEL=WARNING
```

### Security Headers Verification
```bash
# Test security headers
curl -I http://localhost:8080/health

# Expected headers
X-Content-Type-Options: nosniff
X-Frame-Options: DENY
X-XSS-Protection: 1; mode=block
Strict-Transport-Security: max-age=31536000; includeSubDomains
```

## 📚 Additional Resources

### Security Documentation
- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [Flask Security](https://flask-security.readthedocs.io/)
- [Python Security](https://python-security.readthedocs.io/)

### Security Tools
- **Bandit**: Python security linter
- **Safety**: Dependency vulnerability checker
- **Snyk**: Security scanning
- **OWASP ZAP**: Web application security scanner

### Security Standards
- **OWASP ASVS**: Application Security Verification Standard
- **NIST Cybersecurity Framework**: Security best practices
- **ISO 27001**: Information security management

---

**🔒 Security is a continuous process. Regularly review and update security measures based on new threats and best practices.** 