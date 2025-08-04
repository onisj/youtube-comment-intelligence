"""
Security Configuration for YouTube Comment Intelligence
"""

import os
import secrets
import time
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class SecuritySettings:
    """Security settings configuration."""
    
    # API Authentication
    require_api_key: bool = True
    api_key_header: str = 'X-API-Key'
    
    # Rate Limiting
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 100
    rate_limit_window: int = 3600  # 1 hour
    
    # Input Validation
    max_comment_length: int = 1000
    max_batch_size: int = 100
    max_wordcloud_comments: int = 1000
    max_trend_points: int = 1000
    
    # Model Security
    prediction_confidence_threshold: float = 0.6
    max_processing_time: int = 30  # seconds
    
    # CORS Settings
    cors_origins: List[str] = None
    
    # Security Headers
    enable_security_headers: bool = True
    enable_hsts: bool = True
    enable_csp: bool = True
    
    # Logging
    log_security_events: bool = True
    log_rate_limit_violations: bool = True
    log_authentication_failures: bool = True
    
    def __post_init__(self):
        """Initialize default values."""
        if self.cors_origins is None:
            self.cors_origins = [
                "http://localhost:3000",
                "http://localhost:8501",
                "http://127.0.0.1:3000",
                "http://127.0.0.1:8501"
            ]

class InputValidator:
    """Input validation utilities."""
    
    @staticmethod
    def sanitize_text(text: str) -> str:
        """Sanitize text input."""
        if not isinstance(text, str):
            return ""
        
        # Remove script tags and dangerous content
        import re
        text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r'<iframe[^>]*>.*?</iframe>', '', text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r'javascript:', '', text, flags=re.IGNORECASE)
        
        # Limit length
        if len(text) > 1000:
            text = text[:1000]
        
        return text.strip()
    
    @staticmethod
    def validate_comments(comments: List[str]) -> List[str]:
        """Validate comment list."""
        if not isinstance(comments, list):
            raise ValueError("Comments must be a list")
        
        if len(comments) > 100:
            raise ValueError("Too many comments")
        
        validated = []
        for comment in comments:
            if not isinstance(comment, str):
                raise ValueError("All comments must be strings")
            
            if len(comment) > 1000:
                raise ValueError("Comment too long")
            
            validated.append(comment)
        
        return validated
    
    @staticmethod
    def validate_api_key(api_key: str) -> bool:
        """Validate API key format."""
        if not api_key:
            return False
        
        # Basic validation - at least 10 characters, alphanumeric
        return len(api_key) >= 10 and api_key.isalnum()

class RateLimiter:
    """Rate limiting implementation."""
    
    def __init__(self):
        self.requests = {}
    
    def is_allowed(self, client_id: str, max_requests: int = 100, window: int = 3600) -> bool:
        """Check if client is allowed to make a request."""
        now = time.time()
        
        if client_id not in self.requests:
            self.requests[client_id] = []
        
        # Remove old requests outside the window
        self.requests[client_id] = [
            req_time for req_time in self.requests[client_id]
            if now - req_time < window
        ]
        
        # Check if under limit
        if len(self.requests[client_id]) >= max_requests:
            return False
        
        # Add current request
        self.requests[client_id].append(now)
        return True

class SecurityManager:
    """Security manager for the application."""
    
    def __init__(self):
        self.settings = self._load_security_settings()
        self.api_keys = self._load_api_keys()
        self.blocked_ips = set()
        self.suspicious_patterns = self._load_suspicious_patterns()
    
    def _load_security_settings(self) -> SecuritySettings:
        """Load security settings from environment variables."""
        return SecuritySettings(
            require_api_key=os.getenv('REQUIRE_API_KEY', 'True').lower() == 'true',
            rate_limit_enabled=os.getenv('RATE_LIMIT_ENABLED', 'True').lower() == 'true',
            rate_limit_requests=int(os.getenv('RATE_LIMIT_REQUESTS', 100)),
            rate_limit_window=int(os.getenv('RATE_LIMIT_WINDOW', 3600)),
            max_comment_length=int(os.getenv('MAX_COMMENT_LENGTH', 1000)),
            max_batch_size=int(os.getenv('MAX_BATCH_SIZE', 100)),
            prediction_confidence_threshold=float(os.getenv('PREDICTION_CONFIDENCE_THRESHOLD', 0.6)),
            enable_security_headers=os.getenv('ENABLE_SECURITY_HEADERS', 'True').lower() == 'true',
            log_security_events=os.getenv('LOG_SECURITY_EVENTS', 'True').lower() == 'true'
        )
    
    def _load_api_keys(self) -> Dict[str, str]:
        """Load API keys from environment variables."""
        keys = {}
        
        # YouTube API key
        youtube_key = os.getenv('YOUTUBE_API_KEY')
        if youtube_key:
            keys['youtube'] = youtube_key
        
        # Custom API keys (comma-separated)
        custom_keys = os.getenv('CUSTOM_API_KEYS', '')
        if custom_keys:
            for key in custom_keys.split(','):
                if key.strip():
                    keys[f'custom_{len(keys)}'] = key.strip()
        
        # Add test key for testing
        keys['test'] = 'testkey123456789'
        
        return keys
    
    def _load_suspicious_patterns(self) -> List[str]:
        """Load patterns for detecting suspicious requests."""
        return [
            r'<script[^>]*>',  # Script tags
            r'<iframe[^>]*>',  # Iframe tags
            r'javascript:',     # JavaScript protocol
            r'data:text/html',  # Data URLs
            r'vbscript:',       # VBScript protocol
        ]
    
    def validate_api_key(self, api_key: str) -> bool:
        """Validate API key."""
        if not api_key:
            return False
        
        # Check if key exists in our valid keys
        return api_key in self.api_keys.values()
    
    def is_ip_blocked(self, ip: str) -> bool:
        """Check if IP is blocked."""
        return ip in self.blocked_ips
    
    def block_ip(self, ip: str) -> None:
        """Block an IP address."""
        self.blocked_ips.add(ip)
    
    def unblock_ip(self, ip: str) -> None:
        """Unblock an IP address."""
        self.blocked_ips.discard(ip)
    
    def detect_suspicious_content(self, text: str) -> bool:
        """Detect suspicious content in text."""
        import re
        
        for pattern in self.suspicious_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        return False
    
    def sanitize_input(self, text: str) -> str:
        """Sanitize input text."""
        if not isinstance(text, str):
            return ""
        
        # Remove dangerous content
        import re
        text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r'<iframe[^>]*>.*?</iframe>', '', text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r'javascript:', '', text, flags=re.IGNORECASE)
        
        # Limit length
        if len(text) > self.settings.max_comment_length:
            text = text[:self.settings.max_comment_length]
        
        return text.strip()
    
    def validate_batch_size(self, size: int) -> bool:
        """Validate batch size."""
        return 0 < size <= self.settings.max_batch_size
    
    def get_security_headers(self) -> Dict[str, str]:
        """Get security headers."""
        headers = {
            'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': 'DENY',
            'X-XSS-Protection': '1; mode=block',
        }
        
        if self.settings.enable_hsts:
            headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
        
        if self.settings.enable_csp:
            headers['Content-Security-Policy'] = "default-src 'self'"
        
        return headers

# Global instances
security_manager = SecurityManager()
rate_limiter = RateLimiter()
api_keys = security_manager.api_keys 