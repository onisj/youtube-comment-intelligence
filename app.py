import os
import re
import secrets
import time
from datetime import datetime, timedelta
from functools import wraps
from typing import Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd
from flask import (
    Flask, jsonify, request, make_response, 
    abort, g, current_app
)
from flask_cors import CORS
from werkzeug.exceptions import BadRequest, Unauthorized, TooManyRequests
import logging
from logging.handlers import RotatingFileHandler
import hashlib
import hmac
import base64

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configure logging
def setup_logging():
    """Setup comprehensive logging configuration."""
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # File handler with rotation
    file_handler = RotatingFileHandler(
        'logs/app.log', 
        maxBytes=10240000,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        handlers=[file_handler, console_handler]
    )
    
    return logging.getLogger(__name__)

logger = setup_logging()

# Security configuration
class SecurityConfig:
    """Security configuration class."""
    
    def __init__(self):
        self.secret_key = os.getenv('SECRET_KEY', secrets.token_hex(32))
        self.api_keys = self._load_api_keys()
        self.rate_limit_requests = int(os.getenv('RATE_LIMIT_REQUESTS', 100))
        self.rate_limit_window = int(os.getenv('RATE_LIMIT_WINDOW', 3600))
        self.max_comment_length = int(os.getenv('MAX_COMMENT_LENGTH', 1000))
        self.prediction_confidence_threshold = float(os.getenv('PREDICTION_CONFIDENCE_THRESHOLD', 0.6))
        
    def _load_api_keys(self) -> Dict[str, str]:
        """Load API keys from environment variables."""
        keys = {}
        api_key = os.getenv('YOUTUBE_API_KEY')
        if api_key:
            keys['youtube'] = api_key
        return keys

# Rate limiting storage
class RateLimiter:
    """Simple in-memory rate limiter."""
    
    def __init__(self):
        self.requests = {}
    
    def is_allowed(self, client_id: str, max_requests: int, window: int) -> bool:
        """Check if request is allowed based on rate limits."""
        now = time.time()
        
        # Clean old entries
        self.requests = {
            k: v for k, v in self.requests.items() 
            if now - v['timestamp'] < window
        }
        
        if client_id not in self.requests:
            self.requests[client_id] = {
                'count': 1,
                'timestamp': now
            }
            return True
        
        if self.requests[client_id]['count'] >= max_requests:
            return False
        
        self.requests[client_id]['count'] += 1
        return True

# Input validation and sanitization
class InputValidator:
    """Input validation and sanitization utilities."""
    
    @staticmethod
    def sanitize_text(text: str) -> str:
        """Sanitize text input to prevent XSS and injection attacks."""
        if not isinstance(text, str):
            raise ValueError("Input must be a string")
        
        # Remove potentially dangerous characters
        text = re.sub(r'[<>"\']', '', text)
        
        # Limit length
        if len(text) > 1000:
            text = text[:1000]
        
        return text.strip()
    
    @staticmethod
    def validate_comments(comments: List[str]) -> List[str]:
        """Validate and sanitize comment list."""
        if not isinstance(comments, list):
            raise ValueError("Comments must be a list")
        
        if len(comments) > 100:  # Limit batch size
            raise ValueError("Too many comments (max 100)")
        
        sanitized_comments = []
        for comment in comments:
            if not isinstance(comment, str):
                raise ValueError("Each comment must be a string")
            
            sanitized = InputValidator.sanitize_text(comment)
            if sanitized:  # Only add non-empty comments
                sanitized_comments.append(sanitized)
        
        return sanitized_comments
    
    @staticmethod
    def validate_api_key(api_key: str) -> bool:
        """Validate API key format."""
        if not api_key:
            return False
        
        # Basic validation - you can add more sophisticated checks
        return len(api_key) >= 10 and api_key.isalnum()

# Authentication decorator
def require_api_key(f):
    """Decorator to require API key authentication."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        
        if not api_key:
            logger.warning("Missing API key in request")
            abort(401, description="API key required")
        
        if not InputValidator.validate_api_key(api_key):
            logger.warning("Invalid API key format")
            abort(401, description="Invalid API key")
        
        # In production, you'd validate against a database
        # For now, we'll use a simple check
        valid_keys = current_app.config['SECURITY_CONFIG'].api_keys.values()
        if api_key not in valid_keys:
            logger.warning(f"Invalid API key: {api_key[:10]}...")
            abort(401, description="Invalid API key")
        
        return f(*args, **kwargs)
    return decorated_function

# Rate limiting decorator
def rate_limit(f):
    """Decorator to implement rate limiting."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        client_id = request.headers.get('X-Client-ID', request.remote_addr)
        
        if not current_app.config['RATE_LIMITER'].is_allowed(
            client_id,
            current_app.config['SECURITY_CONFIG'].rate_limit_requests,
            current_app.config['SECURITY_CONFIG'].rate_limit_window
        ):
            logger.warning(f"Rate limit exceeded for client: {client_id}")
            abort(429, description="Rate limit exceeded")
        
        return f(*args, **kwargs)
    return decorated_function

# Create Flask app with security features
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', secrets.token_hex(32))

# Security configuration
security_config = SecurityConfig()
app.config['SECURITY_CONFIG'] = security_config

# Rate limiter
rate_limiter = RateLimiter()
app.config['RATE_LIMITER'] = rate_limiter

# CORS configuration
cors_origins = os.getenv('CORS_ORIGINS', '["http://localhost:3000", "http://localhost:8501"]')
try:
    cors_origins = eval(cors_origins)  # Convert string to list
except:
    cors_origins = ["http://localhost:3000", "http://localhost:8501"]

CORS(app, origins=cors_origins, supports_credentials=True)

# Custom exceptions
class ValidationError(Exception):
    """Custom validation error."""
    pass

class SecurityError(Exception):
    """Custom security error."""
    pass

# Error handlers
@app.errorhandler(400)
def bad_request(error):
    """Handle bad request errors."""
    logger.warning(f"Bad request: {error}")
    return jsonify({
        'error': 'Bad Request',
        'message': str(error),
        'timestamp': datetime.utcnow().isoformat()
    }), 400

@app.errorhandler(401)
def unauthorized(error):
    """Handle unauthorized errors."""
    logger.warning(f"Unauthorized access: {error}")
    return jsonify({
        'error': 'Unauthorized',
        'message': 'Authentication required',
        'timestamp': datetime.utcnow().isoformat()
    }), 401

@app.errorhandler(429)
def too_many_requests(error):
    """Handle rate limit errors."""
    logger.warning(f"Rate limit exceeded: {error}")
    return jsonify({
        'error': 'Too Many Requests',
        'message': 'Rate limit exceeded. Please try again later.',
        'timestamp': datetime.utcnow().isoformat()
    }), 429

@app.errorhandler(500)
def internal_error(error):
    """Handle internal server errors."""
    logger.error(f"Internal server error: {error}")
    return jsonify({
        'error': 'Internal Server Error',
        'message': 'An unexpected error occurred',
        'timestamp': datetime.utcnow().isoformat()
    }), 500

# Security headers middleware
@app.after_request
def add_security_headers(response):
    """Add security headers to all responses."""
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    response.headers['Content-Security-Policy'] = "default-src 'self'"
    return response

# Request logging middleware
@app.before_request
def log_request():
    """Log all incoming requests."""
    g.start_time = time.time()
    logger.info(f"Request: {request.method} {request.path} from {request.remote_addr}")

@app.after_request
def log_response(response):
    """Log all outgoing responses."""
    if hasattr(g, 'start_time'):
        duration = time.time() - g.start_time
        logger.info(f"Response: {response.status_code} in {duration:.3f}s")
    return response

# Model loading with error handling
def load_model():
    """Load the trained model with comprehensive error handling."""
    try:
        model_path = os.getenv('MODEL_PATH', 'models/lgbm_model.pkl')
        vectorizer_path = os.getenv('VECTORIZER_PATH', 'models/tfidf_vectorizer.pkl')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        if not os.path.exists(vectorizer_path):
            raise FileNotFoundError(f"Vectorizer file not found: {vectorizer_path}")
        
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        
        logger.info("Model and vectorizer loaded successfully")
        return model, vectorizer
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

# Text preprocessing with security
def preprocess_comment(text: str) -> str:
    """Preprocess comment text with security considerations."""
    try:
        # Input validation
        if not isinstance(text, str):
            raise ValueError("Input must be a string")
        
        # Sanitize input
        text = InputValidator.sanitize_text(text)
        
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace and newlines
        text = re.sub(r'\s+', ' ', text)
        
        # Remove numbers and non-alphanumeric characters
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        
        # Remove stopwords
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer
        
        try:
            stop_words = set(stopwords.words('english'))
            lemmatizer = WordNetLemmatizer()
        except LookupError:
            # Download NLTK data if not available
            import nltk
            nltk.download('stopwords')
            nltk.download('wordnet')
            stop_words = set(stopwords.words('english'))
            lemmatizer = WordNetLemmatizer()
        
        # Additional stopwords for sentiment analysis
        additional_stopwords = {
            'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'can', 'must'
        }
        stop_words.update(additional_stopwords)
        
        # Tokenize and process
        words = text.split()
        processed_words = []
        
        for word in words:
            if word not in stop_words and len(word) > 2:
                lemmatized = lemmatizer.lemmatize(word)
                processed_words.append(lemmatized)
        
        return ' '.join(processed_words)
        
    except Exception as e:
        logger.error(f"Error preprocessing text: {str(e)}")
        raise

# API endpoints with security
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    try:
        # Check if model is loaded
        model, vectorizer = load_model()
        
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'model_loaded': True,
            'version': '1.0.0'
        }), 200
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({
            'status': 'unhealthy',
            'timestamp': datetime.utcnow().isoformat(),
            'error': str(e)
        }), 500

@app.route('/predict', methods=['POST'])
@require_api_key
@rate_limit
def predict():
    """Predict sentiment for comments with comprehensive security."""
    try:
        # Validate content type
        if not request.is_json:
            raise BadRequest("Content-Type must be application/json")
        
        # Get and validate input
        data = request.get_json()
        if not data or 'comments' not in data:
            raise BadRequest("Missing 'comments' field in request body")
        
        comments = data['comments']
        
        # Validate and sanitize input
        try:
            sanitized_comments = InputValidator.validate_comments(comments)
        except ValueError as e:
            raise BadRequest(f"Invalid input: {str(e)}")
        
        if not sanitized_comments:
            raise BadRequest("No valid comments provided")
        
        # Load model
        try:
            model, vectorizer = load_model()
        except Exception as e:
            logger.error(f"Model loading error: {str(e)}")
            raise SecurityError("Model unavailable")
        
        # Process comments
        results = []
        for comment in sanitized_comments:
            try:
                # Preprocess
                processed_text = preprocess_comment(comment)
                
                if not processed_text:
                    results.append({
                        'comment': comment,
                        'sentiment': None,
                        'confidence': 0.0,
                        'processed_text': '',
                        'error': 'Empty text after preprocessing'
                    })
                    continue
                
                # Vectorize
                features = vectorizer.transform([processed_text])
                
                # Predict
                prediction = model.predict(features)[0]
                confidence = np.max(model.predict_proba(features)) if hasattr(model, 'predict_proba') else 0.5
                
                # Apply confidence threshold
                if confidence < security_config.prediction_confidence_threshold:
                    prediction = None
                
                results.append({
                    'comment': comment,
                    'sentiment': int(prediction) if prediction is not None else None,
                    'confidence': float(confidence),
                    'processed_text': processed_text,
                    'error': None
                })
                
            except Exception as e:
                logger.error(f"Error processing comment: {str(e)}")
                results.append({
                    'comment': comment,
                    'sentiment': None,
                    'confidence': 0.0,
                    'processed_text': '',
                    'error': 'Processing error'
                })
        
        # Log successful prediction
        logger.info(f"Successfully processed {len(sanitized_comments)} comments")
        
        return jsonify({
            'results': results,
            'timestamp': datetime.utcnow().isoformat(),
            'total_comments': len(sanitized_comments),
            'successful_predictions': len([r for r in results if r['error'] is None])
        }), 200
        
    except BadRequest as e:
        logger.warning(f"Bad request in predict endpoint: {str(e)}")
        raise
    except SecurityError as e:
        logger.error(f"Security error in predict endpoint: {str(e)}")
        abort(503, description="Service temporarily unavailable")
    except Exception as e:
        logger.error(f"Unexpected error in predict endpoint: {str(e)}")
        abort(500, description="Internal server error")

@app.route('/predict_with_timestamps', methods=['POST'])
@require_api_key
@rate_limit
def predict_with_timestamps():
    """Predict sentiment with timestamps."""
    try:
        if not request.is_json:
            raise BadRequest("Content-Type must be application/json")
        
        data = request.get_json()
        if not data or 'data' not in data:
            raise BadRequest("Missing 'data' field in request body")
        
        # Validate input data
        input_data = data['data']
        if not isinstance(input_data, list):
            raise BadRequest("Data must be a list")
        
        if len(input_data) > 100:
            raise BadRequest("Too many data points (max 100)")
        
        # Process each data point
        results = []
        for item in input_data:
            try:
                if not isinstance(item, dict) or 'comment' not in item:
                    continue
                
                comment = InputValidator.sanitize_text(item['comment'])
                timestamp = item.get('timestamp', datetime.utcnow().isoformat())
                
                if not comment:
                    continue
                
                # Get prediction
                model, vectorizer = load_model()
                processed_text = preprocess_comment(comment)
                
                if not processed_text:
                    continue
                
                features = vectorizer.transform([processed_text])
                prediction = model.predict(features)[0]
                confidence = np.max(model.predict_proba(features)) if hasattr(model, 'predict_proba') else 0.5
                
                results.append({
                    'comment': comment,
                    'sentiment': int(prediction),
                    'confidence': float(confidence),
                    'timestamp': timestamp,
                    'processed_text': processed_text
                })
                
            except Exception as e:
                logger.error(f"Error processing timestamped data: {str(e)}")
                continue
        
        return jsonify({
            'results': results,
            'timestamp': datetime.utcnow().isoformat(),
            'total_processed': len(results)
        }), 200
        
    except BadRequest as e:
        logger.warning(f"Bad request in timestamp endpoint: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error in timestamp endpoint: {str(e)}")
        abort(500, description="Internal server error")

@app.route('/generate_chart', methods=['POST'])
@require_api_key
@rate_limit
def generate_chart():
    """Generate sentiment distribution chart."""
    try:
        if not request.is_json:
            raise BadRequest("Content-Type must be application/json")
        
        data = request.get_json()
        if not data or 'sentiment_counts' not in data:
            raise BadRequest("Missing 'sentiment_counts' field")
        
        counts = data['sentiment_counts']
        if not isinstance(counts, dict):
            raise BadRequest("sentiment_counts must be a dictionary")
        
        # Validate counts
        total = sum(counts.values())
        if total == 0:
            raise BadRequest("No data provided for chart generation")
        
        # Calculate percentages
        percentages = {k: (v / total) * 100 for k, v in counts.items()}
        
        return jsonify({
            'percentages': percentages,
            'total': total,
            'timestamp': datetime.utcnow().isoformat()
        }), 200
        
    except BadRequest as e:
        logger.warning(f"Bad request in chart endpoint: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error in chart endpoint: {str(e)}")
        abort(500, description="Internal server error")

@app.route('/generate_wordcloud', methods=['POST'])
@require_api_key
@rate_limit
def generate_wordcloud():
    """Generate word cloud data."""
    try:
        if not request.is_json:
            raise BadRequest("Content-Type must be application/json")
        
        data = request.get_json()
        if not data or 'comments' not in data:
            raise BadRequest("Missing 'comments' field")
        
        comments = data['comments']
        if not isinstance(comments, list):
            raise BadRequest("comments must be a list")
        
        if len(comments) > 1000:
            raise BadRequest("Too many comments for word cloud (max 1000)")
        
        # Process comments for word cloud
        all_text = ' '.join([InputValidator.sanitize_text(comment) for comment in comments])
        processed_text = preprocess_comment(all_text)
        
        # Simple word frequency (in production, use proper word cloud library)
        words = processed_text.split()
        word_freq = {}
        for word in words:
            if len(word) > 2:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Get top words
        top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:50]
        
        return jsonify({
            'word_frequencies': dict(top_words),
            'total_words': len(words),
            'unique_words': len(word_freq),
            'timestamp': datetime.utcnow().isoformat()
        }), 200
        
    except BadRequest as e:
        logger.warning(f"Bad request in wordcloud endpoint: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error in wordcloud endpoint: {str(e)}")
        abort(500, description="Internal server error")

@app.route('/generate_trend_graph', methods=['POST'])
@require_api_key
@rate_limit
def generate_trend_graph():
    """Generate trend graph data."""
    try:
        if not request.is_json:
            raise BadRequest("Content-Type must be application/json")
        
        data = request.get_json()
        if not data or 'sentiment_data' not in data:
            raise BadRequest("Missing 'sentiment_data' field")
        
        sentiment_data = data['sentiment_data']
        if not isinstance(sentiment_data, list):
            raise BadRequest("sentiment_data must be a list")
        
        if len(sentiment_data) > 1000:
            raise BadRequest("Too much data for trend graph (max 1000 points)")
        
        # Process sentiment data
        df = pd.DataFrame(sentiment_data)
        
        if df.empty:
            raise BadRequest("No valid data provided")
        
        # Calculate trends
        if 'timestamp' in df.columns and 'sentiment' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
            
            # Calculate moving average
            window_size = min(10, len(df))
            if window_size > 0:
                df['moving_average'] = df['sentiment'].rolling(window=window_size).mean()
            
            # Prepare response
            trend_data = {
                'timestamps': df['timestamp'].dt.isoformat().tolist(),
                'sentiments': df['sentiment'].tolist(),
                'moving_average': df.get('moving_average', df['sentiment']).tolist(),
                'total_points': len(df)
            }
        else:
            raise BadRequest("Missing required columns: timestamp, sentiment")
        
        return jsonify({
            'trend_data': trend_data,
            'timestamp': datetime.utcnow().isoformat()
        }), 200
        
    except BadRequest as e:
        logger.warning(f"Bad request in trend endpoint: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error in trend endpoint: {str(e)}")
        abort(500, description="Internal server error")

@app.route('/docs', methods=['GET'])
def api_docs():
    """API documentation endpoint."""
    return jsonify({
        'title': 'YouTube Sentiment Analyzer API',
        'version': '1.0.0',
        'description': 'Secure API for YouTube comment sentiment analysis',
        'endpoints': {
            '/health': {
                'method': 'GET',
                'description': 'Health check endpoint',
                'authentication': 'None'
            },
            '/predict': {
                'method': 'POST',
                'description': 'Predict sentiment for comments',
                'authentication': 'API Key required',
                'rate_limit': 'Yes'
            },
            '/predict_with_timestamps': {
                'method': 'POST',
                'description': 'Predict sentiment with timestamps',
                'authentication': 'API Key required',
                'rate_limit': 'Yes'
            },
            '/generate_chart': {
                'method': 'POST',
                'description': 'Generate sentiment distribution chart',
                'authentication': 'API Key required',
                'rate_limit': 'Yes'
            },
            '/generate_wordcloud': {
                'method': 'POST',
                'description': 'Generate word cloud data',
                'authentication': 'API Key required',
                'rate_limit': 'Yes'
            },
            '/generate_trend_graph': {
                'method': 'POST',
                'description': 'Generate trend graph data',
                'authentication': 'API Key required',
                'rate_limit': 'Yes'
            }
        },
        'security': {
            'authentication': 'API Key in X-API-Key header',
            'rate_limiting': 'Yes',
            'input_validation': 'Yes',
            'cors': 'Configured'
        },
        'timestamp': datetime.utcnow().isoformat()
    }), 200

@app.route('/doc', methods=['GET'])
def api_doc():
    """Alternative API documentation endpoint."""
    return api_docs()

@app.route('/favicon.ico', methods=['GET'])
def favicon():
    """Favicon endpoint."""
    return '', 204

if __name__ == '__main__':
    # Load environment variables
    port = int(os.getenv('FLASK_PORT', 8080))
    host = os.getenv('FLASK_HOST', '0.0.0.0')
    debug = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'
    
    logger.info(f"Starting YouTube Sentiment Analyzer API on {host}:{port}")
    logger.info("Security features enabled: Authentication, Rate Limiting, Input Validation")
    
    app.run(host=host, port=port, debug=debug)
