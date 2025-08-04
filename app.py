import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend before importing pyplot

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import io
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import mlflow
import numpy as np
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from mlflow.tracking import MlflowClient
import matplotlib.dates as mdates
import pickle
import logging
import traceback
from datetime import datetime
import os
from typing import List, Dict, Any, Optional, Union
import json

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
MAX_COMMENT_LENGTH = 10000
MAX_COMMENTS_PER_REQUEST = 1000
ALLOWED_CONTENT_TYPES = ['application/json']

class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass

class ModelError(Exception):
    """Custom exception for model-related errors."""
    pass

def validate_input_data(data: Dict[str, Any]) -> None:
    """Validate input data structure and content."""
    if not isinstance(data, dict):
        raise ValidationError("Input must be a JSON object")
    
    if 'comments' not in data:
        raise ValidationError("Missing 'comments' field in request")
    
    comments = data['comments']
    if not isinstance(comments, list):
        raise ValidationError("'comments' must be a list")
    
    if len(comments) == 0:
        raise ValidationError("Comments list cannot be empty")
    
    if len(comments) > MAX_COMMENTS_PER_REQUEST:
        raise ValidationError(f"Too many comments. Maximum allowed: {MAX_COMMENTS_PER_REQUEST}")
    
    for i, comment in enumerate(comments):
        if not isinstance(comment, str):
            raise ValidationError(f"Comment at index {i} must be a string")
        
        if len(comment) > MAX_COMMENT_LENGTH:
            raise ValidationError(f"Comment at index {i} is too long. Maximum length: {MAX_COMMENT_LENGTH}")

def validate_timestamp_data(data: Dict[str, Any]) -> None:
    """Validate timestamp data structure."""
    if not isinstance(data, dict):
        raise ValidationError("Input must be a JSON object")
    
    if 'comments' not in data:
        raise ValidationError("Missing 'comments' field in request")
    
    comments = data['comments']
    if not isinstance(comments, list):
        raise ValidationError("'comments' must be a list")
    
    if len(comments) == 0:
        raise ValidationError("Comments list cannot be empty")
    
    for i, comment_data in enumerate(comments):
        if not isinstance(comment_data, dict):
            raise ValidationError(f"Comment data at index {i} must be an object")
        
        if 'text' not in comment_data:
            raise ValidationError(f"Missing 'text' field in comment at index {i}")
        
        if 'timestamp' not in comment_data:
            raise ValidationError(f"Missing 'timestamp' field in comment at index {i}")
        
        if not isinstance(comment_data['text'], str):
            raise ValidationError(f"Text field at index {i} must be a string")
        
        if not isinstance(comment_data['timestamp'], str):
            raise ValidationError(f"Timestamp field at index {i} must be a string")

def validate_chart_data(data: Dict[str, Any]) -> None:
    """Validate chart generation data."""
    if not isinstance(data, dict):
        raise ValidationError("Input must be a JSON object")
    
    if 'sentiment_counts' not in data:
        raise ValidationError("Missing 'sentiment_counts' field")
    
    counts = data['sentiment_counts']
    if not isinstance(counts, dict):
        raise ValidationError("'sentiment_counts' must be an object")
    
    valid_keys = {'1', '0', '-1'}
    if not all(key in valid_keys for key in counts.keys()):
        raise ValidationError("Invalid sentiment count keys. Must be '1', '0', or '-1'")

def validate_trend_data(data: Dict[str, Any]) -> None:
    """Validate trend graph data."""
    if not isinstance(data, dict):
        raise ValidationError("Input must be a JSON object")
    
    if 'sentiment_data' not in data:
        raise ValidationError("Missing 'sentiment_data' field")
    
    sentiment_data = data['sentiment_data']
    if not isinstance(sentiment_data, list):
        raise ValidationError("'sentiment_data' must be a list")
    
    if len(sentiment_data) == 0:
        raise ValidationError("Sentiment data list cannot be empty")
    
    for i, item in enumerate(sentiment_data):
        if not isinstance(item, dict):
            raise ValidationError(f"Sentiment data item at index {i} must be an object")
        
        required_fields = ['text', 'sentiment', 'timestamp']
        for field in required_fields:
            if field not in item:
                raise ValidationError(f"Missing '{field}' field in sentiment data at index {i}")

# Define the preprocessing function with improved error handling
def preprocess_comment(comment: str) -> str:
    """Apply preprocessing transformations to a comment."""
    try:
        if comment is None:
            return ""
        
        # Convert to string if not already
        comment = str(comment)
        
        # Convert to lowercase
        comment = comment.lower()

        # Remove trailing and leading whitespaces
        comment = comment.strip()

        # Remove newline characters
        comment = re.sub(r'\n', ' ', comment)

        # Remove numbers and non-alphanumeric characters, except basic punctuation
        comment = re.sub(r'[^A-Za-z\s!?.,]', '', comment)

        # Remove stopwords but retain important ones for sentiment analysis
        stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
        # Also remove common words that might not be in stopwords
        additional_stopwords = {'is', 'are', 'was', 'were', 'be', 'been', 'being'}
        stop_words.update(additional_stopwords)
        comment = ' '.join([word for word in comment.split() if word not in stop_words])
        
        # Remove punctuation from individual words
        comment = re.sub(r'[!?.,]', '', comment)

        # Lemmatize the words
        lemmatizer = WordNetLemmatizer()
        comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])

        # Remove punctuation at the end
        comment = re.sub(r'[!?.,]+$', '', comment)
        comment = re.sub(r'^[!?.,]+', '', comment)

        return comment.strip()
    except Exception as e:
        logger.error(f"Error in preprocessing comment: {e}")
        logger.error(f"Comment that caused error: {comment}")
        return comment

def load_model(model_path: str, vectorizer_path: str) -> tuple:
    """Load the trained model with error handling."""
    try:
        if not os.path.exists(model_path):
            raise ModelError(f"Model file not found: {model_path}")
        
        if not os.path.exists(vectorizer_path):
            raise ModelError(f"Vectorizer file not found: {vectorizer_path}")
        
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        
        with open(vectorizer_path, 'rb') as file:
            vectorizer = pickle.load(file)
      
        logger.info("Model and vectorizer loaded successfully")
        return model, vectorizer
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise ModelError(f"Failed to load model: {str(e)}")

# Initialize the model and vectorizer with error handling
try:
    model, vectorizer = load_model("models/lgbm_model.pkl", "models/tfidf_vectorizer.pkl")
    logger.info("Application initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize application: {e}")
    model, vectorizer = None, None

@app.errorhandler(ValidationError)
def handle_validation_error(error):
    """Handle validation errors."""
    logger.warning(f"Validation error: {error}")
    return jsonify({
        "error": "Validation error",
        "message": str(error),
        "timestamp": datetime.now().isoformat()
    }), 400

@app.errorhandler(ModelError)
def handle_model_error(error):
    """Handle model-related errors."""
    logger.error(f"Model error: {error}")
    return jsonify({
        "error": "Model error",
        "message": str(error),
        "timestamp": datetime.now().isoformat()
    }), 500

@app.errorhandler(Exception)
def handle_generic_error(error):
    """Handle generic errors."""
    logger.error(f"Unexpected error: {error}")
    logger.error(f"Traceback: {traceback.format_exc()}")
    return jsonify({
        "error": "Internal server error",
        "message": "An unexpected error occurred",
        "timestamp": datetime.now().isoformat()
    }), 500

@app.before_request
def log_request():
    """Log incoming requests."""
    logger.info(f"Request: {request.method} {request.path} from {request.remote_addr}")

@app.after_request
def log_response(response):
    """Log outgoing responses."""
    logger.info(f"Response: {response.status_code} for {request.method} {request.path}")
    return response

@app.route('/')
def home():
    """Home endpoint."""
    return jsonify({
        "message": "Welcome to YouTube Sentiment Analysis API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "POST /predict",
            "predict_with_timestamps": "POST /predict_with_timestamps",
            "generate_chart": "POST /generate_chart",
            "generate_wordcloud": "POST /generate_wordcloud",
            "generate_trend_graph": "POST /generate_trend_graph"
        },
        "timestamp": datetime.now().isoformat()
    })

@app.route('/health')
def health_check():
    """Health check endpoint."""
    try:
        # Check if model is loaded
        if model is None or vectorizer is None:
            return jsonify({
                "status": "unhealthy",
                "message": "Model not loaded",
                "timestamp": datetime.now().isoformat()
            }), 503
        
        # Test model prediction
        test_text = "test"
        preprocessed = preprocess_comment(test_text)
        vectorized = vectorizer.transform([preprocessed])
        prediction = model.predict(vectorized)
        
        return jsonify({
            "status": "healthy",
            "message": "All systems operational",
            "model_loaded": True,
            "timestamp": datetime.now().isoformat()
        }), 200
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            "status": "unhealthy",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }), 503

@app.route('/docs')
@app.route('/doc')
def api_docs():
    """API documentation endpoint."""
    return jsonify({
        'message': 'YouTube Sentiment Analysis API Documentation',
        'version': '1.0.0',
        'endpoints': {
            'GET /': 'API information',
            'GET /health': 'Health check',
            'GET /docs': 'API documentation',
            'POST /predict': {
                'description': 'Predict sentiment for comments',
                'input': {'comments': ['comment1', 'comment2', ...]},
                'output': [{'comment': 'text', 'sentiment': 0/1}, ...]
            },
            'POST /predict_with_timestamps': {
                'description': 'Predict sentiment with timestamps',
                'input': {'comments': [{'text': 'comment', 'timestamp': 'time'}, ...]},
                'output': [{'comment': 'text', 'sentiment': 0/1, 'timestamp': 'time'}, ...]
            },
            'POST /generate_chart': {
                'description': 'Generate sentiment distribution chart',
                'input': {'sentiment_data': [0, 1, 1, 0, ...]},
                'output': 'Base64 encoded chart image'
            },
            'POST /generate_wordcloud': {
                'description': 'Generate word cloud from comments',
                'input': {'comments': ['comment1', 'comment2', ...]},
                'output': 'Base64 encoded word cloud image'
            },
            'POST /generate_trend_graph': {
                'description': 'Generate sentiment trend over time',
                'input': {'data': [{'timestamp': 'time', 'sentiment': 0/1}, ...]},
                'output': 'Base64 encoded trend graph'
            }
        }
    })

@app.route('/favicon.ico')
def favicon():
    """Favicon endpoint to prevent 404 errors."""
    return '', 204  # No content response

@app.route('/predict_with_timestamps', methods=['POST'])
def predict_with_timestamps():
    """Predict sentiment with timestamps."""
    try:
        # Validate content type
        if request.content_type not in ALLOWED_CONTENT_TYPES:
            return jsonify({
                "error": "Invalid content type",
                "message": "Content-Type must be application/json"
            }), 400
        
        # Parse and validate JSON
        try:
            data = request.get_json()
        except Exception as e:
            return jsonify({
                "error": "Invalid JSON",
                "message": str(e)
            }), 400
        
        if data is None:
            return jsonify({
                "error": "Invalid JSON",
                "message": "Request body must be valid JSON"
            }), 400
        
        # Validate input data
        validate_timestamp_data(data)
        
        comments_data = data['comments']
        comments = [item['text'] for item in comments_data]
        timestamps = [item['timestamp'] for item in comments_data]

        # Preprocess each comment before vectorizing
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]
        
        # Transform comments using the vectorizer
        transformed_comments = vectorizer.transform(preprocessed_comments)

        # Convert the sparse matrix to dense format
        dense_comments = transformed_comments.toarray()
        
        # Make predictions
        predictions = model.predict(dense_comments).tolist()
        
        # Convert predictions to strings for consistency
        predictions = [str(pred) for pred in predictions]
        
        # Return the response with original comments, predicted sentiments, and timestamps
        response = [
            {
                "comment": comment, 
                "sentiment": sentiment, 
                "timestamp": timestamp
            } 
            for comment, sentiment, timestamp in zip(comments, predictions, timestamps)
        ]
        
        logger.info(f"Successfully processed {len(comments)} comments with timestamps")
        return jsonify(response)
        
    except ValidationError as e:
        return jsonify({
            "error": "Validation error",
            "message": str(e)
        }), 400
    except Exception as e:
        logger.error(f"Error in predict_with_timestamps: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({
            "error": "Prediction failed",
            "message": str(e)
        }), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Predict sentiment for comments."""
    try:
        # Validate content type
        if request.content_type not in ALLOWED_CONTENT_TYPES:
            return jsonify({
                "error": "Invalid content type",
                "message": "Content-Type must be application/json"
            }), 400
        
        # Parse and validate JSON
        try:
            data = request.get_json()
        except Exception as e:
            return jsonify({
                "error": "Invalid JSON",
                "message": str(e)
            }), 400
        
        if data is None:
            return jsonify({
                "error": "Invalid JSON",
                "message": "Request body must be valid JSON"
            }), 400
        
        # Validate input data
        validate_input_data(data)
        
        comments = data['comments']
        logger.info(f"Processing {len(comments)} comments")
        
        # Preprocess each comment before vectorizing
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]
        
        # Transform comments using the vectorizer
        transformed_comments = vectorizer.transform(preprocessed_comments)

        # Convert the sparse matrix to dense format
        dense_comments = transformed_comments.toarray()
        
        # Make predictions
        predictions = model.predict(dense_comments).tolist()
        
        # Return the response with original comments and predicted sentiments
        response = [
            {"comment": comment, "sentiment": sentiment} 
            for comment, sentiment in zip(comments, predictions)
        ]
        
        logger.info(f"Successfully processed {len(comments)} comments")
        return jsonify(response)
        
    except ValidationError as e:
        return jsonify({
            "error": "Validation error",
            "message": str(e)
        }), 400
    except Exception as e:
        logger.error(f"Error in predict: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({
            "error": "Prediction failed",
            "message": str(e)
        }), 500

@app.route('/generate_chart', methods=['POST'])
def generate_chart():
    """Generate sentiment distribution chart."""
    try:
        # Validate content type
        if request.content_type not in ALLOWED_CONTENT_TYPES:
            return jsonify({
                "error": "Invalid content type",
                "message": "Content-Type must be application/json"
            }), 400
        
        # Parse and validate JSON
        try:
            data = request.get_json()
        except Exception as e:
            return jsonify({
                "error": "Invalid JSON",
                "message": str(e)
            }), 400
        
        if data is None:
            return jsonify({
                "error": "Invalid JSON",
                "message": "Request body must be valid JSON"
            }), 400
        
        # Validate chart data
        validate_chart_data(data)
        
        sentiment_counts = data['sentiment_counts']
        
        # Prepare data for the pie chart
        labels = ['Positive', 'Neutral', 'Negative']
        sizes = [
            int(sentiment_counts.get('1', 0)),
            int(sentiment_counts.get('0', 0)),
            int(sentiment_counts.get('-1', 0))
        ]
        
        if sum(sizes) == 0:
            return jsonify({
                "error": "Invalid data",
                "message": "Sentiment counts sum to zero"
            }), 400
        
        colors = ['#36A2EB', '#C9CBCF', '#FF6384']  # Blue, Gray, Red

        # Generate the pie chart
        plt.figure(figsize=(6, 6))
        plt.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct='%1.1f%%',
            startangle=140,
            textprops={'color': 'w'}
        )
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        # Save the chart to a BytesIO object
        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG', transparent=True)
        img_io.seek(0)
        plt.close()

        logger.info("Successfully generated sentiment chart")
        return send_file(img_io, mimetype='image/png')
        
    except ValidationError as e:
        return jsonify({
            "error": "Validation error",
            "message": str(e)
        }), 400
    except Exception as e:
        logger.error(f"Error in generate_chart: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({
            "error": "Chart generation failed",
            "message": str(e)
        }), 500

@app.route('/generate_wordcloud', methods=['POST'])
def generate_wordcloud():
    """Generate word cloud from comments."""
    try:
        # Validate content type
        if request.content_type not in ALLOWED_CONTENT_TYPES:
            return jsonify({
                "error": "Invalid content type",
                "message": "Content-Type must be application/json"
            }), 400
        
        # Parse and validate JSON
        try:
            data = request.get_json()
        except Exception as e:
            return jsonify({
                "error": "Invalid JSON",
                "message": str(e)
            }), 400
        
        if data is None:
            return jsonify({
                "error": "Invalid JSON",
                "message": "Request body must be valid JSON"
            }), 400
        
        # Validate input data
        validate_input_data(data)
        
        comments = data['comments']

        # Preprocess comments
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]

        # Combine all comments into a single string
        text = ' '.join(preprocessed_comments)

        # Generate the word cloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='black',
            colormap='Blues',
            stopwords=set(stopwords.words('english')),
            collocations=False
        ).generate(text)

        # Save the word cloud to a BytesIO object
        img_io = io.BytesIO()
        wordcloud.to_image().save(img_io, format='PNG')
        img_io.seek(0)

        logger.info("Successfully generated word cloud")
        return send_file(img_io, mimetype='image/png')
        
    except ValidationError as e:
        return jsonify({
            "error": "Validation error",
            "message": str(e)
        }), 400
    except Exception as e:
        logger.error(f"Error in generate_wordcloud: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({
            "error": "Word cloud generation failed",
            "message": str(e)
        }), 500

@app.route('/generate_trend_graph', methods=['POST'])
def generate_trend_graph():
    """Generate trend graph from sentiment data."""
    try:
        # Validate content type
        if request.content_type not in ALLOWED_CONTENT_TYPES:
            return jsonify({
                "error": "Invalid content type",
                "message": "Content-Type must be application/json"
            }), 400
        
        # Parse and validate JSON
        try:
            data = request.get_json()
        except Exception as e:
            return jsonify({
                "error": "Invalid JSON",
                "message": str(e)
            }), 400
        
        if data is None:
            return jsonify({
                "error": "Invalid JSON",
                "message": "Request body must be valid JSON"
            }), 400
        
        # Validate trend data
        validate_trend_data(data)
        
        sentiment_data = data['sentiment_data']

        # Convert sentiment_data to DataFrame
        df = pd.DataFrame(sentiment_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Set the timestamp as the index
        df.set_index('timestamp', inplace=True)

        # Ensure the 'sentiment' column is numeric
        df['sentiment'] = df['sentiment'].astype(int)

        # Map sentiment values to labels
        sentiment_labels = {-1: 'Negative', 0: 'Neutral', 1: 'Positive'}

        # Resample the data over monthly intervals and count sentiments
        monthly_counts = df.resample('M')['sentiment'].value_counts().unstack(fill_value=0)

        # Calculate total counts per month
        monthly_totals = monthly_counts.sum(axis=1)

        # Calculate percentages
        monthly_percentages = (monthly_counts.T / monthly_totals).T * 100

        # Ensure all sentiment columns are present
        for sentiment_value in [-1, 0, 1]:
            if sentiment_value not in monthly_percentages.columns:
                monthly_percentages[sentiment_value] = 0

        # Sort columns by sentiment value
        monthly_percentages = monthly_percentages[[-1, 0, 1]]

        # Plotting
        plt.figure(figsize=(12, 6))

        colors = {
            -1: 'red',     # Negative sentiment
            0: 'gray',     # Neutral sentiment
            1: 'green'     # Positive sentiment
        }

        for sentiment_value in [-1, 0, 1]:
            plt.plot(
                monthly_percentages.index,
                monthly_percentages[sentiment_value],
                marker='o',
                linestyle='-',
                label=sentiment_labels[sentiment_value],
                color=colors[sentiment_value]
            )

        plt.title('Monthly Sentiment Percentage Over Time')
        plt.xlabel('Month')
        plt.ylabel('Percentage of Comments (%)')
        plt.grid(True)
        plt.xticks(rotation=45)

        # Format the x-axis dates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=12))

        plt.legend()
        plt.tight_layout()

        # Save the trend graph to a BytesIO object
        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG')
        img_io.seek(0)
        plt.close()

        logger.info("Successfully generated trend graph")
        return send_file(img_io, mimetype='image/png')
        
    except ValidationError as e:
        return jsonify({
            "error": "Validation error",
            "message": str(e)
        }), 400
    except Exception as e:
        logger.error(f"Error in generate_trend_graph: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({
            "error": "Trend graph generation failed",
            "message": str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
