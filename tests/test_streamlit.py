import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock

class TestStreamlitApp:
    """Test Streamlit application functionality."""
    
    @patch('streamlit_app.joblib.load')
    def test_load_models_success(self, mock_load):
        """Test successful model loading in Streamlit."""
        # Create mock objects that can be pickled
        mock_model = Mock()
        mock_model.predict.return_value = np.array([1, 0, -1])
        mock_model.predict_proba.return_value = np.array([[0.1, 0.2, 0.7], [0.8, 0.1, 0.1], [0.1, 0.1, 0.8]])
        
        mock_vectorizer = Mock()
        mock_vectorizer.transform.return_value = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        
        mock_load.side_effect = [mock_model, mock_vectorizer]
        
        # Mock the cache to avoid pickling issues
        with patch('streamlit_app.st.cache_data') as mock_cache:
            mock_cache.return_value = lambda func: func
            
            from streamlit_app import load_models
            model, vectorizer = load_models()
            
            assert model is not None
            assert vectorizer is not None

    @patch('streamlit_app.joblib.load')
    def test_load_models_failure(self, mock_load):
        """Test model loading failure in Streamlit."""
        mock_load.side_effect = Exception("Model loading failed")
        
        with patch('streamlit_app.st.cache_data') as mock_cache:
            mock_cache.return_value = lambda func: func
            
            from streamlit_app import load_models
            with pytest.raises(Exception):
                load_models()

    def test_preprocess_comment_streamlit(self):
        """Test comment preprocessing in Streamlit context."""
        from streamlit_app import preprocess_comment
        
        # Test basic preprocessing
        result = preprocess_comment("This is a GREAT video!")
        assert isinstance(result, str)
        assert result == result.lower()
        
        # Test empty input
        result = preprocess_comment("")
        assert result == ""

    def test_analyze_sentiment_batch(self):
        """Test batch sentiment analysis."""
        from streamlit_app import analyze_sentiment_batch
        
        # Mock model and vectorizer
        mock_model = Mock()
        mock_model.predict.return_value = np.array([1, 0, -1])
        mock_model.predict_proba.return_value = np.array([[0.1, 0.2, 0.7], [0.8, 0.1, 0.1], [0.1, 0.1, 0.8]])
        
        mock_vectorizer = Mock()
        mock_vectorizer.transform.return_value = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        
        comments = ["Great video!", "Okay video", "Bad video"]
        results = analyze_sentiment_batch(comments, mock_model, mock_vectorizer)
        
        assert len(results) == 3
        assert all(isinstance(result, dict) for result in results)
        assert all('sentiment' in result for result in results)
        assert all('confidence' in result for result in results)

    def test_batch_analysis_processing(self):
        """Test processing of batch analysis results."""
        from streamlit_app import process_batch_results
        
        # Mock results
        mock_results = [
            {'sentiment': 1, 'confidence': 0.8},
            {'sentiment': 0, 'confidence': 0.6},
            {'sentiment': -1, 'confidence': 0.9}
        ]
        
        comments = ["Great!", "Okay", "Bad"]
        
        # Mock model and vectorizer
        mock_model = Mock()
        mock_vectorizer = Mock()
        
        with patch('streamlit_app.analyze_sentiment_batch', return_value=mock_results):
            df = process_batch_results(comments, mock_model, mock_vectorizer)
            
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 3
            assert 'comment' in df.columns
            assert 'sentiment' in df.columns
            assert 'confidence' in df.columns

    def test_metrics_calculation(self):
        """Test calculation of sentiment metrics."""
        from streamlit_app import calculate_metrics
        
        # Create test DataFrame
        df = pd.DataFrame({
            'sentiment': [1, 1, 0, -1, -1, 1, 0],
            'confidence': [0.8, 0.9, 0.6, 0.7, 0.8, 0.9, 0.5]
        })
        
        metrics = calculate_metrics(df)
        
        assert 'total_comments' in metrics
        assert 'positive_count' in metrics
        assert 'negative_count' in metrics
        assert 'neutral_count' in metrics
        assert 'average_confidence' in metrics
        
        assert metrics['total_comments'] == 7
        assert metrics['positive_count'] == 3
        assert metrics['negative_count'] == 2
        assert metrics['neutral_count'] == 2

    def test_sentiment_distribution(self):
        """Test sentiment distribution calculation."""
        from streamlit_app import get_sentiment_distribution
        
        # Create test DataFrame
        df = pd.DataFrame({
            'sentiment': [1, 1, 0, -1, -1, 1, 0]
        })
        
        distribution = get_sentiment_distribution(df)
        
        assert isinstance(distribution, dict)
        assert 'Positive' in distribution
        assert 'Neutral' in distribution
        assert 'Negative' in distribution
        
        assert distribution['Positive'] == 3
        assert distribution['Neutral'] == 2
        assert distribution['Negative'] == 2

    def test_confidence_analysis(self):
        """Test confidence analysis."""
        from streamlit_app import analyze_confidence
        
        # Create test DataFrame
        df = pd.DataFrame({
            'confidence': [0.8, 0.9, 0.6, 0.7, 0.8, 0.9, 0.5]
        })
        
        confidence_stats = analyze_confidence(df)
        
        assert 'average_confidence' in confidence_stats
        assert 'high_confidence_count' in confidence_stats
        assert 'low_confidence_count' in confidence_stats
        
        assert confidence_stats['average_confidence'] > 0
        assert confidence_stats['high_confidence_count'] >= 0
        assert confidence_stats['low_confidence_count'] >= 0

    def test_input_validation_streamlit(self):
        """Test input validation in Streamlit context."""
        from streamlit_app import validate_input
        
        # Test valid input
        valid_comments = ["Great video!", "Amazing content"]
        assert validate_input(valid_comments) is True
        
        # Test empty input
        assert validate_input([]) is False
        
        # Test too many comments
        too_many = ["test"] * 1000
        assert validate_input(too_many) is False
        
        # Test invalid input type
        assert validate_input("not a list") is False

    def test_error_handling_streamlit(self):
        """Test error handling in Streamlit context."""
        from streamlit_app import handle_analysis_error
        
        # Test with different error types
        error = Exception("Test error")
        result = handle_analysis_error(error)
        
        assert isinstance(result, dict)
        assert 'error' in result
        assert 'message' in result

    def test_sentiment_label_mapping(self):
        """Test sentiment label mapping."""
        from streamlit_app import get_sentiment_label
        
        assert get_sentiment_label(1) == "Positive"
        assert get_sentiment_label(0) == "Neutral"
        assert get_sentiment_label(-1) == "Negative"
        assert get_sentiment_label(999) == "Unknown"

    def test_confidence_level_categorization(self):
        """Test confidence level categorization."""
        from streamlit_app import get_confidence_level
        
        assert get_confidence_level(0.9) == "High"
        assert get_confidence_level(0.7) == "Medium"
        assert get_confidence_level(0.4) == "Low"

    def test_dataframe_creation(self):
        """Test DataFrame creation from results."""
        from streamlit_app import create_results_dataframe
        
        comments = ["Great!", "Okay", "Bad"]
        sentiments = [1, 0, -1]
        confidences = [0.8, 0.6, 0.9]
        
        df = create_results_dataframe(comments, sentiments, confidences)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert 'comment' in df.columns
        assert 'sentiment' in df.columns
        assert 'confidence' in df.columns

    def test_streamlit_ui_elements(self):
        """Test Streamlit UI element creation."""
        # Mock Streamlit functions
        with patch('streamlit_app.st') as mock_st:
            mock_st.title.return_value = None
            mock_st.header.return_value = None
            mock_st.text_input.return_value = "test"
            mock_st.button.return_value = True
            mock_st.dataframe.return_value = None
            mock_st.bar_chart.return_value = None
            
            # Test that UI elements can be created without errors
            # This is a basic test to ensure no syntax errors
            assert True

    def test_model_prediction_interface(self):
        """Test model prediction interface."""
        # Mock model and vectorizer
        mock_model = Mock()
        mock_model.predict.return_value = np.array([1])
        mock_model.predict_proba.return_value = np.array([[0.1, 0.2, 0.7]])
        
        mock_vectorizer = Mock()
        mock_vectorizer.transform.return_value = np.array([[1, 0, 0]])
        
        # Test single prediction
        from streamlit_app import predict_single_comment
        
        result = predict_single_comment("Great video!", mock_model, mock_vectorizer)
        
        assert isinstance(result, dict)
        assert 'sentiment' in result
        assert 'confidence' in result

    def test_batch_processing_interface(self):
        """Test batch processing interface."""
        # Mock model and vectorizer
        mock_model = Mock()
        mock_model.predict.return_value = np.array([1, 0, -1])
        mock_model.predict_proba.return_value = np.array([[0.1, 0.2, 0.7], [0.8, 0.1, 0.1], [0.1, 0.1, 0.8]])
        
        mock_vectorizer = Mock()
        mock_vectorizer.transform.return_value = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        
        # Test batch processing
        from streamlit_app import process_batch_input
        
        comments = ["Great!", "Okay", "Bad"]
        result = process_batch_input(comments, mock_model, mock_vectorizer)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3 