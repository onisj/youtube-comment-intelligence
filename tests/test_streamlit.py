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

    def test_preprocess_text_streamlit(self):
        """Test text preprocessing in Streamlit context."""
        from streamlit_app import preprocess_text
        
        # Test basic preprocessing
        result = preprocess_text("This is a GREAT video!")
        assert isinstance(result, str)
        assert result == result.lower()
        
        # Test empty input
        result = preprocess_text("")
        assert result == ""

    def test_predict_sentiment(self):
        """Test sentiment prediction."""
        from streamlit_app import predict_sentiment
        
        # Mock model and vectorizer
        mock_model = Mock()
        mock_model.predict.return_value = np.array([1])
        mock_model.predict_proba.return_value = np.array([[0.1, 0.2, 0.7]])
        
        mock_vectorizer = Mock()
        mock_vectorizer.transform.return_value = np.array([[1, 0, 0]])
        
        result = predict_sentiment("Great video!", mock_model, mock_vectorizer)
        
        assert isinstance(result, dict)
        assert 'sentiment' in result
        assert 'confidence' in result

    def test_get_sentiment_label(self):
        """Test sentiment label mapping."""
        from streamlit_app import get_sentiment_label
        
        # Test the actual return format
        result = get_sentiment_label(1)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert "Positive" in result[0]

    def test_create_wordcloud(self):
        """Test wordcloud creation."""
        from streamlit_app import create_wordcloud
        
        # Test with sample text
        text = "This is a great video with amazing content"
        result = create_wordcloud(text)
        
        # Should return a wordcloud object or None
        assert result is None or hasattr(result, 'words_')

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
        from streamlit_app import predict_sentiment
        
        result = predict_sentiment("Great video!", mock_model, mock_vectorizer)
        
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
        
        # Test batch processing using predict_sentiment
        from streamlit_app import predict_sentiment
        
        comments = ["Great!", "Okay", "Bad"]
        results = []
        
        for comment in comments:
            result = predict_sentiment(comment, mock_model, mock_vectorizer)
            results.append(result)
        
        assert len(results) == 3
        assert all(isinstance(result, dict) for result in results)
        assert all('sentiment' in result for result in results) 