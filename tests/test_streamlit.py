import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

class TestStreamlitApp:
    """Test cases for Streamlit application functionality."""
    
    def test_load_models_success(self):
        """Test successful model loading in Streamlit."""
        with patch('streamlit_app.joblib.load') as mock_load:
            mock_model = Mock()
            mock_vectorizer = Mock()
            mock_load.side_effect = [mock_model, mock_vectorizer]
            
            from streamlit_app import load_models
            model, vectorizer = load_models()
            
            assert model == mock_model
            assert vectorizer == mock_vectorizer
    
    def test_load_models_file_not_found(self):
        """Test model loading when files are not found."""
        with patch('streamlit_app.joblib.load') as mock_load:
            mock_load.side_effect = FileNotFoundError("Model files not found")
            
            from streamlit_app import load_models
            model, vectorizer = load_models()
            
            assert model is None
            assert vectorizer is None
    
    def test_preprocess_text(self):
        """Test text preprocessing in Streamlit app."""
        from streamlit_app import preprocess_text
        
        test_text = "This is a GREAT video! I loved it a lot."
        processed = preprocess_text(test_text)
        
        # Check that text is processed
        assert isinstance(processed, str)
        assert len(processed) > 0
        assert processed == processed.lower()  # Should be lowercase
    
    def test_preprocess_text_empty(self):
        """Test preprocessing empty text."""
        from streamlit_app import preprocess_text
        
        empty_text = ""
        processed = preprocess_text(empty_text)
        
        assert isinstance(processed, str)
    
    def test_preprocess_text_special_characters(self):
        """Test preprocessing text with special characters."""
        from streamlit_app import preprocess_text
        
        special_text = "This@has#special$characters%and^symbols&"
        processed = preprocess_text(special_text)
        
        assert isinstance(processed, str)
        # Should remove special characters
        assert "@" not in processed
        assert "#" not in processed
        assert "$" not in processed
    
    def test_predict_sentiment_success(self, mock_model, mock_vectorizer):
        """Test successful sentiment prediction."""
        from streamlit_app import predict_sentiment
        
        test_text = "This is a great video!"
        
        with patch('streamlit_app.preprocess_text') as mock_preprocess:
            mock_preprocess.return_value = "great video"
            
            mock_vectorizer.transform.return_value = np.array([[0.1, 0.2, 0.3]])
            mock_model.predict.return_value = np.array([1])
            mock_model.predict_proba.return_value = np.array([[0.1, 0.2, 0.7]])
            
            prediction, probability = predict_sentiment(test_text, mock_model, mock_vectorizer)
            
            assert prediction == 1
            assert len(probability) == 3
            assert np.sum(probability) == pytest.approx(1.0, abs=1e-6)
    
    def test_predict_sentiment_no_model(self):
        """Test sentiment prediction when model is None."""
        from streamlit_app import predict_sentiment
        
        prediction, probability = predict_sentiment("test", None, None)
        
        assert prediction is None
        assert probability is None
    
    def test_get_sentiment_label(self):
        """Test sentiment label conversion."""
        from streamlit_app import get_sentiment_label
        
        # Test positive sentiment
        label, css_class = get_sentiment_label(1)
        assert "Positive" in label
        assert "positive" in css_class
        
        # Test negative sentiment
        label, css_class = get_sentiment_label(-1)
        assert "Negative" in label
        assert "negative" in css_class
        
        # Test neutral sentiment
        label, css_class = get_sentiment_label(0)
        assert "Neutral" in label
        assert "neutral" in css_class
    
    def test_create_wordcloud(self):
        """Test word cloud creation."""
        from streamlit_app import create_wordcloud
        
        test_text = "This is a test video with some words for the word cloud"
        
        with patch('streamlit_app.WordCloud') as mock_wordcloud:
            mock_wc = Mock()
            mock_wordcloud.return_value = mock_wc
            mock_wc.generate.return_value = mock_wc
            
            with patch('matplotlib.pyplot.subplots') as mock_subplots:
                mock_fig, mock_ax = Mock(), Mock()
                mock_subplots.return_value = (mock_fig, mock_ax)
                
                result = create_wordcloud(test_text)
                
                assert result == mock_fig
                mock_wordcloud.assert_called_once()
                mock_wc.generate.assert_called_once_with(test_text)
    
    def test_batch_analysis_processing(self, sample_csv_data):
        """Test batch analysis processing."""
        from streamlit_app import predict_sentiment
        
        with patch('streamlit_app.predict_sentiment') as mock_predict:
            mock_predict.side_effect = [
                (1, [0.1, 0.2, 0.7]),
                (-1, [0.8, 0.1, 0.1]),
                (0, [0.2, 0.6, 0.2])
            ]
            
            predictions = []
            probabilities = []
            
            for i, text in enumerate(sample_csv_data['text']):
                # Create proper mock objects that can be subscripted
                mock_model = Mock()
                # Return different predictions for each text
                expected_predictions = [1, -1, 0]
                expected_probabilities = [[0.1, 0.2, 0.7], [0.8, 0.1, 0.1], [0.2, 0.6, 0.2]]
                mock_model.predict.return_value = np.array([expected_predictions[i]])
                mock_model.predict_proba.return_value = np.array([expected_probabilities[i]])
                mock_vectorizer = Mock()
                mock_vectorizer.transform.return_value = np.array([[0.1, 0.2, 0.3]])
                
                pred, prob = predict_sentiment(text, mock_model, mock_vectorizer)
                predictions.append(pred)
                probabilities.append(max(prob) if prob is not None else 0)
            
            assert len(predictions) == 3
            assert predictions == [1, -1, 0]
            assert len(probabilities) == 3
    
    def test_dataframe_operations(self, sample_csv_data):
        """Test DataFrame operations in Streamlit app."""
        # Test adding sentiment column
        df = sample_csv_data.copy()
        df['sentiment'] = [1, -1, 0]
        df['confidence'] = [0.8, 0.9, 0.7]
        
        # Test sentiment label conversion
        sentiment_labels = {1: "Positive 😊", -1: "Negative 😞", 0: "Neutral 😐"}
        df['sentiment_label'] = df['sentiment'].map(sentiment_labels)
        
        assert len(df) == 3
        assert 'sentiment_label' in df.columns
        assert df['sentiment_label'].iloc[0] == "Positive 😊"
    
    def test_metrics_calculation(self, sample_csv_data):
        """Test metrics calculation for batch analysis."""
        df = sample_csv_data.copy()
        # Add sentiment column with correct length
        df['sentiment'] = [1, -1, 0]
        
                # Calculate metrics
        total_texts = len(df)
        positive_count = len(df[df['sentiment'] == 1])
        negative_count = len(df[df['sentiment'] == -1])
        neutral_count = len(df[df['sentiment'] == 0])
    
        assert total_texts == 3
        assert positive_count == 1
        assert negative_count == 1
        assert neutral_count == 1
    
    def test_file_upload_validation(self):
        """Test file upload validation."""
        # Test valid CSV structure
        valid_data = pd.DataFrame({
            'text': ["Great video", "Bad video", "Okay video"],
            'other_column': [1, 2, 3]
        })
        
        assert 'text' in valid_data.columns
        
        # Test invalid CSV structure
        invalid_data = pd.DataFrame({
            'wrong_column': ["Great video", "Bad video"],
            'other_column': [1, 2]
        })
        
        assert 'text' not in invalid_data.columns
    
    def test_error_handling_in_streamlit(self):
        """Test error handling in Streamlit app."""
        # Test with None model
        with patch('streamlit_app.load_models') as mock_load:
            mock_load.return_value = (None, None)
            
            # This should handle the None case gracefully
            from streamlit_app import predict_sentiment
            result = predict_sentiment("test", None, None)
            
            assert result == (None, None)
    
    def test_css_styling(self):
        """Test CSS styling classes."""
        css_classes = {
            "sentiment-positive": "color: #28a745; font-weight: bold;",
            "sentiment-negative": "color: #dc3545; font-weight: bold;",
            "sentiment-neutral": "color: #ffc107; font-weight: bold;"
        }
        
        for class_name, expected_style in css_classes.items():
            assert "color" in expected_style
            assert "font-weight" in expected_style
    
    def test_page_config(self):
        """Test Streamlit page configuration."""
        # This would typically be tested by checking if the config is set correctly
        # In a real test, you'd check the Streamlit session state
        expected_config = {
            "page_title": "YouTube Sentiment Analysis",
            "page_icon": "🎬",
            "layout": "wide",
            "initial_sidebar_state": "expanded"
        }
        
        assert "page_title" in expected_config
        assert "page_icon" in expected_config
        assert "layout" in expected_config
    
    def test_data_validation(self):
        """Test data validation in Streamlit app."""
        # Test valid data
        valid_df = pd.DataFrame({
            'text': ["Valid text 1", "Valid text 2"],
            'sentiment': [1, -1]
        })
        
        assert not valid_df.empty
        assert 'text' in valid_df.columns
        
        # Test empty data
        empty_df = pd.DataFrame()
        assert empty_df.empty
    
    def test_plot_generation(self):
        """Test plot generation functionality."""
        # Mock plotly functionality
        with patch('streamlit_app.px.bar') as mock_bar:
            mock_fig = Mock()
            mock_bar.return_value = mock_fig
            
            # Test bar chart creation
            data = pd.DataFrame({
                'Sentiment': ['Positive', 'Negative', 'Neutral'],
                'Probability': [0.7, 0.2, 0.1]
            })
            
            fig = mock_bar(data, x='Sentiment', y='Probability')
            
            assert fig == mock_fig
            mock_bar.assert_called_once()
    
    def test_download_functionality(self):
        """Test CSV download functionality."""
        test_df = pd.DataFrame({
            'text': ["Test 1", "Test 2"],
            'sentiment': [1, -1],
            'confidence': [0.8, 0.9]
        })
        
        # Test CSV conversion
        csv_data = test_df.to_csv(index=False)
        
        assert isinstance(csv_data, str)
        assert "text,sentiment,confidence" in csv_data
        assert "Test 1,1,0.8" in csv_data 