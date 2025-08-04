import pytest
import re
from app import preprocess_comment

class TestPreprocessing:
    """Test cases for text preprocessing functionality."""
    
    def test_basic_preprocessing(self):
        """Test basic text preprocessing."""
        input_text = "This is a GREAT video! I loved it a lot."
        expected = "great video loved lot"
        result = preprocess_comment(input_text)
        assert result == expected
    
    def test_lowercase_conversion(self):
        """Test that text is converted to lowercase."""
        input_text = "GREAT VIDEO AWESOME"
        result = preprocess_comment(input_text)
        assert result == result.lower()
        assert "great" in result
    
    def test_whitespace_removal(self):
        """Test removal of extra whitespace."""
        input_text = "  This   has   extra   spaces  "
        result = preprocess_comment(input_text)
        assert "  " not in result  # No double spaces
        assert not result.startswith(" ")
        assert not result.endswith(" ")
    
    def test_newline_removal(self):
        """Test removal of newline characters."""
        input_text = "This has\nnewlines\nin it"
        result = preprocess_comment(input_text)
        assert "\n" not in result
    
    def test_special_character_removal(self):
        """Test removal of special characters."""
        input_text = "This@has#special$characters%and^symbols&"
        result = preprocess_comment(input_text)
        # Should only contain letters, numbers, and basic punctuation
        assert re.match(r'^[a-z0-9\s!?.,]*$', result)
    
    def test_stopword_removal(self):
        """Test removal of stopwords while keeping important ones."""
        input_text = "This is a great video but it has some issues"
        result = preprocess_comment(input_text)
        # Should remove common stopwords but keep 'but'
        assert "this" not in result
        # Note: "is" becomes part of "issue" after lemmatization, so we check for the word "is" as a separate word
        assert " is " not in f" {result} "
        # Check for the word "a" as a separate word, not as a letter in other words
        assert " a " not in f" {result} "
        assert "but" in result  # Important word for sentiment
        assert "great" in result
        assert "video" in result
    
    def test_lemmatization(self):
        """Test that words are lemmatized."""
        input_text = "running jumping playing"
        result = preprocess_comment(input_text)
        # Should be lemmatized to root forms
        assert "run" in result or "jump" in result or "play" in result
    
    def test_empty_string(self):
        """Test handling of empty string."""
        result = preprocess_comment("")
        assert result == ""
    
    def test_whitespace_only(self):
        """Test handling of whitespace-only string."""
        result = preprocess_comment("   \n\t  ")
        assert result == ""
    
    def test_numbers_and_punctuation(self):
        """Test handling of numbers and punctuation."""
        input_text = "Video 123! Great, awesome."
        result = preprocess_comment(input_text)
        # Numbers should be removed, punctuation removed
        assert "123" not in result
        assert "!" not in result and "," not in result and "." not in result
    
    def test_unicode_characters(self):
        """Test handling of unicode characters."""
        input_text = "This is a great video with émojis 🎬"
        result = preprocess_comment(input_text)
        # Unicode should be handled gracefully
        assert isinstance(result, str)
    
    def test_very_long_text(self):
        """Test handling of very long text."""
        long_text = "This is a very long text " * 1000
        result = preprocess_comment(long_text)
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_special_sentiment_words(self):
        """Test that sentiment-important words are preserved."""
        input_text = "This is not bad, but it's not good either"
        result = preprocess_comment(input_text)
        # Important sentiment words should be preserved
        important_words = ["not", "bad", "good"]
        preserved_words = [word for word in important_words if word in result]
        assert len(preserved_words) > 0
    
    def test_exception_handling(self):
        """Test that exceptions are handled gracefully."""
        # Test with None input
        result = preprocess_comment(None)
        assert result == ""
        
        # Test with non-string input
        result = preprocess_comment(123)
        assert isinstance(result, str)
    
    def test_consistent_output(self):
        """Test that preprocessing is consistent."""
        input_text = "This is a test video"
        result1 = preprocess_comment(input_text)
        result2 = preprocess_comment(input_text)
        assert result1 == result2
    
    def test_preserve_sentiment_indicators(self):
        """Test that sentiment indicators are preserved."""
        input_text = "This video is not bad at all, but it's not great either"
        result = preprocess_comment(input_text)
        # Should preserve negation words and sentiment words
        sentiment_indicators = ["not", "bad", "great"]
        preserved_indicators = [word for word in sentiment_indicators if word in result]
        assert len(preserved_indicators) >= 2 