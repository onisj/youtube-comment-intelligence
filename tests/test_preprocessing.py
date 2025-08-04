import pytest
from app import preprocess_comment

class TestPreprocessing:
    """Test preprocessing functionality."""
    
    def test_basic_preprocessing(self):
        """Test basic text preprocessing."""
        input_text = "Hello World! This is a test."
        result = preprocess_comment(input_text)
        
        # Should be lowercase
        assert result == result.lower()
        
        # Should not contain punctuation
        assert "!" not in result
        assert "." not in result
        
        # Should contain the main words
        assert "hello" in result
        assert "world" in result
        assert "test" in result

    def test_stopword_removal(self):
        """Test removal of stopwords while keeping important ones."""
        input_text = "This is a great video but it has some issues"
        result = preprocess_comment(input_text)
        # Should remove common stopwords but keep important words
        assert "this" not in result
        # Note: "is" becomes part of "issue" after lemmatization, so we check for the word "is" as a separate word
        assert " is " not in f" {result} "
        # Check for the word "a" as a separate word, not as a letter in other words
        assert " a " not in f" {result} "
        assert "great" in result  # Important word for sentiment
        assert "video" in result  # Important word for sentiment

    def test_numbers_and_punctuation(self):
        """Test handling of numbers and punctuation."""
        input_text = "Video #123 has 5 stars! Amazing."
        result = preprocess_comment(input_text)
        
        # Should not contain numbers
        assert "123" not in result
        assert "5" not in result
        
        # Should not contain punctuation
        assert "#" not in result
        assert "!" not in result
        assert "." not in result
        
        # Should contain the main words (lemmatized)
        assert "video" in result
        assert "star" in result  # lemmatized from 'stars'
        assert "amazing" in result

    def test_special_characters(self):
        """Test handling of special characters."""
        input_text = "This video is @awesome! 🎬📱💻"
        result = preprocess_comment(input_text)
        
        # Should not contain special characters
        assert "@" not in result
        assert "🎬" not in result
        assert "📱" not in result
        assert "💻" not in result
        
        # Should contain the main words
        assert "video" in result
        assert "awesome" in result

    def test_whitespace_handling(self):
        """Test handling of whitespace and newlines."""
        input_text = "  This   video\n\tis   great!  "
        result = preprocess_comment(input_text)
        
        # Should not have extra whitespace
        assert "  " not in result  # No double spaces
        assert "\n" not in result  # No newlines
        assert "\t" not in result  # No tabs
        
        # Should contain the main words
        assert "video" in result
        assert "great" in result

    def test_empty_input(self):
        """Test handling of empty input."""
        result = preprocess_comment("")
        assert result == ""
        
        result = preprocess_comment("   ")
        assert result == ""

    def test_case_sensitivity(self):
        """Test case sensitivity handling."""
        input_text = "HELLO World hello WORLD"
        result = preprocess_comment(input_text)
        
        # Should be lowercase
        assert result == result.lower()
        
        # Should contain the words
        assert "hello" in result
        assert "world" in result

    def test_lemmatization(self):
        """Test lemmatization of words."""
        input_text = "I am running and watching videos"
        result = preprocess_comment(input_text)
        
        # Should contain lemmatized forms
        assert "run" in result or "running" in result
        assert "watch" in result or "watching" in result
        assert "video" in result

    def test_long_text(self):
        """Test handling of long text."""
        long_text = "This is a very long comment " * 100
        result = preprocess_comment(long_text)
        
        # Should not be empty
        assert len(result) > 0
        
        # Should contain some words
        assert "comment" in result

    def test_mixed_language(self):
        """Test handling of mixed language text."""
        input_text = "This video is 测试文本 and مرحبا بالعالم"
        result = preprocess_comment(input_text)
        
        # Should contain English words
        assert "video" in result
        
        # Should handle non-English characters gracefully
        assert len(result) > 0

    def test_exception_handling(self):
        """Test that exceptions are handled gracefully."""
        # Test with None input
        with pytest.raises(ValueError):
            preprocess_comment(None)
        
        # Test with non-string input
        with pytest.raises(ValueError):
            preprocess_comment(123)
        
        # Test with list input
        with pytest.raises(ValueError):
            preprocess_comment(["test"])

    def test_unicode_handling(self):
        """Test handling of unicode characters."""
        input_text = "This video is 🎬📱💻 and has émojis"
        result = preprocess_comment(input_text)
        
        # Should handle unicode gracefully
        assert len(result) > 0
        
        # Should contain English words
        assert "video" in result

    def test_repeated_words(self):
        """Test handling of repeated words."""
        input_text = "This video video video is great great great"
        result = preprocess_comment(input_text)
        
        # Should handle repeated words gracefully
        assert "video" in result
        assert "great" in result

    def test_contractions(self):
        """Test handling of contractions."""
        input_text = "This video isn't great, it's terrible"
        result = preprocess_comment(input_text)
        
        # Should handle contractions
        assert "video" in result
        assert "great" in result
        assert "terrible" in result 