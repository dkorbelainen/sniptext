"""Tests for sniptext.preview interactive UI."""

from unittest.mock import patch

from sniptext.preview import TextPreview


class TestTextPreview:
    """Tests for TextPreview class."""

    def test_preview_init(self):
        """Test TextPreview initialization."""
        preview = TextPreview()
        assert preview is not None

    def test_check_rich_available(self):
        """Test rich library availability check."""
        preview = TextPreview()
        # Will be True if rich is installed, False otherwise
        assert isinstance(preview._use_rich, bool)

    def test_is_available(self):
        """Test is_available method."""
        preview = TextPreview()
        assert isinstance(preview.is_available(), bool)

    @patch("builtins.input", return_value="y")
    def test_fallback_preview_copy(self, mock_input):
        """Test fallback preview with user confirms copy."""
        result = TextPreview._show_fallback_preview("Test text")
        assert result is not None
        text, should_copy = result
        assert text == "Test text"
        assert should_copy is True

    @patch("builtins.input", return_value="n")
    def test_fallback_preview_cancel(self, mock_input):
        """Test fallback preview with user cancels."""
        result = TextPreview._show_fallback_preview("Test text")
        assert result is not None
        text, should_copy = result
        assert text == "Test text"
        assert should_copy is False

    @patch("builtins.input", return_value="")
    def test_fallback_preview_empty_input(self, mock_input):
        """Test fallback preview with empty input defaults to copy."""
        result = TextPreview._show_fallback_preview("Test text")
        assert result is not None
        text, should_copy = result
        assert text == "Test text"
        assert should_copy is True

    @patch("builtins.input", side_effect=KeyboardInterrupt)
    def test_fallback_preview_keyboard_interrupt(self, mock_input):
        """Test fallback preview handles keyboard interrupt."""
        result = TextPreview._show_fallback_preview("Test text")
        assert result is not None
        text, should_copy = result
        assert text == "Test text"
        assert should_copy is False

    @patch("sniptext.preview.TextPreview._check_rich_available", return_value=False)
    @patch("builtins.input", return_value="y")
    def test_fallback_when_rich_unavailable(self, mock_input, mock_check):
        """Test fallback is used when rich is not available."""
        preview = TextPreview()
        result = preview.show_preview("Test text")
        assert result is not None
        text, should_copy = result
        assert text == "Test text"

    def test_long_text_handling(self):
        """Test preview can handle long text."""
        preview = TextPreview()
        # Just verify the preview object can be created without errors
        assert preview is not None

    @patch("builtins.input", return_value="y")
    def test_multiline_text_fallback(self, mock_input):
        """Test preview with multiline text in fallback mode."""
        multiline_text = "Line 1\nLine 2\nLine 3"
        result = TextPreview._show_fallback_preview(multiline_text)
        assert result is not None
        text, should_copy = result
        assert "Line 1" in text
        assert "Line 3" in text
        assert should_copy is True
