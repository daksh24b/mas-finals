"""
Tests for utility functions.
"""

import pytest
import asyncio

from src.utils.helpers import (
    sanitize_text,
    extract_domain,
    generate_content_hash,
    chunk_text,
    parse_multi_query,
    format_confidence,
    truncate_text,
    merge_dicts
)


class TestSanitizeText:
    """Tests for text sanitization."""
    
    def test_basic_sanitization(self):
        """Test basic text cleaning."""
        assert sanitize_text("  hello world  ") == "hello world"
        assert sanitize_text("hello\n\n\nworld") == "hello world"
    
    def test_null_byte_removal(self):
        """Test null byte removal."""
        text = "hello\x00world"
        assert "\x00" not in sanitize_text(text)
    
    def test_empty_string(self):
        """Test empty string handling."""
        assert sanitize_text("") == ""
        assert sanitize_text(None) == ""


class TestExtractDomain:
    """Tests for domain extraction."""
    
    def test_basic_domains(self):
        """Test basic domain extraction."""
        assert extract_domain("https://www.example.com/page") == "example.com"
        assert extract_domain("https://example.com/page") == "example.com"
        assert extract_domain("http://subdomain.example.com/page") == "subdomain.example.com"
    
    def test_www_removal(self):
        """Test www prefix removal."""
        assert extract_domain("https://www.nasa.gov/article") == "nasa.gov"
    
    def test_invalid_url(self):
        """Test invalid URL handling."""
        assert extract_domain("not-a-url") == ""
        assert extract_domain("") == ""


class TestGenerateContentHash:
    """Tests for content hashing."""
    
    def test_deterministic(self):
        """Test hash is deterministic."""
        text = "The Earth is round."
        hash1 = generate_content_hash(text)
        hash2 = generate_content_hash(text)
        assert hash1 == hash2
    
    def test_case_insensitive(self):
        """Test hash is case-insensitive."""
        hash1 = generate_content_hash("Hello World")
        hash2 = generate_content_hash("hello world")
        assert hash1 == hash2
    
    def test_whitespace_normalized(self):
        """Test whitespace is normalized."""
        hash1 = generate_content_hash("hello world")
        hash2 = generate_content_hash("hello   world")
        assert hash1 == hash2


class TestChunkText:
    """Tests for text chunking."""
    
    def test_short_text(self):
        """Test short text returns single chunk."""
        text = "Short text"
        chunks = chunk_text(text, chunk_size=100)
        assert len(chunks) == 1
        assert chunks[0] == text
    
    def test_long_text(self):
        """Test long text is split."""
        text = "This is a sentence. " * 100
        chunks = chunk_text(text, chunk_size=100, overlap=20)
        assert len(chunks) > 1
    
    def test_overlap(self):
        """Test chunks have overlap."""
        text = "A" * 200
        chunks = chunk_text(text, chunk_size=100, overlap=20)
        # With overlap, chunks should share some content
        assert len(chunks) >= 2


class TestParseMultiQuery:
    """Tests for multi-query parsing."""
    
    def test_single_query(self):
        """Test single query returns as-is."""
        queries = parse_multi_query("Is the Earth round?")
        assert len(queries) == 1
        assert queries[0] == "Is the Earth round?"
    
    def test_and_separator(self):
        """Test AND separator."""
        queries = parse_multi_query("Is the Earth round AND is the sky blue?")
        assert len(queries) == 2
    
    def test_semicolon_separator(self):
        """Test semicolon separator."""
        queries = parse_multi_query("Question one; Question two")
        assert len(queries) == 2


class TestFormatConfidence:
    """Tests for confidence formatting."""
    
    def test_very_high_confidence(self):
        """Test very high confidence formatting."""
        result = format_confidence(0.95)
        assert "Very High" in result
        assert "95%" in result
    
    def test_high_confidence(self):
        """Test high confidence formatting."""
        result = format_confidence(0.80)
        assert "High" in result
        assert "80%" in result
    
    def test_medium_confidence(self):
        """Test medium confidence formatting."""
        result = format_confidence(0.60)
        assert "Medium" in result
    
    def test_low_confidence(self):
        """Test low confidence formatting."""
        result = format_confidence(0.30)
        assert "Low" in result


class TestTruncateText:
    """Tests for text truncation."""
    
    def test_short_text(self):
        """Test short text is not truncated."""
        text = "Short text"
        result = truncate_text(text, max_length=100)
        assert result == text
    
    def test_long_text(self):
        """Test long text is truncated."""
        text = "A" * 1000
        result = truncate_text(text, max_length=100)
        assert len(result) <= 100
        assert result.endswith("...")
    
    def test_word_boundary(self):
        """Test truncation respects word boundaries."""
        text = "Hello wonderful amazing world of programming"
        result = truncate_text(text, max_length=20)
        # Should not cut in middle of word
        assert not result.endswith("a...")


class TestMergeDicts:
    """Tests for dictionary merging."""
    
    def test_simple_merge(self):
        """Test simple dictionary merge."""
        base = {"a": 1, "b": 2}
        updates = {"b": 3, "c": 4}
        
        result = merge_dicts(base, updates)
        
        assert result["a"] == 1
        assert result["b"] == 3
        assert result["c"] == 4
    
    def test_nested_merge(self):
        """Test nested dictionary merge."""
        base = {"outer": {"inner1": 1, "inner2": 2}}
        updates = {"outer": {"inner2": 3, "inner3": 4}}
        
        result = merge_dicts(base, updates)
        
        assert result["outer"]["inner1"] == 1
        assert result["outer"]["inner2"] == 3
        assert result["outer"]["inner3"] == 4
