"""
Test configuration and fixtures.
"""

import os
import sys
from pathlib import Path

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set test environment variables
os.environ.setdefault("TRUTH_GUARDIAN_GEMINI_API_KEY", "test_key")
os.environ.setdefault("TRUTH_GUARDIAN_TAVILY_API_KEY", "test_key")
os.environ.setdefault("TRUTH_GUARDIAN_QDRANT_URL", "http://localhost:6333")
os.environ.setdefault("TRUTH_GUARDIAN_LOG_LEVEL", "DEBUG")


@pytest.fixture
def sample_claim():
    """Sample claim for testing."""
    return "The Earth is approximately 4.5 billion years old"


@pytest.fixture
def sample_false_claim():
    """Sample false claim for testing."""
    return "The Earth is flat and 6000 years old"


@pytest.fixture
def sample_uncertain_claim():
    """Sample uncertain claim for testing."""
    return "Aliens have visited Earth in the past decade"


@pytest.fixture
def trusted_sources():
    """Sample trusted sources."""
    return [
        "https://www.nasa.gov/article",
        "https://www.bbc.com/news/science",
        "https://www.nature.com/articles/study"
    ]


@pytest.fixture
def untrusted_sources():
    """Sample untrusted sources."""
    return [
        "https://randomsite.xyz/fake-news",
        "https://infowars.com/article",
        "https://unknownblog.wordpress.com/post"
    ]
