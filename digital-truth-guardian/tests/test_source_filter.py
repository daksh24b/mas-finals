"""
Tests for the source filter functionality.
"""

import pytest
from src.tools.source_filter import SourceFilter, TrustTier, TrustedSourceConfig


class TestTrustTier:
    """Tests for TrustTier enum."""
    
    def test_tier_ordering(self):
        """Test that tiers are properly ordered."""
        assert TrustTier.TIER_1_AUTHORITATIVE < TrustTier.TIER_2_MAJOR_NEWS
        assert TrustTier.TIER_2_MAJOR_NEWS < TrustTier.TIER_3_REPUTABLE
        assert TrustTier.TIER_3_REPUTABLE < TrustTier.TIER_4_GENERAL
        assert TrustTier.TIER_4_GENERAL < TrustTier.TIER_5_UNTRUSTED
    
    def test_from_string(self):
        """Test string to TrustTier conversion."""
        assert TrustTier.from_string("authoritative") == TrustTier.TIER_1_AUTHORITATIVE
        assert TrustTier.from_string("MAJOR_NEWS") == TrustTier.TIER_2_MAJOR_NEWS
        assert TrustTier.from_string("unknown") == TrustTier.TIER_5_UNTRUSTED


class TestSourceFilter:
    """Tests for SourceFilter class."""
    
    @pytest.fixture
    def filter(self):
        """Create a source filter with default config."""
        return SourceFilter()
    
    def test_tier_1_domains(self, filter):
        """Test Tier 1 (authoritative) domain detection."""
        tier1_urls = [
            "https://www.nasa.gov/article",
            "https://cdc.gov/health/info",
            "https://harvard.edu/research/study",
            "https://www.mit.edu/news"
        ]
        
        for url in tier1_urls:
            tier = filter.get_trust_tier(url)
            assert tier == TrustTier.TIER_1_AUTHORITATIVE, f"Failed for {url}"
    
    def test_tier_1_suffixes(self, filter):
        """Test Tier 1 by suffix (e.g., .gov, .edu)."""
        suffix_urls = [
            "https://anysite.gov/page",
            "https://university.edu/research",
            "https://military.mil/news"
        ]
        
        for url in suffix_urls:
            tier = filter.get_trust_tier(url)
            assert tier == TrustTier.TIER_1_AUTHORITATIVE, f"Failed for {url}"
    
    def test_tier_2_domains(self, filter):
        """Test Tier 2 (major news) domain detection."""
        tier2_urls = [
            "https://www.reuters.com/article",
            "https://www.bbc.com/news/world",
            "https://www.nytimes.com/article",
            "https://snopes.com/fact-check/claim"
        ]
        
        for url in tier2_urls:
            tier = filter.get_trust_tier(url)
            assert tier == TrustTier.TIER_2_MAJOR_NEWS, f"Failed for {url}"
    
    def test_tier_3_domains(self, filter):
        """Test Tier 3 (reputable) domain detection."""
        tier3_urls = [
            "https://www.wired.com/article",
            "https://arstechnica.com/science",
            "https://www.theatlantic.com/ideas"
        ]
        
        for url in tier3_urls:
            tier = filter.get_trust_tier(url)
            assert tier == TrustTier.TIER_3_REPUTABLE, f"Failed for {url}"
    
    def test_unknown_domains(self, filter):
        """Test unknown domains default to Tier 4."""
        unknown_urls = [
            "https://randomsite.com/page",
            "https://myblog.xyz/post",
            "https://unknownnews.net/article"
        ]
        
        for url in unknown_urls:
            tier = filter.get_trust_tier(url)
            assert tier == TrustTier.TIER_4_GENERAL, f"Failed for {url}"
    
    def test_blocked_domains(self, filter):
        """Test blocked domains are marked as untrusted."""
        blocked_urls = [
            "https://infowars.com/article",
            "https://naturalnews.com/health",
            "https://theonion.com/satire"  # Satire sites
        ]
        
        for url in blocked_urls:
            tier = filter.get_trust_tier(url)
            assert tier == TrustTier.TIER_5_UNTRUSTED, f"Failed for {url}"
    
    def test_is_trusted(self, filter):
        """Test is_trusted method."""
        assert filter.is_trusted("https://nasa.gov/article") == True
        assert filter.is_trusted("https://bbc.com/news") == True
        assert filter.is_trusted("https://wired.com/tech") == True
        assert filter.is_trusted("https://randomsite.com/page") == False
    
    def test_is_trusted_with_threshold(self, filter):
        """Test is_trusted with different thresholds."""
        url = "https://bbc.com/news"  # Tier 2
        
        assert filter.is_trusted(url, max_tier=TrustTier.TIER_1_AUTHORITATIVE) == False
        assert filter.is_trusted(url, max_tier=TrustTier.TIER_2_MAJOR_NEWS) == True
        assert filter.is_trusted(url, max_tier=TrustTier.TIER_3_REPUTABLE) == True
    
    def test_filter_results(self, filter):
        """Test filtering a list of search results."""
        results = [
            {"url": "https://nasa.gov/article", "title": "NASA"},
            {"url": "https://bbc.com/news", "title": "BBC"},
            {"url": "https://randomsite.com/page", "title": "Random"},
            {"url": "https://infowars.com/fake", "title": "Blocked"}
        ]
        
        filtered = filter.filter_results(results)
        
        # Should keep NASA, BBC; filter Random, Blocked
        assert len(filtered) == 2
        assert filtered[0]["_domain"] == "nasa.gov"
        assert filtered[1]["_domain"] == "bbc.com"
    
    def test_should_memorize(self, filter):
        """Test should_memorize method."""
        # High-tier sources should allow memorization
        high_tier_sources = ["https://nasa.gov/article", "https://bbc.com/news"]
        assert filter.should_memorize(high_tier_sources) == True
        
        # Low-tier sources should not allow memorization
        low_tier_sources = ["https://randomsite.com/page", "https://myblog.xyz/post"]
        assert filter.should_memorize(low_tier_sources) == False
    
    def test_get_source_summary(self, filter):
        """Test source summary generation."""
        sources = [
            "https://nasa.gov/article",
            "https://bbc.com/news",
            "https://randomsite.com/page"
        ]
        
        summary = filter.get_source_summary(sources)
        
        assert summary["overall_quality"] == "high"  # Best is Tier 1
        assert summary["best_tier"] == 1
        assert "TIER_1_AUTHORITATIVE" in summary["tier_counts"]


class TestEdgeCases:
    """Edge case tests."""
    
    def test_empty_url(self):
        """Test handling of empty URL."""
        filter = SourceFilter()
        tier = filter.get_trust_tier("")
        assert tier == TrustTier.TIER_5_UNTRUSTED
    
    def test_invalid_url(self):
        """Test handling of invalid URL."""
        filter = SourceFilter()
        tier = filter.get_trust_tier("not-a-url")
        assert tier == TrustTier.TIER_5_UNTRUSTED
    
    def test_www_prefix_handling(self):
        """Test that www. prefix is handled correctly."""
        filter = SourceFilter()
        
        # Both should resolve to same domain
        tier1 = filter.get_trust_tier("https://www.nasa.gov/article")
        tier2 = filter.get_trust_tier("https://nasa.gov/article")
        
        assert tier1 == tier2 == TrustTier.TIER_1_AUTHORITATIVE
