"""
Source filtering and trust tier management.

Implements the "Trusted Source Protocol" for filtering external
search results based on source credibility.
"""

import json
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Dict, List, Optional, Set
from functools import lru_cache

from ..core.config import settings
from ..utils.logger import get_logger
from ..utils.helpers import extract_domain


logger = get_logger()


class TrustTier(IntEnum):
    """
    Trust tier classification for sources.
    
    Lower tier = higher trust.
    """
    TIER_1_AUTHORITATIVE = 1   # Government, major academic institutions
    TIER_2_MAJOR_NEWS = 2      # Major news outlets, established media
    TIER_3_REPUTABLE = 3       # Reputable news, fact-checking orgs
    TIER_4_GENERAL = 4         # General web sources
    TIER_5_UNTRUSTED = 5       # Unknown/untrusted sources
    
    @classmethod
    def from_string(cls, s: str) -> "TrustTier":
        """Convert string to TrustTier."""
        mapping = {
            "authoritative": cls.TIER_1_AUTHORITATIVE,
            "major_news": cls.TIER_2_MAJOR_NEWS,
            "reputable": cls.TIER_3_REPUTABLE,
            "general": cls.TIER_4_GENERAL,
            "untrusted": cls.TIER_5_UNTRUSTED
        }
        return mapping.get(s.lower(), cls.TIER_5_UNTRUSTED)


@dataclass
class TrustedSourceConfig:
    """Configuration for trusted sources."""
    
    # Tier 1: Authoritative sources (government, academic)
    tier_1_domains: Set[str]
    tier_1_suffixes: Set[str]  # e.g., .gov, .edu
    
    # Tier 2: Major news outlets
    tier_2_domains: Set[str]
    
    # Tier 3: Reputable sources
    tier_3_domains: Set[str]
    
    # Blocked domains (always reject)
    blocked_domains: Set[str]
    
    @classmethod
    def from_json(cls, path: str) -> "TrustedSourceConfig":
        """Load configuration from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        
        return cls(
            tier_1_domains=set(data.get("tier_1_domains", [])),
            tier_1_suffixes=set(data.get("tier_1_suffixes", [])),
            tier_2_domains=set(data.get("tier_2_domains", [])),
            tier_3_domains=set(data.get("tier_3_domains", [])),
            blocked_domains=set(data.get("blocked_domains", []))
        )
    
    @classmethod
    def default(cls) -> "TrustedSourceConfig":
        """Create default trusted source configuration."""
        return cls(
            tier_1_domains={
                # Government
                "whitehouse.gov", "usa.gov", "cdc.gov", "nih.gov",
                "fda.gov", "epa.gov", "nasa.gov", "noaa.gov",
                "fbi.gov", "justice.gov", "state.gov", "treasury.gov",
                "who.int", "un.org", "europa.eu",
                "gov.uk", "canada.ca", "australia.gov.au",
                
                # Academic
                "harvard.edu", "mit.edu", "stanford.edu", "yale.edu",
                "princeton.edu", "columbia.edu", "berkeley.edu",
                "oxford.ac.uk", "cam.ac.uk", "nature.com",
                "science.org", "sciencedirect.com", "springer.com",
                "pubmed.ncbi.nlm.nih.gov", "arxiv.org",
            },
            tier_1_suffixes={".gov", ".edu", ".mil", ".ac.uk"},
            tier_2_domains={
                # Major Wire Services
                "reuters.com", "apnews.com", "afp.com",
                
                # Major International News
                "bbc.com", "bbc.co.uk", "cnn.com", "nytimes.com",
                "washingtonpost.com", "theguardian.com", "wsj.com",
                "ft.com", "economist.com", "bloomberg.com",
                "npr.org", "pbs.org", "abc.net.au",
                "dw.com", "france24.com", "aljazeera.com",
                
                # Fact-Checking Organizations
                "factcheck.org", "politifact.com", "snopes.com",
                "fullfact.org", "checkyourfact.com",
            },
            tier_3_domains={
                # Reputable Tech/Science News
                "wired.com", "arstechnica.com", "theverge.com",
                "techcrunch.com", "engadget.com", "cnet.com",
                "scientificamerican.com", "newscientist.com",
                "nationalgeographic.com", "smithsonianmag.com",
                
                # Reputable General News
                "usatoday.com", "latimes.com", "chicagotribune.com",
                "bostonglobe.com", "sfchronicle.com", "seattletimes.com",
                "time.com", "newsweek.com", "theatlantic.com",
                "newyorker.com", "slate.com", "vox.com",
                "axios.com", "politico.com", "thehill.com",
            },
            blocked_domains={
                # Known misinformation sites (examples)
                "infowars.com", "naturalnews.com", "beforeitsnews.com",
                # Satire (should not be used as sources)
                "theonion.com", "babylonbee.com", "clickhole.com",
            }
        )


class SourceFilter:
    """
    Filter for evaluating and classifying source trustworthiness.
    
    Implements the "Allow-List" protocol for external search results.
    """
    
    def __init__(self, config: TrustedSourceConfig = None, config_path: str = None):
        """
        Initialize source filter.
        
        Args:
            config: Pre-loaded configuration
            config_path: Path to JSON configuration file
        """
        if config:
            self.config = config
        elif config_path and Path(config_path).exists():
            self.config = TrustedSourceConfig.from_json(config_path)
        else:
            self.config = TrustedSourceConfig.default()
        
        logger.with_agent("Executor").info(
            f"Initialized source filter with {len(self._all_trusted_domains())} trusted domains"
        )
    
    def _all_trusted_domains(self) -> Set[str]:
        """Get all trusted domains across all tiers."""
        return (
            self.config.tier_1_domains |
            self.config.tier_2_domains |
            self.config.tier_3_domains
        )
    
    def get_trust_tier(self, url: str) -> TrustTier:
        """
        Determine the trust tier for a URL.
        
        Args:
            url: URL to evaluate
            
        Returns:
            TrustTier classification
        """
        domain = extract_domain(url)
        
        if not domain:
            return TrustTier.TIER_5_UNTRUSTED
        
        # Check blocked list first
        if domain in self.config.blocked_domains:
            return TrustTier.TIER_5_UNTRUSTED
        
        # Check Tier 1 (exact match or suffix)
        if domain in self.config.tier_1_domains:
            return TrustTier.TIER_1_AUTHORITATIVE
        
        for suffix in self.config.tier_1_suffixes:
            if domain.endswith(suffix):
                return TrustTier.TIER_1_AUTHORITATIVE
        
        # Check Tier 2
        if domain in self.config.tier_2_domains:
            return TrustTier.TIER_2_MAJOR_NEWS
        
        # Check Tier 3
        if domain in self.config.tier_3_domains:
            return TrustTier.TIER_3_REPUTABLE
        
        # Unknown source
        return TrustTier.TIER_4_GENERAL
    
    def is_trusted(self, url: str, max_tier: TrustTier = TrustTier.TIER_3_REPUTABLE) -> bool:
        """
        Check if a URL meets the trust threshold.
        
        Args:
            url: URL to check
            max_tier: Maximum acceptable tier (lower = more restrictive)
            
        Returns:
            True if source is trusted at or above the threshold
        """
        tier = self.get_trust_tier(url)
        return tier <= max_tier
    
    def is_blocked(self, url: str) -> bool:
        """Check if a URL is on the blocked list."""
        domain = extract_domain(url)
        return domain in self.config.blocked_domains
    
    def filter_results(
        self,
        results: List[Dict],
        max_tier: TrustTier = TrustTier.TIER_3_REPUTABLE,
        url_key: str = "url"
    ) -> List[Dict]:
        """
        Filter a list of search results by trust tier.
        
        Args:
            results: List of result dictionaries
            max_tier: Maximum acceptable tier
            url_key: Key in result dict containing the URL
            
        Returns:
            Filtered list of trusted results
        """
        trusted = []
        filtered_count = 0
        
        for result in results:
            url = result.get(url_key, "")
            tier = self.get_trust_tier(url)
            
            if tier <= max_tier:
                result["_trust_tier"] = tier.value
                result["_domain"] = extract_domain(url)
                trusted.append(result)
            else:
                filtered_count += 1
        
        logger.with_agent("Executor").info(
            f"Source filter: kept {len(trusted)}, filtered {filtered_count} results"
        )
        
        return trusted
    
    def rank_by_trust(self, results: List[Dict], url_key: str = "url") -> List[Dict]:
        """
        Sort results by trust tier (most trusted first).
        
        Args:
            results: List of result dictionaries
            url_key: Key in result dict containing the URL
            
        Returns:
            Sorted list of results
        """
        for result in results:
            if "_trust_tier" not in result:
                result["_trust_tier"] = self.get_trust_tier(
                    result.get(url_key, "")
                ).value
        
        return sorted(results, key=lambda r: r["_trust_tier"])
    
    def should_memorize(
        self,
        sources: List[str],
        min_tier: TrustTier = None
    ) -> bool:
        """
        Determine if evidence from these sources should be memorized.
        
        Implements Safety Filter 3: Only memorize from high-tier sources.
        
        Args:
            sources: List of source URLs/domains
            min_tier: Minimum tier required for memorization
            
        Returns:
            True if at least one source meets the threshold
        """
        min_tier = min_tier or TrustTier(settings.min_source_tier_for_memory)
        
        for source in sources:
            tier = self.get_trust_tier(source)
            if tier <= min_tier:
                return True
        
        return False
    
    def get_source_summary(self, sources: List[str]) -> Dict[str, any]:
        """
        Generate a summary of source trustworthiness.
        
        Args:
            sources: List of source URLs
            
        Returns:
            Dictionary with source analysis
        """
        tiers = [self.get_trust_tier(s) for s in sources]
        
        if not tiers:
            return {
                "overall_quality": "unknown",
                "best_tier": None,
                "tier_counts": {},
                "recommendation": "No sources available"
            }
        
        best_tier = min(tiers)
        tier_counts = {}
        for tier in tiers:
            tier_name = tier.name
            tier_counts[tier_name] = tier_counts.get(tier_name, 0) + 1
        
        # Determine overall quality
        if best_tier <= TrustTier.TIER_1_AUTHORITATIVE:
            quality = "high"
            recommendation = "Authoritative sources available - high confidence"
        elif best_tier <= TrustTier.TIER_2_MAJOR_NEWS:
            quality = "good"
            recommendation = "Major news sources available - good confidence"
        elif best_tier <= TrustTier.TIER_4_GENERAL:
            quality = "moderate"
            recommendation = "Reputable sources available - moderate confidence"
        else:
            quality = "low"
            recommendation = "Only general/unknown sources - low confidence"
        
        return {
            "overall_quality": quality,
            "best_tier": best_tier.value,
            "tier_counts": tier_counts,
            "recommendation": recommendation
        }


# ==================== Cached Instance ====================

@lru_cache()
def get_source_filter() -> SourceFilter:
    """Get cached source filter instance."""
    config_path = Path(settings.trusted_sources_path)
    if config_path.exists():
        return SourceFilter(config_path=str(config_path))
    return SourceFilter()
