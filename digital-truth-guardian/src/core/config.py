"""
Configuration management for Digital Truth Guardian.

Handles environment variables, API keys, and system configuration.
Uses Pydantic Settings for validation and type safety.
"""

import os
from functools import lru_cache
from typing import Optional, List
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Config(BaseSettings):
    """
    Application configuration with environment variable support.
    
    All settings can be overridden via environment variables with the
    TRUTH_GUARDIAN_ prefix (e.g., TRUTH_GUARDIAN_GEMINI_API_KEY).
    """
    
    # ==================== API Keys ====================
    gemini_api_key: str = Field(
        default="",
        description="Google Gemini API key for LLM operations"
    )
    tavily_api_key: str = Field(
        default="",
        description="Tavily AI API key for web search"
    )
    qdrant_api_key: Optional[str] = Field(
        default=None,
        description="Qdrant Cloud API key (optional for local)"
    )
    
    # ==================== Qdrant Settings ====================
    qdrant_url: str = Field(
        default="http://localhost:6333",
        description="Qdrant server URL"
    )
    qdrant_collection_name: str = Field(
        default="knowledge_base",
        description="Main knowledge base collection name"
    )
    qdrant_timeout: int = Field(
        default=30,
        description="Qdrant operation timeout in seconds"
    )
    
    # ==================== Embedding Settings ====================
    dense_embedding_model: str = Field(
        default="gemini-embedding-001",
        description="Google dense embedding model"
    )
    dense_embedding_dim: int = Field(
        default=3072,
        description="Dense embedding vector dimension"
    )
    sparse_embedding_model: str = Field(
        default="prithivida/Splade_PP_en_v1",
        description="FastEmbed sparse model for keyword matching"
    )
    
    # ==================== LLM Settings ====================
    gemini_pro_model: str = Field(
        default="gemini-3-flash-preview",
        description="Gemini Pro model for deep reasoning (Critic)"
    )
    gemini_flash_model: str = Field(
        default="gemini-3-flash-preview",
        description="Gemini Flash model for speed (Planner, Archivist)"
    )
    llm_temperature: float = Field(
        default=0.1,
        description="LLM temperature for consistent outputs"
    )
    llm_max_tokens: int = Field(
        default=4096,
        description="Maximum tokens for LLM responses"
    )
    
    # ==================== Search Settings ====================
    tavily_max_results: int = Field(
        default=10,
        description="Maximum results from Tavily search"
    )
    retriever_top_k: int = Field(
        default=5,
        description="Number of top results to retrieve from Qdrant"
    )
    hybrid_search_alpha: float = Field(
        default=0.7,
        description="Weight for dense vs sparse (0=sparse, 1=dense)"
    )
    confidence_threshold: float = Field(
        default=0.75,
        description="Minimum confidence score for retrieval"
    )
    
    # ==================== Safety Settings ====================
    trusted_sources_path: str = Field(
        default="data/trusted_sources.json",
        description="Path to trusted sources JSON file"
    )
    max_feedback_loops: int = Field(
        default=3,
        description="Maximum feedback loops before escalation"
    )
    enable_memory_write: bool = Field(
        default=True,
        description="Enable writing to knowledge base"
    )
    min_source_tier_for_memory: int = Field(
        default=4,
        description="Minimum source tier required for memory write (1=highest)"
    )
    
    # ==================== System Settings ====================
    log_level: str = Field(
        default="INFO",
        description="Logging level"
    )
    debug_mode: bool = Field(
        default=False,
        description="Enable debug mode with verbose logging"
    )
    async_batch_size: int = Field(
        default=10,
        description="Batch size for async operations"
    )
    
    model_config = {
        "env_prefix": "TRUTH_GUARDIAN_",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore"
    }
    
    @field_validator("gemini_api_key", "tavily_api_key")
    @classmethod
    def validate_required_keys(cls, v: str, info) -> str:
        """Validate that required API keys are provided."""
        if not v and info.field_name in ["gemini_api_key"]:
            # Allow empty for testing, but warn
            import warnings
            warnings.warn(f"{info.field_name} is not set. Some features may not work.")
        return v
    
    @field_validator("hybrid_search_alpha")
    @classmethod
    def validate_alpha(cls, v: float) -> float:
        """Validate alpha is between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError("hybrid_search_alpha must be between 0 and 1")
        return v
    
    def get_qdrant_config(self) -> dict:
        """Get Qdrant connection configuration."""
        config = {
            "url": self.qdrant_url,
            "timeout": self.qdrant_timeout,
        }
        if self.qdrant_api_key:
            config["api_key"] = self.qdrant_api_key
        return config
    
    def get_embedding_config(self) -> dict:
        """Get embedding configuration."""
        return {
            "dense_model": self.dense_embedding_model,
            "dense_dim": self.dense_embedding_dim,
            "sparse_model": self.sparse_embedding_model,
        }


@lru_cache()
def get_settings() -> Config:
    """
    Get cached application settings.
    
    Returns:
        Config: Application configuration instance
    """
    return Config()


# Global settings instance
settings = get_settings()


# ==================== Model Configuration ====================

class ModelConfig:
    """Configuration for different agent models."""
    
    PLANNER = {
        "model": settings.gemini_flash_model,
        "temperature": 0.1,
        "max_tokens": 2048,
        "purpose": "Intent classification and task decomposition"
    }
    
    CRITIC = {
        "model": settings.gemini_pro_model,
        "temperature": 0.0,
        "max_tokens": 4096,
        "purpose": "Deep reasoning and entailment checking"
    }
    
    ARCHIVIST = {
        "model": settings.gemini_flash_model,
        "temperature": 0.0,
        "max_tokens": 2048,
        "purpose": "Memory management and categorization"
    }


# ==================== Collection Configuration ====================

class CollectionConfig:
    """Configuration for Qdrant collections."""
    
    KNOWLEDGE_BASE = {
        "name": settings.qdrant_collection_name,
        "dense_vector_name": "dense",
        "sparse_vector_name": "sparse",
        "dense_dim": settings.dense_embedding_dim,
        "distance": "Cosine",
        "quantization": {
            "binary": {
                "always_ram": True
            }
        },
        "on_disk": True,
        "hnsw_config": {
            "m": 16,
            "ef_construct": 100
        }
    }
