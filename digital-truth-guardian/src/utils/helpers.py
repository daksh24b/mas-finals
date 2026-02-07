"""
Helper utilities for Digital Truth Guardian.

Contains common utility functions used across the application.
"""

import asyncio
import hashlib
import re
from functools import wraps
from typing import Any, Callable, List, Optional, TypeVar
from urllib.parse import urlparse

T = TypeVar("T")


def async_retry(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,)
) -> Callable:
    """
    Decorator for async functions with retry logic.
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff: Multiplier for delay after each retry
        exceptions: Tuple of exceptions to catch and retry
        
    Returns:
        Decorated function with retry logic
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            current_delay = delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff
            
            raise last_exception
        
        return wrapper
    return decorator


def sync_retry(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,)
) -> Callable:
    """
    Decorator for sync functions with retry logic.
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            import time
            current_delay = delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        time.sleep(current_delay)
                        current_delay *= backoff
            
            raise last_exception
        
        return wrapper
    return decorator


def sanitize_text(text: str) -> str:
    """
    Sanitize text for safe storage and processing.
    
    Args:
        text: Input text to sanitize
        
    Returns:
        Sanitized text string
    """
    if not text:
        return ""
    
    # Remove null bytes
    text = text.replace("\x00", "")
    
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)
    
    # Remove control characters except newlines and tabs
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def extract_domain(url: str) -> str:
    """
    Extract the domain from a URL.
    
    Args:
        url: Full URL string
        
    Returns:
        Domain string (e.g., "reuters.com")
    """
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        
        # Remove www. prefix
        if domain.startswith("www."):
            domain = domain[4:]
        
        return domain
    except Exception:
        return ""


def generate_content_hash(content: str) -> str:
    """
    Generate a hash for content deduplication.
    
    Args:
        content: Text content to hash
        
    Returns:
        SHA256 hash string
    """
    normalized = sanitize_text(content.lower())
    return hashlib.sha256(normalized.encode()).hexdigest()


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 200
) -> List[str]:
    """
    Split text into overlapping chunks.
    
    Args:
        text: Text to chunk
        chunk_size: Maximum size of each chunk
        overlap: Number of characters to overlap between chunks
        
    Returns:
        List of text chunks
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Try to find a good break point (sentence end)
        if end < len(text):
            # Look for sentence endings
            for sep in [". ", "! ", "? ", "\n\n", "\n", " "]:
                last_sep = text[start:end].rfind(sep)
                if last_sep > chunk_size // 2:
                    end = start + last_sep + len(sep)
                    break
        
        chunks.append(text[start:end].strip())
        start = end - overlap
    
    return chunks


def parse_multi_query(query: str) -> List[str]:
    """
    Parse a complex query into sub-queries.
    
    Splits on common conjunctions and separators.
    
    Args:
        query: Complex query string
        
    Returns:
        List of sub-query strings
    """
    # Common separators for multi-part queries
    separators = [
        r"\s+AND\s+",
        r"\s+and\s+",
        r"\s*,\s*and\s+",
        r"\s*;\s*",
        r"\s+also\s+",
        r"\s+additionally\s+",
    ]
    
    # Try each separator
    for sep in separators:
        parts = re.split(sep, query, flags=re.IGNORECASE)
        if len(parts) > 1:
            # Clean up each part
            return [sanitize_text(p) for p in parts if p.strip()]
    
    return [query]


def format_confidence(confidence: float) -> str:
    """
    Format confidence score for display.
    
    Args:
        confidence: Float between 0 and 1
        
    Returns:
        Formatted string like "High (92%)"
    """
    percentage = int(confidence * 100)
    
    if confidence >= 0.9:
        level = "Very High"
    elif confidence >= 0.75:
        level = "High"
    elif confidence >= 0.5:
        level = "Medium"
    elif confidence >= 0.25:
        level = "Low"
    else:
        level = "Very Low"
    
    return f"{level} ({percentage}%)"


def truncate_text(text: str, max_length: int = 500, suffix: str = "...") -> str:
    """
    Truncate text to maximum length, preserving word boundaries.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to append if truncated
        
    Returns:
        Truncated text string
    """
    if len(text) <= max_length:
        return text
    
    truncated = text[:max_length - len(suffix)]
    
    # Find last space to avoid cutting words
    last_space = truncated.rfind(" ")
    if last_space > max_length // 2:
        truncated = truncated[:last_space]
    
    return truncated + suffix


def merge_dicts(base: dict, updates: dict) -> dict:
    """
    Recursively merge two dictionaries.
    
    Args:
        base: Base dictionary
        updates: Dictionary with updates to merge
        
    Returns:
        Merged dictionary
    """
    result = base.copy()
    
    for key, value in updates.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    
    return result


async def gather_with_concurrency(
    tasks: List[Callable],
    max_concurrent: int = 5
) -> List[Any]:
    """
    Execute async tasks with limited concurrency.
    
    Args:
        tasks: List of async callables
        max_concurrent: Maximum concurrent tasks
        
    Returns:
        List of results
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def limited_task(task):
        async with semaphore:
            return await task()
    
    return await asyncio.gather(*[limited_task(t) for t in tasks])
