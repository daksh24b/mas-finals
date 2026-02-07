"""
Logging utilities for Digital Truth Guardian.

Provides structured logging with agent-specific context.
"""

import logging
import sys
from datetime import datetime
from typing import Optional
from functools import lru_cache

from ..core.config import settings


class AgentLogFormatter(logging.Formatter):
    """Custom formatter for agent-specific logging."""
    
    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m"
    }
    
    AGENT_COLORS = {
        "Planner": "\033[94m",    # Light Blue
        "Retriever": "\033[96m",  # Light Cyan
        "Executor": "\033[93m",   # Light Yellow
        "Critic": "\033[95m",     # Light Magenta
        "Archivist": "\033[92m",  # Light Green
    }
    
    def __init__(self, use_colors: bool = True):
        super().__init__()
        self.use_colors = use_colors
    
    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        
        # Get agent name if present
        agent_name = getattr(record, "agent", "System")
        
        if self.use_colors:
            level_color = self.COLORS.get(record.levelname, "")
            agent_color = self.AGENT_COLORS.get(agent_name, "\033[37m")
            reset = self.COLORS["RESET"]
            
            return (
                f"{timestamp} | "
                f"{level_color}{record.levelname:8s}{reset} | "
                f"{agent_color}[{agent_name:10s}]{reset} | "
                f"{record.getMessage()}"
            )
        else:
            return (
                f"{timestamp} | {record.levelname:8s} | "
                f"[{agent_name:10s}] | {record.getMessage()}"
            )


class AgentLogger(logging.Logger):
    """Extended logger with agent context support."""
    
    def __init__(self, name: str, level: int = logging.NOTSET):
        super().__init__(name, level)
        self._agent_name = "System"
    
    def with_agent(self, agent_name: str) -> "AgentLogger":
        """Create a logger context with agent name."""
        self._agent_name = agent_name
        return self
    
    def _log(self, level, msg, args, exc_info=None, extra=None, **kwargs):
        if extra is None:
            extra = {}
        extra["agent"] = self._agent_name
        super()._log(level, msg, args, exc_info, extra, **kwargs)


def setup_logger(
    name: str = "truth_guardian",
    level: Optional[str] = None,
    use_colors: bool = True
) -> AgentLogger:
    """
    Set up and configure the application logger.
    
    Args:
        name: Logger name
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        use_colors: Whether to use colored output
        
    Returns:
        Configured AgentLogger instance
    """
    # Register custom logger class
    logging.setLoggerClass(AgentLogger)
    
    logger = logging.getLogger(name)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Set level
    log_level = level or settings.log_level
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(AgentLogFormatter(use_colors=use_colors))
    logger.addHandler(console_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger


@lru_cache()
def get_logger(name: str = "truth_guardian") -> AgentLogger:
    """
    Get or create a cached logger instance.
    
    Args:
        name: Logger name
        
    Returns:
        AgentLogger instance
    """
    return setup_logger(name)


# ==================== Logging Decorators ====================

def log_agent_action(agent_name: str):
    """Decorator to log agent action entry and exit."""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            logger = get_logger().with_agent(agent_name)
            logger.info(f"Starting action: {func.__name__}")
            try:
                result = await func(*args, **kwargs)
                logger.info(f"Completed action: {func.__name__}")
                return result
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {str(e)}")
                raise
        
        def sync_wrapper(*args, **kwargs):
            logger = get_logger().with_agent(agent_name)
            logger.info(f"Starting action: {func.__name__}")
            try:
                result = func(*args, **kwargs)
                logger.info(f"Completed action: {func.__name__}")
                return result
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {str(e)}")
                raise
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator
