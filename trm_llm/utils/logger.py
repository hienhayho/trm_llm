"""Global logger for TRM-LLM using structlog

Provides rank-aware logging that only outputs on the main process (rank 0)
when running with DDP (Distributed Data Parallel) or DeepSpeed.

Usage:
    from trm_llm.utils.logger import log

    log("Training started")  # Simple logging
"""

import logging
import os
import sys
import structlog
import torch.distributed as dist
from typing import Optional


# Global state for rank awareness
_is_main_process: Optional[bool] = None
_logger_configured: bool = False


def is_main_process() -> bool:
    """Check if this is the main process (rank 0)

    Returns True if:
    - Not using distributed training
    - Using distributed training and rank == 0

    Also checks environment variables (RANK, LOCAL_RANK) for DeepSpeed
    which may set these before dist.init_process_group is called.
    """
    global _is_main_process

    # Use cached value if available
    if _is_main_process is not None:
        return _is_main_process

    # Check distributed state first
    if dist.is_initialized():
        _is_main_process = dist.get_rank() == 0
        return _is_main_process

    # Check environment variables (for DeepSpeed/torchrun before dist init)
    rank_env = os.environ.get("RANK")
    local_rank_env = os.environ.get("LOCAL_RANK")

    if rank_env is not None:
        _is_main_process = int(rank_env) == 0
    elif local_rank_env is not None:
        _is_main_process = int(local_rank_env) == 0
    else:
        # No distributed training detected
        _is_main_process = True

    return _is_main_process


def reset_main_process_cache():
    """Reset the main process cache (call after DDP init)"""
    global _is_main_process
    _is_main_process = None


def _filter_by_rank(logger, method_name, event_dict):
    """Structlog processor that filters logs based on rank"""
    if not is_main_process():
        raise structlog.DropEvent
    return event_dict


def configure_logger(
    level: str = "INFO",
    json_output: bool = False,
):
    """Configure the global structlog logger

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        json_output: If True, output JSON logs (useful for production)
    """
    global _logger_configured

    if _logger_configured:
        return

    # Convert level string to logging constant
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Build processor chain
    processors = [
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S"),
        _filter_by_rank,  # Filter out non-main process logs
    ]

    # Add final renderer
    if json_output:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer(
            colors=True,
            exception_formatter=structlog.dev.plain_traceback,
        ))

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )

    _logger_configured = True


def get_logger(name: Optional[str] = None) -> structlog.BoundLogger:
    """Get a logger instance

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured structlog logger
    """
    # Auto-configure if not done
    if not _logger_configured:
        configure_logger()

    return structlog.get_logger(name)


# Simple logging functions for convenience
def log_info(msg: str, **kwargs):
    """Log info message (only on main process)"""
    get_logger().info(msg, **kwargs)


def log_warning(msg: str, **kwargs):
    """Log warning message (only on main process)"""
    get_logger().warning(msg, **kwargs)


def log_error(msg: str, **kwargs):
    """Log error message (only on main process)"""
    get_logger().error(msg, **kwargs)


def log_debug(msg: str, **kwargs):
    """Log debug message (only on main process)"""
    get_logger().debug(msg, **kwargs)


def log(msg: str, **kwargs):
    """Structured log that only logs on main process using structlog

    Args:
        msg: Log message
        **kwargs: Additional structured data to log

    Examples:
        log("Training started")
        log("Loaded model", params="92M", hidden_dim=768, frozen=True)
    """
    get_logger().info(msg, **kwargs)
