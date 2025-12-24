"""Global logger for TRM-LLM using structlog

Provides rank-aware logging that only outputs on the main process (rank 0)
when running with DDP (Distributed Data Parallel) or DeepSpeed.

Usage:
    from trm_llm.utils.logger import log, setup_file_logging

    setup_file_logging("checkpoints/train.log")  # Enable file logging
    log("Training started")  # Simple logging
"""

import logging
import os
import re
import sys
import structlog
import torch.distributed as dist
from datetime import datetime
from typing import Optional, TextIO


# Global state for rank awareness
_is_main_process: Optional[bool] = None
_logger_configured: bool = False
_log_file: Optional[TextIO] = None
_log_file_path: Optional[str] = None
_original_stdout: Optional[TextIO] = None
_original_stderr: Optional[TextIO] = None

# Regex to strip ANSI escape codes
_ansi_escape_pattern = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')


class TeeOutput:
    """Tee output to both terminal and file, stripping ANSI codes for file"""

    def __init__(self, original_stream: TextIO, log_file: TextIO, stream_name: str = "stdout"):
        self.original_stream = original_stream
        self.log_file = log_file
        self.stream_name = stream_name

    def write(self, message: str) -> int:
        # Write to original stream (terminal) with colors
        if self.original_stream is not None:
            try:
                self.original_stream.write(message)
                self.original_stream.flush()
            except (IOError, ValueError):
                pass

        # Write to file without ANSI codes
        if self.log_file is not None and message:
            try:
                clean_message = _ansi_escape_pattern.sub('', message)
                self.log_file.write(clean_message)
                self.log_file.flush()
            except (IOError, ValueError):
                pass

        return len(message)

    def flush(self) -> None:
        if self.original_stream is not None:
            try:
                self.original_stream.flush()
            except (IOError, ValueError):
                pass
        if self.log_file is not None:
            try:
                self.log_file.flush()
            except (IOError, ValueError):
                pass

    def fileno(self) -> int:
        """Return file descriptor for compatibility"""
        if self.original_stream is not None:
            return self.original_stream.fileno()
        return -1

    def isatty(self) -> bool:
        """Check if stream is a TTY (for tqdm compatibility)"""
        if self.original_stream is not None:
            try:
                return self.original_stream.isatty()
            except (IOError, ValueError):
                pass
        return False

    # Additional methods for compatibility
    def readable(self) -> bool:
        return False

    def writable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return False


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


def setup_file_logging(log_path: str) -> str:
    """Setup file logging - captures ALL terminal output (stdout/stderr)

    Args:
        log_path: Path to log file (will be created if doesn't exist)

    Returns:
        Actual log file path used
    """
    global _log_file, _log_file_path, _logger_configured
    global _original_stdout, _original_stderr

    # Only main process should write logs
    if not is_main_process():
        return log_path

    # Close existing log file if any
    if _log_file is not None:
        _log_file.close()

    # Restore original streams if they were redirected
    if _original_stdout is not None:
        sys.stdout = _original_stdout
    if _original_stderr is not None:
        sys.stderr = _original_stderr

    # Create directory if needed
    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    # Open log file in append mode
    _log_file = open(log_path, "a", encoding="utf-8")
    _log_file_path = log_path

    # Write header
    _log_file.write(f"\n{'='*60}\n")
    _log_file.write(f"Training started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    _log_file.write(f"{'='*60}\n\n")
    _log_file.flush()

    # Save original streams
    _original_stdout = sys.stdout
    _original_stderr = sys.stderr

    # Redirect stdout and stderr to TeeOutput (captures EVERYTHING)
    sys.stdout = TeeOutput(_original_stdout, _log_file, "stdout")
    sys.stderr = TeeOutput(_original_stderr, _log_file, "stderr")

    # Reconfigure logger to use dual output
    _logger_configured = False
    configure_logger()

    return log_path


def close_file_logging():
    """Close the log file and restore original stdout/stderr"""
    global _log_file, _log_file_path
    global _original_stdout, _original_stderr

    # Restore original streams first
    if _original_stdout is not None:
        sys.stdout = _original_stdout
        _original_stdout = None
    if _original_stderr is not None:
        sys.stderr = _original_stderr
        _original_stderr = None

    # Close log file
    if _log_file is not None:
        try:
            _log_file.write(f"\n{'='*60}\n")
            _log_file.write(f"Training ended at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            _log_file.write(f"{'='*60}\n")
            _log_file.flush()
            _log_file.close()
        except (IOError, ValueError):
            pass
        _log_file = None
        _log_file_path = None


class DualPrintLogger:
    """Logger that prints to stdout

    Note: With TeeOutput redirecting stdout/stderr, all output is
    automatically captured to the log file.
    """

    def __init__(self, file: Optional[TextIO] = None):
        self.file = file  # Kept for compatibility but not used

    def _log(self, message: str) -> None:
        # Print to sys.stdout - TeeOutput will handle file writing automatically
        try:
            print(message, file=sys.stdout)
            sys.stdout.flush()
        except (IOError, ValueError):
            pass

    # Required methods for structlog
    def msg(self, message: str) -> None:
        self._log(message)

    def info(self, message: str) -> None:
        self._log(message)

    def warning(self, message: str) -> None:
        self._log(message)

    def error(self, message: str) -> None:
        self._log(message)

    def debug(self, message: str) -> None:
        self._log(message)

    def critical(self, message: str) -> None:
        self._log(message)

    def __repr__(self) -> str:
        return f"<DualPrintLogger(file={self.file})>"


class DualLoggerFactory:
    """Factory that creates DualPrintLogger instances"""

    def __call__(self, *args) -> DualPrintLogger:
        return DualPrintLogger(file=_log_file)


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
        logger_factory=DualLoggerFactory(),
        cache_logger_on_first_use=False,  # Don't cache so file logging can be enabled later
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
