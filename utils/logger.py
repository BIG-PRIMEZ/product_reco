"""
Centralized logging configuration for the product recommendation system.

Provides structured logging with different levels and formatters for
development and production environments.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
import json


class JSONFormatter(logging.Formatter):
    """
    Custom JSON formatter for structured logging.

    Outputs log records as JSON for easy parsing and analysis.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }

        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)

        # Add extra fields
        if hasattr(record, 'user_id'):
            log_data['user_id'] = record.user_id
        if hasattr(record, 'request_id'):
            log_data['request_id'] = record.request_id
        if hasattr(record, 'query'):
            log_data['query'] = record.query
        if hasattr(record, 'duration_ms'):
            log_data['duration_ms'] = record.duration_ms

        return json.dumps(log_data)


def setup_logger(
    name: str,
    level: str = 'INFO',
    log_file: Optional[str] = None,
    json_format: bool = False
) -> logging.Logger:
    """
    Set up and configure a logger.

    Args:
        name: Logger name (usually __name__ of the module)
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        log_file: Optional file path to write logs to
        json_format: If True, use JSON formatter for structured logging

    Returns:
        Configured logger instance

    Example:
        >>> logger = setup_logger(__name__, level='DEBUG')
        >>> logger.info("Service initialized", extra={'service': 'recommender'})
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))

    # Choose formatter
    if json_format:
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with default configuration.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Logger instance

    Example:
        >>> from utils.logger import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing query")
    """
    return logging.getLogger(name)


class LoggerAdapter(logging.LoggerAdapter):
    """
    Custom logger adapter for adding context to log records.

    Example:
        >>> logger = get_logger(__name__)
        >>> adapter = LoggerAdapter(logger, {'request_id': '123', 'user_id': 'user_456'})
        >>> adapter.info("Processing request")
        # Logs with request_id and user_id automatically added
    """

    def process(self, msg, kwargs):
        """Add context fields to log record."""
        # Add context to extra
        extra = kwargs.get('extra', {})
        extra.update(self.extra)
        kwargs['extra'] = extra
        return msg, kwargs


# Initialize root application logger
def init_app_logging(
    level: str = 'INFO',
    log_dir: Optional[str] = None,
    json_format: bool = False
):
    """
    Initialize application-wide logging.

    Args:
        level: Global logging level
        log_dir: Directory to store log files
        json_format: Use JSON formatting

    Example:
        >>> init_app_logging(level='DEBUG', log_dir='logs', json_format=True)
    """
    # Create log directory if specified
    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)

        # Set up file for app logs
        app_log_file = Path(log_dir) / f'app_{datetime.now().strftime("%Y%m%d")}.log'
        error_log_file = Path(log_dir) / f'error_{datetime.now().strftime("%Y%m%d")}.log'

        # Set up root logger with file handlers
        root_logger = logging.getLogger('product_reco')
        root_logger.setLevel(getattr(logging, level.upper()))

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))

        # App log file handler
        app_file_handler = logging.FileHandler(app_log_file)
        app_file_handler.setLevel(getattr(logging, level.upper()))

        # Error log file handler (only errors and critical)
        error_file_handler = logging.FileHandler(error_log_file)
        error_file_handler.setLevel(logging.ERROR)

        # Set formatter
        if json_format:
            formatter = JSONFormatter()
        else:
            formatter = logging.Formatter(
                fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )

        console_handler.setFormatter(formatter)
        app_file_handler.setFormatter(formatter)
        error_file_handler.setFormatter(formatter)

        root_logger.addHandler(console_handler)
        root_logger.addHandler(app_file_handler)
        root_logger.addHandler(error_file_handler)

        root_logger.propagate = False
    else:
        # Just console logging
        setup_logger('product_reco', level=level, json_format=json_format)
