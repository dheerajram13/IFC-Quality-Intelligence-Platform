"""Logging configuration for IFC Quality Intelligence Platform.

This module provides centralized logging setup with support for:
- Console and file logging
- Colored output for better readability
- Structured logging with context
- Log level configuration via environment variables
"""

import logging
import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler


class LoggerSetup:
    """Centralized logger setup and management."""

    _initialized = False
    _loggers: dict[str, logging.Logger] = {}

    @classmethod
    def setup_logging(
        cls,
        level: str = "INFO",
        log_file: Optional[Path] = None,
        log_format: Optional[str] = None,
        enable_rich: bool = True,
    ) -> None:
        """Set up logging configuration for the application.

        Args:
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
            log_file: Optional path to log file. If None, logs only to console.
            log_format: Custom log format string. If None, uses default format.
            enable_rich: Use Rich library for colored console output.
        """
        if cls._initialized:
            return

        # Set the root logger level
        log_level = getattr(logging, level.upper(), logging.INFO)
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)

        # Remove existing handlers
        root_logger.handlers.clear()

        # Configure console handler
        if enable_rich:
            # Use Rich handler for beautiful console output
            console_handler = RichHandler(
                rich_tracebacks=True,
                show_time=True,
                show_path=True,
                markup=True,
                console=Console(stderr=True),
            )
        else:
            # Standard console handler
            console_handler = logging.StreamHandler(sys.stderr)
            formatter = logging.Formatter(
                log_format
                or "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            console_handler.setFormatter(formatter)

        console_handler.setLevel(log_level)
        root_logger.addHandler(console_handler)

        # Configure file handler if log_file is provided
        if log_file is not None:
            log_file.parent.mkdir(parents=True, exist_ok=True)

            file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
            file_formatter = logging.Formatter(
                log_format
                or "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            file_handler.setFormatter(file_formatter)
            file_handler.setLevel(log_level)
            root_logger.addHandler(file_handler)

        # Silence noisy third-party loggers
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("matplotlib").setLevel(logging.WARNING)
        logging.getLogger("PIL").setLevel(logging.WARNING)

        cls._initialized = True

    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """Get or create a logger with the given name.

        Args:
            name: Logger name (typically __name__ of the calling module).

        Returns:
            logging.Logger: Configured logger instance.
        """
        if name not in cls._loggers:
            logger = logging.getLogger(name)
            cls._loggers[name] = logger

        return cls._loggers[name]

    @classmethod
    def reset(cls) -> None:
        """Reset logging configuration (useful for testing)."""
        cls._initialized = False
        cls._loggers.clear()
        logging.getLogger().handlers.clear()


def get_logger(name: str) -> logging.Logger:
    """Convenience function to get a logger.

    Args:
        name: Logger name (typically __name__ of the calling module).

    Returns:
        logging.Logger: Configured logger instance.

    Example:
        >>> from ifcqi.logger import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing started")
    """
    return LoggerSetup.get_logger(name)


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    log_format: Optional[str] = None,
    enable_rich: bool = True,
) -> None:
    """Convenience function to set up logging.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Optional path to log file. If None, logs only to console.
        log_format: Custom log format string. If None, uses default format.
        enable_rich: Use Rich library for colored console output.

    Example:
        >>> from ifcqi.logger import setup_logging
        >>> setup_logging(level="DEBUG", log_file=Path("logs/app.log"))
    """
    LoggerSetup.setup_logging(
        level=level,
        log_file=log_file,
        log_format=log_format,
        enable_rich=enable_rich,
    )


class LogContext:
    """Context manager for temporary log level changes.

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Normal logging")
        >>> with LogContext(logger, logging.DEBUG):
        ...     logger.debug("This will be shown")
        >>> logger.debug("This won't be shown if level is INFO")
    """

    def __init__(self, logger: logging.Logger, level: int) -> None:
        """Initialize log context.

        Args:
            logger: Logger to modify.
            level: Temporary log level.
        """
        self.logger = logger
        self.new_level = level
        self.original_level = logger.level

    def __enter__(self) -> logging.Logger:
        """Enter context and set new log level."""
        self.logger.setLevel(self.new_level)
        return self.logger

    def __exit__(self, exc_type: type, exc_val: Exception, exc_tb: type) -> None:
        """Exit context and restore original log level."""
        self.logger.setLevel(self.original_level)
