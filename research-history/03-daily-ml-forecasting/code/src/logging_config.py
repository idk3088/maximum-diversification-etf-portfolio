"""Shared console and rotating-file logging configuration."""

from __future__ import annotations

import logging
import os
from logging.config import dictConfig
from pathlib import Path

from src.paths import LOGS_DIR

DEFAULT_LOGGER_NAME = "etf_portfolio"
DEFAULT_LOG_FILE = LOGS_DIR / "etf_portfolio.log"


def configure_logging(
    level: str | None = None,
    log_file: str | Path | None = None,
) -> logging.Logger:
    """Configure idempotent project-wide console and file logging."""
    resolved_level = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    resolved_log_file = Path(log_file) if log_file else DEFAULT_LOG_FILE
    resolved_log_file.parent.mkdir(parents=True, exist_ok=True)

    dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "standard": {
                    "format": "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                    "datefmt": "%Y-%m-%d %H:%M:%S",
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": resolved_level,
                    "formatter": "standard",
                },
                "file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": resolved_level,
                    "formatter": "standard",
                    "filename": str(resolved_log_file),
                    "maxBytes": 5_000_000,
                    "backupCount": 3,
                    "encoding": "utf-8",
                },
            },
            "loggers": {
                DEFAULT_LOGGER_NAME: {
                    "level": resolved_level,
                    "handlers": ["console", "file"],
                    "propagate": False,
                }
            },
        }
    )
    return logging.getLogger(DEFAULT_LOGGER_NAME)
