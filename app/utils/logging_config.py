"""
Logging configuration for the LLM Inference Engine.

Sets up structured logging with configurable levels and formats.
"""

import logging
import sys


def setup_logging(level: str = "INFO", fmt: str | None = None) -> None:
    """
    Configure the root logger for the inference engine.

    Args:
        level: Log level string (DEBUG, INFO, WARNING, ERROR).
        fmt: Optional format string. Uses a sensible default if None.
    """
    if fmt is None:
        fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

    # Remove existing handlers to avoid duplicate output
    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S"))

    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.addHandler(handler)

    # Suppress noisy third-party loggers
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
