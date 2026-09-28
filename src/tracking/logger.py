"""Structured logger with console and JSONL output for research experiments."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional
from src.utils.io import ensure_dir


class JsonlHandler(logging.Handler):
    """Custom logging handler to record structured metrics into a JSONL file."""

    def __init__(self, filepath: Path):
        super().__init__()
        self.filepath = filepath
        ensure_dir(filepath.parent)

    def emit(self, record: logging.LogRecord) -> None:
        if hasattr(record, "structured_data"):
            data = getattr(record, "structured_data")
            with open(self.filepath, "a", encoding="utf-8") as f:
                f.write(json.dumps(data) + "\n")


def setup_logger(
    name: str = "fl_osr",
    log_dir: Optional[str | Path] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Setup structured application logger with stream and optional file handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers if setup_logger is called repeatedly
    if logger.hasHandlers():
        logger.handlers.clear()

    # Stream Handler
    stream_fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(stream_fmt)
    logger.addHandler(sh)

    if log_dir is not None:
        ld = ensure_dir(log_dir)
        text_handler = logging.FileHandler(ld / "experiment.log", encoding="utf-8")
        text_handler.setFormatter(stream_fmt)
        logger.addHandler(text_handler)

        jsonl_handler = JsonlHandler(ld / "metrics.jsonl")
        logger.addHandler(jsonl_handler)

    return logger
