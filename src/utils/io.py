"""File and path I/O helpers."""

import json
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    """Ensure directory exists and return Path object."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(data: Any, path: str | Path, indent: int = 2) -> None:
    """Save data to JSON file with UTF-8 encoding."""
    p = Path(path)
    ensure_dir(p.parent)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_json(path: str | Path) -> Any:
    """Load JSON file."""
    p = Path(path)
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)
