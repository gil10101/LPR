"""Minimal .env loader (no external dependency).

Reads ``KEY=VALUE`` lines from a ``.env`` file at the project root and copies
them into ``os.environ`` (without overwriting variables already set in the real
environment). Used to pass secrets like Kaggle credentials to the download
script without committing them.
"""
from __future__ import annotations

import os
from typing import Optional

# Project root = two levels up from this file (lpr/utils/env.py -> repo root).
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_dotenv(path: Optional[str] = None, override: bool = False) -> dict:
    """Load a .env file into os.environ. Returns the dict of parsed values.

    Lines are ``KEY=VALUE``; blank lines and ``#`` comments are ignored, optional
    surrounding quotes are stripped, and a leading ``export`` is tolerated. Real
    environment variables win unless ``override`` is True.
    """
    path = path or os.path.join(_PROJECT_ROOT, ".env")
    values: dict = {}
    if not os.path.exists(path):
        return values
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.lower().startswith("export "):
                line = line[len("export "):]
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if not key:
                continue
            values[key] = value
            if override or key not in os.environ:
                os.environ[key] = value
    return values
