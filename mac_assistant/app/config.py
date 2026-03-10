"""
mac_assistant/app/config.py
─────────────────────────────────────────────────────────────────────────────
Load user_needs.json and memory_rules.json for runtime configuration.
"""

from __future__ import annotations

import json
from pathlib import Path

_CONFIG_DIR = Path(__file__).resolve().parent.parent / "data"
USER_NEEDS_PATH = _CONFIG_DIR / "user_needs.json"
MEMORY_RULES_PATH = _CONFIG_DIR / "memory_rules.json"

_user_needs_cache: dict | None = None
_memory_rules_cache: dict | None = None


def load_user_needs() -> dict:
    """Load user_needs.json (assistant profile defaults)."""
    global _user_needs_cache
    if _user_needs_cache is not None:
        return _user_needs_cache
    if USER_NEEDS_PATH.exists():
        with open(USER_NEEDS_PATH) as f:
            _user_needs_cache = json.load(f)
    else:
        _user_needs_cache = {}
    return _user_needs_cache


def load_memory_rules() -> dict:
    """Load memory_rules.json (extraction policy)."""
    global _memory_rules_cache
    if _memory_rules_cache is not None:
        return _memory_rules_cache
    if MEMORY_RULES_PATH.exists():
        with open(MEMORY_RULES_PATH) as f:
            _memory_rules_cache = json.load(f)
    else:
        _memory_rules_cache = {}
    return _memory_rules_cache


def get(key: str, default=None):
    """Get a key from user_needs."""
    return load_user_needs().get(key, default)
