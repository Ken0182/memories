"""
mac_assistant/app/extractor.py
─────────────────────────────────────────────────────────────────────────────
Memory extraction from conversation turns.

  - should_extract: pre-filter (rules.py)
  - extract_memory_candidates: LLM call + validation + returns candidates
  - integrate with retrieval for ADD vs UPD vs DEL resolution
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import json
from typing import Optional

import requests

from mac_assistant.app.config import get
from mac_assistant.app.prompt_builder import build_extraction_prompt
from mac_assistant.app.retrieval import find_similar_active
from mac_assistant.app.rules import should_extract, validate_candidates
from mac_assistant.app.schema import get_connection

OLLAMA_URL = "http://localhost:11434/api/generate"


def call_ollama(prompt: str,
                system_prompt: Optional[str] = None,
                model: Optional[str] = None) -> str:
    """Call Ollama generate API."""
    model = model or get("extraction_model", "qwen3:8b") or get("preferred_model", "llama3.2")
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }
    if system_prompt:
        payload["system"] = system_prompt

    response = requests.post(OLLAMA_URL, json=payload, timeout=120)
    response.raise_for_status()
    return response.json().get("response", "")


def _parse_json_array(raw: str) -> list[dict]:
    """Extract JSON array from LLM response."""
    start = raw.find("[")
    end = raw.rfind("]")
    if start == -1 or end == -1:
        return []

    json_text = raw[start : end + 1]
    try:
        return json.loads(json_text)
    except json.JSONDecodeError:
        return []


def extract_memory_candidates(user_message: str,
                              user_id: str = "default",
                              existing_memories: Optional[list[dict]] = None,
                              conn=None) -> tuple[list[dict], list[dict]]:
    """
    Extract and validate memory candidates from a user message.

    Returns (valid_candidates, rejected_candidates).
    Uses rules.should_extract for pre-filter, then LLM extraction, then validate_candidates.
    """
    do_extract, reason = should_extract(user_message)
    if not do_extract:
        return [], [{"reason": reason, "message_preview": user_message[:50]}]

    prompt = build_extraction_prompt(user_message, existing_memories)

    try:
        raw = call_ollama(prompt)
    except Exception as e:
        return [], [{"reason": f"LLM call failed: {e}", "message_preview": user_message[:50]}]

    candidates = _parse_json_array(raw)
    valid, rejected = validate_candidates(candidates)

    return valid, rejected


def extract_and_resolve(user_message: str,
                        user_id: str = "default",
                        conn=None) -> tuple[list[dict], list[dict]]:
    """
    Full extraction pipeline: fetch similar memories, extract, resolve ADD/UPD/DEL.
    Returns (valid_candidates_with_targets, rejected).
    """
    close = False
    if conn is None:
        conn = get_connection()
        close = True

    try:
        # Get similar memories for UPD/DEL resolution
        # Extract keywords from message for tag-based lookup (simplified)
        words = [w.lower() for w in user_message.split() if len(w) >= 4][:5]
        existing = []
        if words:
            existing = find_similar_active(conn, user_id, "preference", words, limit=8)
            if not existing:
                for mtype in ("fact", "preference", "goal"):
                    existing = find_similar_active(conn, user_id, mtype, words, limit=5)
                    if existing:
                        break

        valid, rejected = extract_memory_candidates(
            user_message, user_id, existing_memories=existing, conn=conn
        )

        # Optionally auto-resolve UPD/DEL targets by matching to existing
        # (LLM can provide updates_memory_id/contradicts_memory_id; we don't override)
        return valid, rejected
    finally:
        if close:
            conn.close()
