from __future__ import annotations

import json
from typing import Any

import requests

from mac_assistant.app.models import MemoryCandidate, MemoryExtractionResult
from mac_assistant.app.prompt_builder import build_extraction_prompt
from mac_assistant.app.rules import should_extract, validate_candidates

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "qwen3:8b"


def call_ollama(prompt: str, model: str = DEFAULT_MODEL, timeout: int = 120) -> str:
    response = requests.post(
        OLLAMA_URL,
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()
    return str(data.get("response", "")).strip()


def _extract_json_slice(raw: str) -> str | None:
    raw = raw.strip()
    if not raw:
        return None

    for opening, closing in (("[", "]"), ("{", "}")):
        start = raw.find(opening)
        end = raw.rfind(closing)
        if start != -1 and end != -1 and end > start:
            return raw[start : end + 1]
    return None


def _coerce_candidate_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        if "candidate_memories" in payload and isinstance(payload["candidate_memories"], list):
            return [item for item in payload["candidate_memories"] if isinstance(item, dict)]
        return [payload]
    return []


def extract_memory_candidates(
    user_message: str,
    existing_memories: list[dict[str, Any]] | None = None,
    model: str = DEFAULT_MODEL,
) -> MemoryExtractionResult:
    do_extract, _ = should_extract(user_message)
    if not do_extract:
        return MemoryExtractionResult(candidate_memories=[])

    prompt = build_extraction_prompt(user_message, existing_memories=existing_memories)
    raw = call_ollama(prompt, model=model)
    json_text = _extract_json_slice(raw)
    if not json_text:
        return MemoryExtractionResult(candidate_memories=[])

    try:
        parsed = json.loads(json_text)
    except json.JSONDecodeError:
        return MemoryExtractionResult(candidate_memories=[])

    candidate_dicts = _coerce_candidate_payload(parsed)
    valid, _rejected = validate_candidates(candidate_dicts)
    candidates = [MemoryCandidate.model_validate(item) for item in valid]
    return MemoryExtractionResult(candidate_memories=candidates)
