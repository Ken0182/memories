from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from mac_assistant.app.models import MemoryCandidate

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "memory_rules.json"

DEFAULT_RULES: dict[str, Any] = {
    "version": "1.0",
    "extraction": {
        "min_message_length": 20,
        "min_importance_to_store": 0.15,
        "max_candidates_per_turn": 5,
        "max_existing_memories_in_prompt": 8,
        "extract_from_user_messages_only": True,
    },
}


def load_rules() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        return DEFAULT_RULES
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    merged = DEFAULT_RULES.copy()
    merged.update(data)
    merged["extraction"] = {**DEFAULT_RULES["extraction"], **data.get("extraction", {})}
    return merged


EXTRACTION_SYSTEM_PROMPT = """You are a memory extraction agent.

Your job: analyze a user conversation turn and extract durable facts about the user.
Output ONLY a JSON array of memory candidates. No prose, no markdown.

STRICT RULES:
- Focus on stable preferences, biographical facts, goals, values, skills, emotional patterns, and relationship facts.
- Do NOT extract temporary task context, filler, assistant content, or non-literal jokes/sarcasm.
- Use operation: ADD | UPD | DEL | NOP.
- Use memory_type: preference | fact | goal | strategy | emotional_state | relationship | skill | belief | context | persona.
- summary must be third-person, self-contained, and explicit (use "User").
- confidence and importance must be numbers in [0.0, 1.0].
- semantic_tags should be lowercase-hyphenated tags.
- emotional_tags may be omitted unless emotionally relevant.
- UPD requires updates_memory_id.
- DEL requires contradicts_memory_id.

If nothing durable exists, return a single NOP candidate with low importance.

EXISTING MEMORIES:
{existing_memories_block}

CONVERSATION TURN:
{conversation_turn}

Respond with ONLY JSON array:
"""


def build_extraction_prompt(
    conversation_turn: str,
    existing_memories: list[dict[str, Any]] | None = None,
) -> str:
    rules = load_rules()
    max_existing = int(rules["extraction"]["max_existing_memories_in_prompt"])
    if existing_memories:
        lines: list[str] = []
        for memory in existing_memories[:max_existing]:
            tags = ", ".join(memory.get("semantic_tags", [])[:4])
            memory_id = str(memory.get("id", ""))[:8]
            lines.append(
                f"  [{memory_id}] ({memory.get('memory_type')}) {memory.get('summary')} | tags: {tags}"
            )
        memory_block = "\n".join(lines)
    else:
        memory_block = "  (none)"

    return EXTRACTION_SYSTEM_PROMPT.format(
        existing_memories_block=memory_block,
        conversation_turn=conversation_turn,
    )


_TRIVIAL_PATTERNS = [
    r"^(ok|okay|sure|thanks|thank you|got it|sounds good|great|cool|nice|yep|yes|no|nope|maybe)[\.\!\?]?$",
    r"^(hi|hey|hello|bye|goodbye|see you|ttyl)[\.\!\?]?$",
    r"^\?+$",
    r"^[\s\.\!\?]+$",
]
_TRIVIAL_RE = [re.compile(p, re.IGNORECASE) for p in _TRIVIAL_PATTERNS]
_QUESTION_RE = re.compile(
    r"^\s*(what|who|where|when|why|how|can you|could you|would you|please|help)\b",
    re.IGNORECASE,
)


def should_extract(message: str) -> tuple[bool, str]:
    rules = load_rules()
    min_len = int(rules["extraction"]["min_message_length"])
    msg = message.strip()
    if len(msg) < min_len:
        return False, f"too short ({len(msg)} chars, min {min_len})"
    for pattern in _TRIVIAL_RE:
        if pattern.match(msg):
            return False, "trivial/filler message"
    if _QUESTION_RE.match(msg) and len(msg) < 60:
        return False, "short question with no clear personal disclosure"
    return True, "ok"


def validate_candidate(candidate: dict[str, Any]) -> tuple[bool, str | None]:
    rules = load_rules()
    min_importance = float(rules["extraction"]["min_importance_to_store"])
    try:
        validated = MemoryCandidate.model_validate(candidate)
    except ValidationError as e:
        return False, str(e.errors())
    if validated.operation.value != "NOP" and validated.importance < min_importance:
        return (
            False,
            f"importance {validated.importance:.2f} below threshold {min_importance:.2f}",
        )
    return True, None


def validate_candidates(raw: list[Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rules = load_rules()
    max_candidates = int(rules["extraction"]["max_candidates_per_turn"])
    valid: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for idx, candidate in enumerate(raw):
        if idx >= max_candidates:
            rejected.append(
                {
                    "index": idx,
                    "reason": f"exceeds max_candidates_per_turn={max_candidates}",
                    "candidate": candidate,
                }
            )
            continue
        if not isinstance(candidate, dict):
            rejected.append({"index": idx, "reason": "not a dict", "candidate": candidate})
            continue
        ok, reason = validate_candidate(candidate)
        if ok:
            valid.append(MemoryCandidate.model_validate(candidate).model_dump(mode="json"))
        else:
            rejected.append({"index": idx, "reason": reason, "candidate": candidate})
    return valid, rejected


def should_gist(memory_type: str, count: int, oldest_days: int) -> tuple[bool, str]:
    rules = load_rules()
    limit = int(rules.get("gist_count_limit", 50))
    age_days = int(rules.get("gist_age_days", 30))
    if count > limit:
        return True, f"{memory_type} has {count} memories (limit {limit})"
    if oldest_days > age_days:
        return True, f"oldest {memory_type} is {oldest_days}d old (limit {age_days}d)"
    return False, "no gisting needed"
