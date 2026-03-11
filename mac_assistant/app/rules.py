"""
mac_assistant/app/rules.py
─────────────────────────────────────────────────────────────────────────────
Memory extraction policy.

This module defines:
  1. What the LLM is told to extract (the extraction prompt)
  2. What gets rejected before even reaching the LLM (pre-filter)
  3. Post-LLM validation rules (what makes a candidate invalid)

Inspired by:
  - Auto Memory's strict policy-driven extraction
  - Roni Memory's "what NOT to store" rules
  - OI capsule's ADD/UPD/DEL/NOP logic
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import re
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════════
# EXTRACTION PROMPT
# ═══════════════════════════════════════════════════════════════════════════

EXTRACTION_SYSTEM_PROMPT = """You are a memory extraction agent.

Your job: analyze a conversation turn and extract durable facts about the user.
Output ONLY a valid JSON array of memory candidates. No explanation. No markdown.

STRICT RULES — follow exactly:

WHAT TO EXTRACT:
  - Stable preferences ("likes dry humor", "prefers concise answers")
  - Biographical facts ("lives in San Francisco", "works as a developer")
  - Long-term goals ("wants to launch a startup", "learning Spanish")
  - Emotional patterns ("gets stressed about financial uncertainty")
  - Behavioral strategies (individual gain, protecting family, social reciprocity)
  - Skills and competencies ("knows Python", "experienced with SQLite")
  - Relationship facts ("has a sibling named X", "works with Y")

WHAT NOT TO EXTRACT:
  - Temporary session context ("user is asking about X right now")
  - Questions the user asked (not facts about them)
  - Assistant responses or summaries
  - Sarcasm, jokes, or non-literal statements
  - Things the user said about OTHER people (unless revealing user's values)
  - Trivial filler ("user said hi", "user said thanks")
  - Requests for help (the task, not the person)

EACH CANDIDATE MUST HAVE:
  - operation: "ADD" | "UPD" | "DEL" | "NOP"
  - memory_type: "preference" | "fact" | "goal" | "strategy" | "emotional_state" | "relationship" | "skill" | "belief" | "context"
  - summary: third-person, self-contained, no pronouns (use "User" not "I")
  - confidence: 0.0 → 1.0 (how certain is this durable fact?)
  - importance: 0.0 → 1.0 (how useful for future sessions?)
  - semantic_tags: list of 2-8 lowercase hyphenated tags
  - emotional_tags: list of {emotion, weight} objects (only if emotionally relevant)

OPTIONAL FIELDS:
  - raw_content: the original phrase that generated this
  - source_type: "chat" (default) | "inference"
  - updates_memory_id: REQUIRED if operation is UPD
  - contradicts_memory_id: REQUIRED if operation is DEL

IF nothing durable can be extracted, return:
  [{"operation": "NOP", "memory_type": "context", "summary": "No durable facts found.", "confidence": 0.1, "importance": 0.0, "semantic_tags": []}]

EXISTING MEMORIES (check before deciding ADD vs UPD vs DEL):
{existing_memories_block}

CONVERSATION TURN:
{conversation_turn}

Respond with ONLY the JSON array:"""


def build_extraction_prompt(conversation_turn: str,
                            existing_memories: list[dict] | None = None) -> str:
    """Render the extraction prompt with context injected."""
    if existing_memories:
        lines = []
        for m in existing_memories[:8]:
            tags = ", ".join(m.get("semantic_tags", [])[:4])
            mid = m.get("id", "")[:8]
            mtype = m.get("memory_type", "")
            summary = m.get("summary", "")
            lines.append(f"  [{mid}] ({mtype}) {summary} | tags: {tags}")
        block = "\n".join(lines)
    else:
        block = "  (none yet)"

    return EXTRACTION_SYSTEM_PROMPT.format(
        existing_memories_block=block,
        conversation_turn=conversation_turn
    )


# ═══════════════════════════════════════════════════════════════════════════
# PRE-FILTER
# ═══════════════════════════════════════════════════════════════════════════

MIN_EXTRACTION_LENGTH = 20

_TRIVIAL_PATTERNS = [
    r"^(ok|okay|sure|thanks|thank you|got it|sounds good|great|cool|nice|yep|yes|no|nope|maybe)[\.\!\?]?$",
    r"^(hi|hey|hello|bye|goodbye|see you|ttyl)[\.\!\?]?$",
    r"^\?+$",
    r"^[\s\.\!\?]+$",
]
_TRIVIAL_RE = [re.compile(p, re.IGNORECASE) for p in _TRIVIAL_PATTERNS]

_QUESTION_RE = re.compile(
    r"^\s*(what|who|where|when|why|how|can you|could you|would you|please|help)",
    re.IGNORECASE
)


def should_extract(message: str) -> tuple[bool, str]:
    """Pre-filter: decide whether to call the LLM for extraction."""
    msg = message.strip()

    if len(msg) < MIN_EXTRACTION_LENGTH:
        return False, f"too short ({len(msg)} chars, min {MIN_EXTRACTION_LENGTH})"

    for pattern in _TRIVIAL_RE:
        if pattern.match(msg):
            return False, "trivial/filler message"

    if _QUESTION_RE.match(msg) and len(msg) < 60:
        return False, "short question with no apparent personal disclosure"

    return True, "ok"


# ═══════════════════════════════════════════════════════════════════════════
# POST-LLM VALIDATION
# ═══════════════════════════════════════════════════════════════════════════

REQUIRED_FIELDS = {"operation", "memory_type", "summary", "confidence", "importance"}

VALID_OPERATIONS = {"ADD", "UPD", "DEL", "NOP"}
VALID_TYPES = {
    "preference", "fact", "goal", "strategy", "emotional_state",
    "relationship", "skill", "belief", "context", "persona"
}

MIN_IMPORTANCE_TO_STORE = 0.15
MIN_SUMMARY_LENGTH = 8


def validate_candidate(candidate: dict) -> tuple[bool, Optional[str]]:
    """Validate a single candidate dict from the LLM."""
    missing = REQUIRED_FIELDS - set(candidate.keys())
    if missing:
        return False, f"missing required fields: {missing}"

    op = str(candidate.get("operation", "")).upper()
    if op not in VALID_OPERATIONS:
        return False, f"invalid operation: '{op}'"

    mtype = str(candidate.get("memory_type", "")).lower()
    if mtype not in VALID_TYPES:
        return False, f"invalid memory_type: '{mtype}'"

    summary = str(candidate.get("summary", "")).strip()
    if len(summary) < MIN_SUMMARY_LENGTH:
        return False, f"summary too short: '{summary}'"

    first_person_starts = ("i ", "i'm", "i've", "my ", "i'd", "i'll")
    if summary.lower().startswith(first_person_starts):
        return False, f"summary is first-person, must be third-person: '{summary[:50]}'"

    confidence = candidate.get("confidence")
    importance = candidate.get("importance")

    if not isinstance(confidence, (int, float)) or not (0.0 <= confidence <= 1.0):
        return False, f"confidence out of range: {confidence}"

    if not isinstance(importance, (int, float)) or not (0.0 <= importance <= 1.0):
        return False, f"importance out of range: {importance}"

    if op != "NOP" and importance < MIN_IMPORTANCE_TO_STORE:
        return False, f"importance {importance} below threshold {MIN_IMPORTANCE_TO_STORE}"

    if op == "UPD" and not candidate.get("updates_memory_id"):
        return False, "UPD operation missing updates_memory_id"

    if op == "DEL" and not candidate.get("contradicts_memory_id"):
        return False, "DEL operation missing contradicts_memory_id"

    for etag in candidate.get("emotional_tags", []):
        if not isinstance(etag, dict):
            return False, f"emotional_tag must be a dict, got: {type(etag)}"
        w = etag.get("weight", 0)
        if not (0.0 <= w <= 1.0):
            return False, f"emotional_tag weight out of range: {w}"

    return True, None


def validate_candidates(raw: list) -> tuple[list[dict], list[dict]]:
    """Validate a list of candidates. Returns (valid, rejected)."""
    valid = []
    rejected = []

    for i, candidate in enumerate(raw):
        if not isinstance(candidate, dict):
            rejected.append({"index": i, "reason": "not a dict", "raw": str(candidate)})
            continue

        ok, reason = validate_candidate(candidate)
        if ok:
            candidate["operation"] = str(candidate["operation"]).upper()
            candidate["memory_type"] = str(candidate["memory_type"]).lower()
            valid.append(candidate)
        else:
            rejected.append({"index": i, "reason": reason, "candidate": candidate})

    return valid, rejected


# ═══════════════════════════════════════════════════════════════════════════
# GIST POLICY
# ═══════════════════════════════════════════════════════════════════════════

GIST_AGE_DAYS = 30
GIST_COUNT_LIMIT = 50


def should_gist(memory_type: str,
                count: int,
                oldest_days: int) -> tuple[bool, str]:
    """Decide whether a memory_type bucket needs gisting."""
    if count > GIST_COUNT_LIMIT:
        return True, f"{memory_type} has {count} memories (limit {GIST_COUNT_LIMIT})"
    if oldest_days > GIST_AGE_DAYS:
        return True, f"oldest {memory_type} is {oldest_days}d old (limit {GIST_AGE_DAYS}d)"
    return False, "no gisting needed"
