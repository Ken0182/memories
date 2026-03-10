"""
mac_assistant/app/memory_store.py
─────────────────────────────────────────────────────────────────────────────
All validated write operations to SQLite.
This is the only module that writes to the database.

Responsibilities:
  - Accept validated MemoryCandidate objects (from extractor.py)
  - Execute ADD / UPD / DEL / NOP logic
  - Write semantic + emotional tags to their join tables
  - Write memory links
  - Log every operation to memory_events (immutable audit trail)
  - Mark contradicted memories as 'obsolete' (never hard-delete)

NOT responsible for:
  - Extraction (extractor.py)
  - Retrieval queries (retrieval.py)
  - LLM calls (extractor.py)
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional
from uuid import uuid4

from mac_assistant.app.schema import DB_PATH, get_connection, init_db


# ─────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.utcnow().isoformat()


def _new_id() -> str:
    return str(uuid4())


def _log_event(conn: sqlite3.Connection,
               event_type: str,
               memory_id: Optional[str] = None,
               notes: Optional[str] = None) -> None:
    """Append-only audit log entry."""
    conn.execute(
        """INSERT INTO memory_events (id, memory_id, event_type, notes, created_at)
           VALUES (?, ?, ?, ?, ?)""",
        (_new_id(), memory_id, event_type, notes, _now())
    )


def _serialize(obj: Optional[dict]) -> Optional[str]:
    """Serialize optional dict to JSON string."""
    return json.dumps(obj) if obj else None


# ═══════════════════════════════════════════════════════════════════════════
# CORE WRITE OPERATIONS
# ═══════════════════════════════════════════════════════════════════════════

def add_memory(conn: sqlite3.Connection,
               user_id: str,
               candidate: dict) -> str:
    """
    ADD operation — write a brand new memory record.
    candidate is a validated MemoryCandidate dict (or Pydantic .model_dump()).
    Returns the new memory_id.
    """
    memory_id = _new_id()
    now = _now()

    conn.execute(
        """INSERT INTO memory_items (
               id, user_id, memory_type, summary, raw_content,
               source_type, source_ref, confidence, importance, status,
               evo_context_json, proactive_hints_json, persona_vector_json,
               created_at, updated_at
           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, ?, ?, ?, ?)""",
        (
            memory_id,
            user_id,
            str(candidate["memory_type"]).lower() if hasattr(candidate["memory_type"], "value") else candidate["memory_type"],
            candidate["summary"],
            candidate.get("raw_content"),
            candidate.get("source_type", "chat"),
            candidate.get("source_ref"),
            candidate["confidence"],
            candidate["importance"],
            _serialize(candidate.get("evo_context")),
            _serialize(candidate.get("proactive_hints")),
            _serialize(candidate.get("persona_vector")),
            now, now
        )
    )

    for tag in candidate.get("semantic_tags", []):
        _upsert_semantic_tag(conn, memory_id, tag)

    for etag in candidate.get("emotional_tags", []):
        emo = etag if isinstance(etag, dict) else {"emotion": etag.emotion, "weight": etag.weight}
        _upsert_emotional_tag(conn, memory_id, emo["emotion"], emo["weight"])

    for entity_id in candidate.get("linked_entity_ids", []):
        _write_link(conn, memory_id, entity_id, "same_entity")

    _log_event(conn, "ADD", memory_id,
               f"type={candidate.get('memory_type', '')} confidence={candidate.get('confidence', 0):.2f}")

    return memory_id


def update_memory(conn: sqlite3.Connection,
                  target_memory_id: str,
                  candidate: dict) -> str:
    """
    UPD operation — enrich an existing memory.
    Merges tags and updates fields where new data has higher confidence.
    """
    now = _now()

    conn.execute(
        """UPDATE memory_items SET
               summary    = ?,
               updated_at = ?,
               confidence = MAX(confidence, ?),
               importance = MAX(importance, ?)
           WHERE id = ? AND status = 'active'""",
        (
            candidate["summary"],
            now,
            candidate["confidence"],
            candidate["importance"],
            target_memory_id
        )
    )

    for tag in candidate.get("semantic_tags", []):
        _upsert_semantic_tag(conn, target_memory_id, tag)

    for etag in candidate.get("emotional_tags", []):
        emo = etag if isinstance(etag, dict) else {"emotion": etag.emotion, "weight": etag.weight}
        _upsert_emotional_tag(conn, target_memory_id, emo["emotion"], emo["weight"])

    _log_event(conn, "UPD", target_memory_id,
               f"enriched: {candidate.get('summary', '')[:80]}")

    return target_memory_id


def obsolete_memory(conn: sqlite3.Connection,
                    target_memory_id: str,
                    reason: str,
                    new_memory_id: Optional[str] = None) -> None:
    """
    DEL operation — mark a memory as obsolete (contradicted by newer data).
    Never hard-deletes. Writes CONTRADICTS link for audit trail.
    """
    now = _now()

    conn.execute(
        "UPDATE memory_items SET status = 'obsolete', updated_at = ? WHERE id = ?",
        (now, target_memory_id)
    )

    if new_memory_id:
        _write_link(conn, new_memory_id, target_memory_id, "contradicts")

    _log_event(conn, "DEL", target_memory_id, f"obsoleted: {reason[:120]}")


def nop_memory(conn: sqlite3.Connection, reason: str) -> None:
    """NOP operation — log that extraction decided not to store."""
    _log_event(conn, "NOP", None, f"rejected: {reason[:120]}")


# ═══════════════════════════════════════════════════════════════════════════
# DISPATCH
# ═══════════════════════════════════════════════════════════════════════════

def dispatch_candidate(conn: sqlite3.Connection,
                       user_id: str,
                       candidate: dict) -> Optional[str]:
    """
    Main entry point. Routes each candidate to ADD / UPD / DEL / NOP.
    Returns the memory_id that was written (or None for NOP/DEL without target).
    """
    op = str(candidate.get("operation", "NOP")).upper()

    if op == "ADD":
        memory_id = add_memory(conn, user_id, candidate)
        conn.commit()
        return memory_id

    elif op == "UPD":
        target = candidate.get("updates_memory_id")
        if not target:
            memory_id = add_memory(conn, user_id, candidate)
            conn.commit()
            return memory_id
        update_memory(conn, target, candidate)
        conn.commit()
        return target

    elif op == "DEL":
        target = candidate.get("contradicts_memory_id")
        if target:
            new_id = add_memory(conn, user_id, candidate)
            obsolete_memory(conn, target, candidate.get("summary", ""), new_id)
            conn.commit()
            return new_id
        return None

    else:  # NOP
        nop_memory(conn, candidate.get("summary", "no reason given"))
        conn.commit()
        return None


# ═══════════════════════════════════════════════════════════════════════════
# TAG HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _upsert_semantic_tag(conn: sqlite3.Connection,
                         memory_id: str, tag: str) -> None:
    tag_val = tag.lower().strip().replace(" ", "-") if isinstance(tag, str) else str(tag)
    conn.execute(
        "INSERT OR IGNORE INTO memory_semantic_tags (memory_id, tag) VALUES (?, ?)",
        (memory_id, tag_val)
    )


def _upsert_emotional_tag(conn: sqlite3.Connection,
                          memory_id: str, emotion: str, weight: float) -> None:
    em = emotion.lower().strip()
    w = float(weight)
    conn.execute(
        "DELETE FROM memory_emotional_tags WHERE memory_id = ? AND emotion = ?",
        (memory_id, em)
    )
    conn.execute(
        "INSERT INTO memory_emotional_tags (memory_id, emotion, weight) VALUES (?, ?, ?)",
        (memory_id, em, w)
    )


# ═══════════════════════════════════════════════════════════════════════════
# LINK HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _write_link(conn: sqlite3.Connection,
                from_id: str, to_id: str,
                link_type: str, weight: float = 1.0) -> None:
    conn.execute(
        """INSERT OR IGNORE INTO memory_links
           (id, from_memory_id, to_memory_id, link_type, weight, created_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (_new_id(), from_id, to_id, link_type, weight, _now())
    )


def write_link(conn: sqlite3.Connection,
               from_id: str, to_id: str,
               link_type: str, weight: float = 1.0) -> None:
    """Public API for explicit link creation."""
    _write_link(conn, from_id, to_id, link_type, weight)
    conn.commit()


# ═══════════════════════════════════════════════════════════════════════════
# GOAL OPERATIONS
# ═══════════════════════════════════════════════════════════════════════════

def upsert_goal(conn: sqlite3.Connection, user_id: str, goal: dict) -> str:
    """Create or update a goal record."""
    now = _now()
    goal_id = goal.get("id") or _new_id()

    conn.execute(
        """INSERT INTO goals (id, user_id, title, description, status, progress,
                              linked_memory_ids, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(id) DO UPDATE SET
               title              = excluded.title,
               description        = excluded.description,
               status             = excluded.status,
               progress           = excluded.progress,
               linked_memory_ids  = excluded.linked_memory_ids,
               updated_at         = excluded.updated_at""",
        (
            goal_id, user_id,
            goal["title"],
            goal.get("description"),
            goal.get("status", "IN_PROGRESS"),
            goal.get("progress", 0.0),
            json.dumps(goal.get("linked_memory_ids", [])),
            goal.get("created_at", now),
            now
        )
    )
    conn.commit()
    return goal_id


# ═══════════════════════════════════════════════════════════════════════════
# ASSISTANT PROFILE
# ═══════════════════════════════════════════════════════════════════════════

def upsert_profile(conn: sqlite3.Connection, profile: dict) -> None:
    """Create or update assistant profile for a user."""
    now = _now()
    conn.execute(
        """INSERT INTO assistant_profile (
               user_id, humor_mode, proactive_level, preferred_model,
               notification_style, memory_rules_version, steering_enabled,
               sycophancy_threshold, aggression_threshold, created_at, updated_at
           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(user_id) DO UPDATE SET
               humor_mode            = excluded.humor_mode,
               proactive_level       = excluded.proactive_level,
               preferred_model       = excluded.preferred_model,
               notification_style    = excluded.notification_style,
               steering_enabled      = excluded.steering_enabled,
               sycophancy_threshold  = excluded.sycophancy_threshold,
               aggression_threshold  = excluded.aggression_threshold,
               updated_at            = excluded.updated_at""",
        (
            profile["user_id"],
            profile.get("humor_mode", "dry"),
            profile.get("proactive_level", "medium"),
            profile.get("preferred_model", "llama3.2"),
            profile.get("notification_style", "subtle"),
            profile.get("memory_rules_version", "1.0"),
            int(profile.get("steering_enabled", True)),
            profile.get("sycophancy_threshold", 0.65),
            profile.get("aggression_threshold", 0.65),
            profile.get("created_at", now),
            now
        )
    )
    conn.commit()


# ═══════════════════════════════════════════════════════════════════════════
# PERSONA SNAPSHOT
# ═══════════════════════════════════════════════════════════════════════════

def write_persona_snapshot(conn: sqlite3.Connection,
                          user_id: str,
                          vector: dict,
                          session_id: Optional[str] = None,
                          steering_applied: bool = False,
                          notes: Optional[str] = None) -> str:
    """Write a persona vector reading to the historical snapshots table."""
    snap_id = _new_id()
    conn.execute(
        """INSERT INTO persona_snapshots (
               id, user_id, session_id, sycophancy, aggression,
               hallucination_risk, status_sensitivity, openness, mood_valence,
               steering_applied, notes, created_at
           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            snap_id, user_id, session_id,
            vector.get("sycophancy", 0.0),
            vector.get("aggression", 0.0),
            vector.get("hallucination_risk", 0.0),
            vector.get("status_sensitivity", 0.0),
            vector.get("openness", 0.5),
            vector.get("mood_valence", 0.5),
            int(steering_applied),
            notes,
            _now()
        )
    )
    conn.commit()
    return snap_id


# ═══════════════════════════════════════════════════════════════════════════
# GISTING
# ═══════════════════════════════════════════════════════════════════════════

def gist_memories(conn: sqlite3.Connection,
                  user_id: str,
                  memory_ids: list[str],
                  gist_summary: str,
                  gist_tags: list[str]) -> str:
    """
    Replace a list of old memories with a single compressed gist memory.
    """
    now = _now()
    gist_id = _new_id()

    conn.execute(
        """INSERT INTO memory_items (
               id, user_id, memory_type, summary, source_type,
               confidence, importance, status, created_at, updated_at
           ) VALUES (?, ?, 'context', ?, 'inference', 0.9, 0.6, 'gist', ?, ?)""",
        (gist_id, user_id, gist_summary, now, now)
    )

    for tag in gist_tags:
        _upsert_semantic_tag(conn, gist_id, tag)

    for source_id in memory_ids:
        conn.execute(
            "UPDATE memory_items SET status = 'gist', updated_at = ? WHERE id = ?",
            (now, source_id)
        )
        _write_link(conn, gist_id, source_id, "refines")

    _log_event(conn, "ADD", gist_id, f"gist of {len(memory_ids)} memories")

    conn.commit()
    return gist_id