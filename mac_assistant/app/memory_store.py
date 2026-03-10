from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator
from uuid import uuid4

from mac_assistant.app.models import (
    AssistantProfile,
    Goal,
    GoalStatus,
    MemoryCandidate,
    MemoryOperation,
    MemoryType,
    PersonaVector,
)
from mac_assistant.app.schema import DB_PATH, get_connection, init_db


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id() -> str:
    return str(uuid4())


@contextmanager
def managed_connection(db_path: str | None = None) -> Iterator[sqlite3.Connection]:
    conn = get_connection(db_path or DB_PATH)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _normalize_candidate(candidate: MemoryCandidate | dict[str, Any]) -> MemoryCandidate:
    return candidate if isinstance(candidate, MemoryCandidate) else MemoryCandidate.model_validate(candidate)


def _log_event(
    conn: sqlite3.Connection,
    event_type: MemoryOperation | str,
    memory_id: str | None = None,
    notes: str | None = None,
) -> None:
    event = event_type.value if isinstance(event_type, MemoryOperation) else event_type
    conn.execute(
        """INSERT INTO memory_events (id, memory_id, event_type, notes, created_at)
           VALUES (?, ?, ?, ?, ?)""",
        (_new_id(), memory_id, event, notes, _utc_now()),
    )


def _upsert_semantic_tag(conn: sqlite3.Connection, memory_id: str, tag: str) -> None:
    conn.execute(
        """INSERT OR IGNORE INTO memory_semantic_tags (memory_id, tag)
           VALUES (?, ?)""",
        (memory_id, tag.lower().strip().replace(" ", "-")),
    )


def _upsert_emotional_tag(
    conn: sqlite3.Connection,
    memory_id: str,
    emotion: str,
    weight: float,
) -> None:
    conn.execute(
        """INSERT INTO memory_emotional_tags (memory_id, emotion, weight)
           VALUES (?, ?, ?)
           ON CONFLICT(memory_id, emotion) DO UPDATE SET weight = excluded.weight""",
        (memory_id, emotion.lower().strip(), weight),
    )


def _write_link(
    conn: sqlite3.Connection,
    from_id: str,
    to_id: str,
    link_type: str,
    weight: float = 1.0,
) -> None:
    conn.execute(
        """INSERT OR IGNORE INTO memory_links
           (id, from_memory_id, to_memory_id, link_type, weight, created_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (_new_id(), from_id, to_id, link_type, weight, _utc_now()),
    )


def add_memory(
    conn: sqlite3.Connection,
    user_id: str,
    candidate: MemoryCandidate | dict[str, Any],
) -> str:
    c = _normalize_candidate(candidate)
    memory_id = _new_id()
    now = _utc_now()

    conn.execute(
        """INSERT INTO memory_items (
               id, user_id, memory_type, summary, raw_content, source_type, source_ref,
               confidence, importance, status, evo_context_json, proactive_hints_json,
               persona_vector_json, created_at, updated_at
           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, ?, ?, ?, ?)""",
        (
            memory_id,
            user_id,
            c.memory_type.value,
            c.summary,
            c.raw_content,
            c.source_type.value,
            c.source_ref,
            c.confidence,
            c.importance,
            json.dumps(c.evo_context.model_dump()) if c.evo_context else None,
            json.dumps(c.proactive_hints.model_dump()) if c.proactive_hints else None,
            json.dumps(c.persona_vector.model_dump()) if c.persona_vector else None,
            now,
            now,
        ),
    )

    for tag in c.semantic_tags:
        _upsert_semantic_tag(conn, memory_id, tag)
    for etag in c.emotional_tags:
        _upsert_emotional_tag(conn, memory_id, etag.emotion, etag.weight)

    for entity_id in c.linked_entity_ids:
        _write_link(conn, memory_id, entity_id, "same_entity")

    _log_event(
        conn,
        MemoryOperation.ADD,
        memory_id,
        f"type={c.memory_type.value} confidence={c.confidence:.2f}",
    )
    return memory_id


def update_memory(
    conn: sqlite3.Connection,
    target_memory_id: str,
    candidate: MemoryCandidate | dict[str, Any],
) -> str:
    c = _normalize_candidate(candidate)

    conn.execute(
        """UPDATE memory_items
           SET summary = ?, updated_at = ?,
               confidence = MAX(confidence, ?),
               importance = MAX(importance, ?),
               source_ref = COALESCE(?, source_ref)
           WHERE id = ? AND status = 'active'""",
        (c.summary, _utc_now(), c.confidence, c.importance, c.source_ref, target_memory_id),
    )

    for tag in c.semantic_tags:
        _upsert_semantic_tag(conn, target_memory_id, tag)
    for etag in c.emotional_tags:
        _upsert_emotional_tag(conn, target_memory_id, etag.emotion, etag.weight)

    _log_event(
        conn,
        MemoryOperation.UPD,
        target_memory_id,
        f"summary={c.summary[:120]}",
    )
    return target_memory_id


def obsolete_memory(
    conn: sqlite3.Connection,
    target_memory_id: str,
    reason: str,
    new_memory_id: str | None = None,
) -> None:
    conn.execute(
        "UPDATE memory_items SET status = 'obsolete', updated_at = ? WHERE id = ?",
        (_utc_now(), target_memory_id),
    )
    if new_memory_id:
        _write_link(conn, new_memory_id, target_memory_id, "contradicts")
    _log_event(conn, MemoryOperation.DEL, target_memory_id, reason[:200])


def nop_memory(conn: sqlite3.Connection, reason: str) -> None:
    _log_event(conn, MemoryOperation.NOP, None, reason[:200])


def dispatch_candidate(
    conn: sqlite3.Connection,
    user_id: str,
    candidate: MemoryCandidate | dict[str, Any],
) -> str | None:
    c = _normalize_candidate(candidate)

    if c.operation == MemoryOperation.ADD:
        return add_memory(conn, user_id, c)

    if c.operation == MemoryOperation.UPD:
        if c.updates_memory_id:
            return update_memory(conn, c.updates_memory_id, c)
        # Fallback to ADD for robustness in live systems.
        return add_memory(conn, user_id, c)

    if c.operation == MemoryOperation.DEL:
        if not c.contradicts_memory_id:
            nop_memory(conn, "DEL candidate without contradict target")
            return None
        new_memory_id = add_memory(conn, user_id, c)
        obsolete_memory(conn, c.contradicts_memory_id, c.summary, new_memory_id)
        return new_memory_id

    nop_memory(conn, c.summary)
    return None


def dispatch_candidates(
    conn: sqlite3.Connection,
    user_id: str,
    candidates: list[MemoryCandidate | dict[str, Any]],
) -> list[str]:
    written: list[str] = []
    for candidate in candidates:
        memory_id = dispatch_candidate(conn, user_id, candidate)
        if memory_id:
            written.append(memory_id)
    return written


def write_link(
    conn: sqlite3.Connection,
    from_id: str,
    to_id: str,
    link_type: str,
    weight: float = 1.0,
) -> None:
    _write_link(conn, from_id, to_id, link_type, weight)
    _log_event(conn, "ADD", from_id, f"link:{link_type} -> {to_id}")


def upsert_goal(conn: sqlite3.Connection, user_id: str, goal: Goal | dict[str, Any]) -> str:
    g = goal if isinstance(goal, Goal) else Goal.model_validate({**goal, "user_id": user_id})
    now = _utc_now()
    conn.execute(
        """INSERT INTO goals (id, user_id, title, description, status, progress,
                              linked_memory_ids, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(id) DO UPDATE SET
                title = excluded.title,
                description = excluded.description,
                status = excluded.status,
                progress = excluded.progress,
                linked_memory_ids = excluded.linked_memory_ids,
                updated_at = excluded.updated_at""",
        (
            g.id,
            user_id,
            g.title,
            g.description,
            g.status.value,
            g.progress,
            json.dumps(g.linked_memory_ids),
            g.created_at.isoformat(),
            now,
        ),
    )
    return g.id


def update_goal_status(
    conn: sqlite3.Connection,
    goal_id: str,
    status: GoalStatus,
    progress: float | None = None,
) -> None:
    if progress is None:
        conn.execute("UPDATE goals SET status = ?, updated_at = ? WHERE id = ?", (status.value, _utc_now(), goal_id))
        return
    conn.execute(
        "UPDATE goals SET status = ?, progress = ?, updated_at = ? WHERE id = ?",
        (status.value, progress, _utc_now(), goal_id),
    )


def upsert_profile(
    conn: sqlite3.Connection,
    profile: AssistantProfile | dict[str, Any],
) -> None:
    p = profile if isinstance(profile, AssistantProfile) else AssistantProfile.model_validate(profile)
    now = _utc_now()
    conn.execute(
        """INSERT INTO assistant_profile (
               user_id, assistant_name, humor_mode, proactive_level, preferred_model,
               extraction_model, notification_style, memory_rules_version, steering_enabled,
               sycophancy_threshold, aggression_threshold, persona_snapshot_every_n_sessions,
               max_memories_in_prompt, gist_age_days, gist_count_limit, notes, created_at, updated_at
           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(user_id) DO UPDATE SET
               assistant_name = excluded.assistant_name,
               humor_mode = excluded.humor_mode,
               proactive_level = excluded.proactive_level,
               preferred_model = excluded.preferred_model,
               extraction_model = excluded.extraction_model,
               notification_style = excluded.notification_style,
               memory_rules_version = excluded.memory_rules_version,
               steering_enabled = excluded.steering_enabled,
               sycophancy_threshold = excluded.sycophancy_threshold,
               aggression_threshold = excluded.aggression_threshold,
               persona_snapshot_every_n_sessions = excluded.persona_snapshot_every_n_sessions,
               max_memories_in_prompt = excluded.max_memories_in_prompt,
               gist_age_days = excluded.gist_age_days,
               gist_count_limit = excluded.gist_count_limit,
               notes = excluded.notes,
               updated_at = excluded.updated_at""",
        (
            p.user_id,
            p.assistant_name,
            p.humor_mode,
            p.proactive_level,
            p.preferred_model,
            p.extraction_model,
            p.notification_style,
            p.memory_rules_version,
            int(p.steering_enabled),
            p.sycophancy_threshold,
            p.aggression_threshold,
            p.persona_snapshot_every_n_sessions,
            p.max_memories_in_prompt,
            p.gist_age_days,
            p.gist_count_limit,
            p.notes,
            p.created_at.isoformat(),
            now,
        ),
    )


def write_persona_snapshot(
    conn: sqlite3.Connection,
    user_id: str,
    vector: PersonaVector | dict[str, Any],
    session_id: str | None = None,
    steering_applied: bool = False,
    notes: str | None = None,
) -> str:
    v = vector if isinstance(vector, PersonaVector) else PersonaVector.model_validate(vector)
    snap_id = _new_id()
    conn.execute(
        """INSERT INTO persona_snapshots (
               id, user_id, session_id, sycophancy, aggression, hallucination_risk,
               status_sensitivity, openness, mood_valence, steering_applied, notes, created_at
           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            snap_id,
            user_id,
            session_id,
            v.sycophancy,
            v.aggression,
            v.hallucination_risk,
            v.status_sensitivity,
            v.openness,
            v.mood_valence,
            int(steering_applied),
            notes,
            _utc_now(),
        ),
    )
    return snap_id


def gist_memories(
    conn: sqlite3.Connection,
    user_id: str,
    memory_ids: list[str],
    gist_summary: str,
    gist_tags: list[str],
) -> str:
    gist_candidate = MemoryCandidate(
        operation=MemoryOperation.ADD,
        memory_type=MemoryType.CONTEXT,
        summary=gist_summary,
        confidence=0.90,
        importance=0.60,
        source_type="inference",
        semantic_tags=gist_tags,
    )
    gist_id = add_memory(conn, user_id, gist_candidate)

    for source_id in memory_ids:
        conn.execute(
            "UPDATE memory_items SET status = 'gist', updated_at = ? WHERE id = ?",
            (_utc_now(), source_id),
        )
        _write_link(conn, gist_id, source_id, "refines")
    _log_event(conn, MemoryOperation.ADD, gist_id, f"gist_of={len(memory_ids)}")
    return gist_id


def bootstrap_defaults(user_id: str = "default") -> None:
    init_db()
    defaults = AssistantProfile(user_id=user_id)
    with managed_connection() as conn:
        upsert_profile(conn, defaults)
