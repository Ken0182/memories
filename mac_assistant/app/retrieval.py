"""
mac_assistant/app/retrieval.py
─────────────────────────────────────────────────────────────────────────────
All read queries from SQLite. This module never writes to the database.

Retrieval patterns (matching OI architecture):
  1. by_type        — category browsing (goals, preferences, facts…)
  2. by_recency     — most recently updated active memories
  3. by_tags        — semantic tag intersection search
  4. by_emotion     — emotional resonance matching
  5. top_for_prompt — recency × importance × emotion (main prompt builder feed)
  6. multi_hop      — follow link chains from a seed memory
  7. persona_trend  — recent persona snapshots for drift detection
  8. search_by_text — text search (fallback for simple setups)

Phase 2 will add:
  9. by_embedding   — semantic similarity via stored vectors
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta
from typing import Optional

from mac_assistant.app.schema import DB_PATH, get_connection


# ─────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────

def _row_to_memory(row: sqlite3.Row) -> dict:
    """Convert a raw DB row to a clean dict. Deserializes JSON columns."""
    d = dict(row)
    for key in ("evo_context_json", "proactive_hints_json", "persona_vector_json"):
        raw = d.pop(key, None)
        field = key.replace("_json", "")
        d[field] = json.loads(raw) if raw else None
    return d


def _attach_tags(conn: sqlite3.Connection, memory: dict) -> dict:
    """Attach semantic + emotional tags to a memory dict."""
    mid = memory["id"]

    sem = conn.execute(
        "SELECT tag FROM memory_semantic_tags WHERE memory_id = ?", (mid,)
    ).fetchall()
    memory["semantic_tags"] = [r["tag"] for r in sem]

    emo = conn.execute(
        "SELECT emotion, weight FROM memory_emotional_tags WHERE memory_id = ?", (mid,)
    ).fetchall()
    memory["emotional_tags"] = [{"emotion": r["emotion"], "weight": r["weight"]} for r in emo]

    return memory


def _score(memory: dict,
           recency_weight: float = 0.4,
           importance_weight: float = 0.4,
           emotion_weight: float = 0.2,
           reference_emotion: Optional[str] = None) -> float:
    """
    OI relevance score: recency × salience × emotional match
    """
    updated = memory.get("updated_at", "")
    try:
        dt = datetime.fromisoformat(updated)
        days_old = (datetime.utcnow() - dt).days
        recency = 1.0 / (1.0 + days_old)
    except Exception:
        recency = 0.0

    importance = memory.get("importance", 0.5)

    emotion_score = 0.0
    if reference_emotion:
        for etag in memory.get("emotional_tags", []):
            if etag.get("emotion") == reference_emotion.lower():
                emotion_score = etag.get("weight", 0)
                break

    return (
        recency_weight * recency +
        importance_weight * importance +
        emotion_weight * (emotion_score if reference_emotion else importance)
    )


# ═══════════════════════════════════════════════════════════════════════════
# PRIMARY RETRIEVAL FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def by_type(conn: sqlite3.Connection,
            user_id: str,
            memory_type: str,
            limit: int = 10,
            include_obsolete: bool = False) -> list[dict]:
    """Retrieve memories by type, ordered by importance then recency."""
    statuses = "('active')" if not include_obsolete else "('active','obsolete','gist')"
    rows = conn.execute(
        f"""SELECT * FROM memory_items
            WHERE user_id = ? AND memory_type = ? AND status IN {statuses}
            ORDER BY importance DESC, updated_at DESC
            LIMIT ?""",
        (user_id, memory_type, limit)
    ).fetchall()

    return [_attach_tags(conn, _row_to_memory(r)) for r in rows]


def by_recency(conn: sqlite3.Connection,
               user_id: str,
               limit: int = 20,
               since_hours: Optional[int] = None) -> list[dict]:
    """Most recently updated active memories."""
    if since_hours:
        cutoff = (datetime.utcnow() - timedelta(hours=since_hours)).isoformat()
        rows = conn.execute(
            """SELECT * FROM memory_items
               WHERE user_id = ? AND status = 'active' AND updated_at >= ?
               ORDER BY updated_at DESC LIMIT ?""",
            (user_id, cutoff, limit)
        ).fetchall()
    else:
        rows = conn.execute(
            """SELECT * FROM memory_items
               WHERE user_id = ? AND status = 'active'
               ORDER BY updated_at DESC LIMIT ?""",
            (user_id, limit)
        ).fetchall()

    return [_attach_tags(conn, _row_to_memory(r)) for r in rows]


def by_tags(conn: sqlite3.Connection,
            user_id: str,
            tags: list[str],
            require_all: bool = False,
            limit: int = 15) -> list[dict]:
    """Find memories that share semantic tags with the query."""
    if not tags:
        return []

    tags_lower = [t.lower().strip() for t in tags]
    placeholders = ",".join("?" * len(tags_lower))

    if require_all:
        rows = conn.execute(
            f"""SELECT mi.*, COUNT(mst.tag) as tag_hits
                FROM memory_items mi
                JOIN memory_semantic_tags mst ON mi.id = mst.memory_id
                WHERE mi.user_id = ? AND mi.status = 'active'
                  AND mst.tag IN ({placeholders})
                GROUP BY mi.id
                HAVING COUNT(DISTINCT mst.tag) = ?
                ORDER BY tag_hits DESC, mi.importance DESC
                LIMIT ?""",
            (user_id, *tags_lower, len(tags_lower), limit)
        ).fetchall()
    else:
        rows = conn.execute(
            f"""SELECT mi.*, COUNT(mst.tag) as tag_hits
                FROM memory_items mi
                JOIN memory_semantic_tags mst ON mi.id = mst.memory_id
                WHERE mi.user_id = ? AND mi.status = 'active'
                  AND mst.tag IN ({placeholders})
                GROUP BY mi.id
                ORDER BY tag_hits DESC, mi.importance DESC
                LIMIT ?""",
            (user_id, *tags_lower, limit)
        ).fetchall()

    return [_attach_tags(conn, _row_to_memory(r)) for r in rows]


def by_emotion(conn: sqlite3.Connection,
               user_id: str,
               emotion: str,
               min_weight: float = 0.4,
               limit: int = 10) -> list[dict]:
    """Find memories tagged with a specific emotion above a weight threshold."""
    rows = conn.execute(
        """SELECT mi.*
           FROM memory_items mi
           JOIN memory_emotional_tags met ON mi.id = met.memory_id
           WHERE mi.user_id = ? AND mi.status = 'active'
             AND met.emotion = ? AND met.weight >= ?
           ORDER BY met.weight DESC, mi.importance DESC
           LIMIT ?""",
        (user_id, emotion.lower(), min_weight, limit)
    ).fetchall()

    return [_attach_tags(conn, _row_to_memory(r)) for r in rows]


def search_by_text(conn: sqlite3.Connection,
                  user_id: str,
                  query: str,
                  limit: int = 10) -> list[dict]:
    """Text search on summary, raw_content, and tags. Fallback for simple retrieval."""
    like_q = f"%{query.lower()}%"
    rows = conn.execute(
        """SELECT DISTINCT mi.*
           FROM memory_items mi
           LEFT JOIN memory_semantic_tags mst ON mi.id = mst.memory_id
           WHERE mi.user_id = ? AND mi.status = 'active'
             AND (
               lower(mi.summary) LIKE ?
               OR lower(mi.raw_content) LIKE ?
               OR lower(COALESCE(mst.tag, '')) LIKE ?
             )
           ORDER BY mi.importance DESC, mi.updated_at DESC
           LIMIT ?""",
        (user_id, like_q, like_q, like_q, limit)
    ).fetchall()
    return [_attach_tags(conn, _row_to_memory(r)) for r in rows]


def top_for_prompt(conn: sqlite3.Connection,
                   user_id: str,
                   limit: int = 7,
                   reference_emotion: Optional[str] = None,
                   reference_tags: Optional[list[str]] = None) -> list[dict]:
    """
    Main retrieval function for prompt_builder.
    Fetches candidates and re-ranks by recency × importance × emotional match.
    """
    candidate_limit = min(limit * 4, 60)
    rows = conn.execute(
        """SELECT * FROM memory_items
           WHERE user_id = ? AND status = 'active'
           ORDER BY importance DESC, updated_at DESC
           LIMIT ?""",
        (user_id, candidate_limit)
    ).fetchall()

    memories = [_attach_tags(conn, _row_to_memory(r)) for r in rows]

    if reference_tags:
        tag_memories = by_tags(conn, user_id, reference_tags, limit=20)
        existing_ids = {m["id"] for m in memories}
        for tm in tag_memories:
            if tm["id"] not in existing_ids:
                memories.append(tm)

    for m in memories:
        m["_score"] = _score(m, reference_emotion=reference_emotion)

    memories.sort(key=lambda x: x["_score"], reverse=True)

    result = memories[:limit]
    for m in result:
        m["relevance_score"] = round(m.pop("_score", 0), 4)

    return result


def retrieve_relevant_memories(user_id: str,
                               user_message: str,
                               limit: int = 5,
                               conn: Optional[sqlite3.Connection] = None) -> list[dict]:
    """
    High-level API: try text search first, fall back to recent memories.
    Use top_for_prompt when you have a full session context.
    """
    close = False
    if conn is None:
        conn = get_connection()
        close = True

    try:
        if user_message and len(user_message.strip()) >= 3:
            hits = search_by_text(conn, user_id, user_message, limit=limit)
            if hits:
                return hits
        return by_recency(conn, user_id, limit=limit)
    finally:
        if close:
            conn.close()


# ═══════════════════════════════════════════════════════════════════════════
# MULTI-HOP GRAPH TRAVERSAL
# ═══════════════════════════════════════════════════════════════════════════

def multi_hop(conn: sqlite3.Connection,
              seed_memory_id: str,
              link_types: Optional[list[str]] = None,
              max_depth: int = 2,
              max_results: int = 10) -> list[dict]:
    """Follow memory links from a seed up to max_depth hops."""
    visited = set()
    queue = [(seed_memory_id, 0)]
    results = []

    while queue and len(results) < max_results:
        current_id, depth = queue.pop(0)
        if current_id in visited or depth > max_depth:
            continue
        visited.add(current_id)

        if depth > 0:
            row = conn.execute(
                "SELECT * FROM memory_items WHERE id = ? AND status != 'obsolete'",
                (current_id,)
            ).fetchone()
            if row:
                results.append(_attach_tags(conn, _row_to_memory(row)))

        if depth < max_depth:
            if link_types:
                placeholders = ",".join("?" * len(link_types))
                links = conn.execute(
                    f"""SELECT to_memory_id FROM memory_links
                        WHERE from_memory_id = ? AND link_type IN ({placeholders})""",
                    (current_id, *link_types)
                ).fetchall()
            else:
                links = conn.execute(
                    "SELECT to_memory_id FROM memory_links WHERE from_memory_id = ?",
                    (current_id,)
                ).fetchall()

            for link in links:
                next_id = link["to_memory_id"]
                if next_id not in visited:
                    queue.append((next_id, depth + 1))

    return results


# ═══════════════════════════════════════════════════════════════════════════
# GOALS
# ═══════════════════════════════════════════════════════════════════════════

def get_active_goals(conn: sqlite3.Connection,
                    user_id: str) -> list[dict]:
    """All non-completed, non-abandoned goals."""
    rows = conn.execute(
        """SELECT * FROM goals
           WHERE user_id = ? AND status NOT IN ('COMPLETED', 'ABANDONED')
           ORDER BY updated_at DESC""",
        (user_id,)
    ).fetchall()

    result = []
    for r in rows:
        d = dict(r)
        d["linked_memory_ids"] = json.loads(d.get("linked_memory_ids", "[]"))
        result.append(d)
    return result


# ═══════════════════════════════════════════════════════════════════════════
# PERSONA DRIFT
# ═══════════════════════════════════════════════════════════════════════════

def persona_trend(conn: sqlite3.Connection,
                  user_id: str,
                  last_n: int = 10) -> list[dict]:
    """Last N persona snapshots, oldest first."""
    rows = conn.execute(
        """SELECT * FROM persona_snapshots
           WHERE user_id = ?
           ORDER BY created_at DESC
           LIMIT ?""",
        (user_id, last_n)
    ).fetchall()
    return list(reversed([dict(r) for r in rows]))


def latest_persona(conn: sqlite3.Connection,
                   user_id: str) -> Optional[dict]:
    """Most recent persona snapshot."""
    row = conn.execute(
        """SELECT * FROM persona_snapshots
           WHERE user_id = ?
           ORDER BY created_at DESC LIMIT 1""",
        (user_id,)
    ).fetchone()
    return dict(row) if row else None


def persona_needs_steering(conn: sqlite3.Connection,
                           user_id: str,
                           sycophancy_threshold: float = 0.65,
                           aggression_threshold: float = 0.65) -> tuple[bool, list[str]]:
    """Check if latest persona snapshot is above any warn threshold."""
    snap = latest_persona(conn, user_id)
    if not snap:
        return False, []

    flags = []
    if snap.get("sycophancy", 0) > sycophancy_threshold:
        flags.append(f"sycophancy:{snap['sycophancy']:.2f}")
    if snap.get("aggression", 0) > aggression_threshold:
        flags.append(f"aggression:{snap['aggression']:.2f}")
    if snap.get("status_sensitivity", 0) > sycophancy_threshold:
        flags.append(f"status_sensitivity:{snap['status_sensitivity']:.2f}")

    return len(flags) > 0, flags


# ═══════════════════════════════════════════════════════════════════════════
# ASSISTANT PROFILE
# ═══════════════════════════════════════════════════════════════════════════

def get_profile(conn: sqlite3.Connection,
                user_id: str) -> Optional[dict]:
    row = conn.execute(
        "SELECT * FROM assistant_profile WHERE user_id = ?", (user_id,)
    ).fetchone()
    return dict(row) if row else None


# ═══════════════════════════════════════════════════════════════════════════
# CONFLICT DETECTION & AUDIT
# ═══════════════════════════════════════════════════════════════════════════

def find_similar_active(conn: sqlite3.Connection,
                        user_id: str,
                        memory_type: str,
                        tags: list[str],
                        limit: int = 5) -> list[dict]:
    """Find existing active memories that might conflict with a new candidate."""
    return by_tags(conn, user_id, tags, require_all=False, limit=limit)


def audit_log(conn: sqlite3.Connection,
              memory_id: Optional[str] = None,
              limit: int = 50) -> list[dict]:
    """Retrieve audit events."""
    if memory_id:
        rows = conn.execute(
            """SELECT * FROM memory_events WHERE memory_id = ?
               ORDER BY created_at DESC LIMIT ?""",
            (memory_id, limit)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM memory_events ORDER BY created_at DESC LIMIT ?",
            (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def list_all_memories(conn: sqlite3.Connection,
                      user_id: Optional[str] = None) -> list[dict]:
    """List all memories, optionally filtered by user."""
    if user_id:
        rows = conn.execute(
            """SELECT id, memory_type, summary, status, importance, created_at
               FROM memory_items WHERE user_id = ? ORDER BY id DESC""",
            (user_id,)
        ).fetchall()
    else:
        rows = conn.execute(
            """SELECT id, user_id, memory_type, summary, status, importance, created_at
               FROM memory_items ORDER BY id DESC"""
        ).fetchall()
    return [dict(r) for r in rows]
