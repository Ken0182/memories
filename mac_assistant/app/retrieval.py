from __future__ import annotations

import json
import sqlite3
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Any

from mac_assistant.app.schema import DB_PATH, get_connection


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _row_to_memory(row: sqlite3.Row) -> dict[str, Any]:
    item = dict(row)
    for key in ("evo_context_json", "proactive_hints_json", "persona_vector_json"):
        raw = item.pop(key, None)
        item[key.replace("_json", "")] = json.loads(raw) if raw else None
    return item


def _attach_tags(conn: sqlite3.Connection, memory: dict[str, Any]) -> dict[str, Any]:
    memory_id = memory["id"]
    sem_rows = conn.execute(
        "SELECT tag FROM memory_semantic_tags WHERE memory_id = ? ORDER BY tag",
        (memory_id,),
    ).fetchall()
    emo_rows = conn.execute(
        """SELECT emotion, weight FROM memory_emotional_tags
           WHERE memory_id = ? ORDER BY weight DESC""",
        (memory_id,),
    ).fetchall()
    memory["semantic_tags"] = [r["tag"] for r in sem_rows]
    memory["emotional_tags"] = [{"emotion": r["emotion"], "weight": r["weight"]} for r in emo_rows]
    return memory


def _score(
    memory: dict[str, Any],
    now: datetime,
    reference_emotion: str | None = None,
    recency_weight: float = 0.4,
    importance_weight: float = 0.4,
    emotion_weight: float = 0.2,
) -> float:
    updated_dt = _parse_iso(memory.get("updated_at"))
    if updated_dt is None:
        recency = 0.0
    else:
        days_old = max((now - updated_dt).days, 0)
        recency = 1.0 / (1.0 + days_old)

    importance = float(memory.get("importance", 0.5))
    emotion_score = 0.0
    if reference_emotion:
        ref = reference_emotion.lower().strip()
        emotion_score = max(
            (float(e["weight"]) for e in memory.get("emotional_tags", []) if e["emotion"] == ref),
            default=0.0,
        )

    return (
        recency_weight * recency
        + importance_weight * importance
        + emotion_weight * (emotion_score if reference_emotion else importance)
    )


def by_type(
    conn: sqlite3.Connection,
    user_id: str,
    memory_type: str,
    limit: int = 10,
    include_obsolete: bool = False,
) -> list[dict[str, Any]]:
    statuses = ("active",) if not include_obsolete else ("active", "obsolete", "gist")
    placeholders = ",".join("?" for _ in statuses)
    rows = conn.execute(
        f"""SELECT * FROM memory_items
            WHERE user_id = ? AND memory_type = ? AND status IN ({placeholders})
            ORDER BY importance DESC, updated_at DESC
            LIMIT ?""",
        (user_id, memory_type, *statuses, limit),
    ).fetchall()
    return [_attach_tags(conn, _row_to_memory(row)) for row in rows]


def by_recency(
    conn: sqlite3.Connection,
    user_id: str,
    limit: int = 20,
    since_hours: int | None = None,
) -> list[dict[str, Any]]:
    if since_hours:
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=since_hours)).isoformat()
        rows = conn.execute(
            """SELECT * FROM memory_items
               WHERE user_id = ? AND status = 'active' AND updated_at >= ?
               ORDER BY updated_at DESC LIMIT ?""",
            (user_id, cutoff, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            """SELECT * FROM memory_items
               WHERE user_id = ? AND status = 'active'
               ORDER BY updated_at DESC LIMIT ?""",
            (user_id, limit),
        ).fetchall()
    return [_attach_tags(conn, _row_to_memory(row)) for row in rows]


def by_tags(
    conn: sqlite3.Connection,
    user_id: str,
    tags: list[str],
    require_all: bool = False,
    limit: int = 15,
) -> list[dict[str, Any]]:
    normalized = [t.lower().strip().replace(" ", "-") for t in tags if t.strip()]
    if not normalized:
        return []
    placeholders = ",".join("?" * len(normalized))
    if require_all:
        rows = conn.execute(
            f"""SELECT mi.*, COUNT(DISTINCT mst.tag) AS tag_hits
                FROM memory_items mi
                JOIN memory_semantic_tags mst ON mi.id = mst.memory_id
                WHERE mi.user_id = ? AND mi.status = 'active' AND mst.tag IN ({placeholders})
                GROUP BY mi.id
                HAVING COUNT(DISTINCT mst.tag) = ?
                ORDER BY tag_hits DESC, mi.importance DESC
                LIMIT ?""",
            (user_id, *normalized, len(normalized), limit),
        ).fetchall()
    else:
        rows = conn.execute(
            f"""SELECT mi.*, COUNT(DISTINCT mst.tag) AS tag_hits
                FROM memory_items mi
                JOIN memory_semantic_tags mst ON mi.id = mst.memory_id
                WHERE mi.user_id = ? AND mi.status = 'active' AND mst.tag IN ({placeholders})
                GROUP BY mi.id
                ORDER BY tag_hits DESC, mi.importance DESC
                LIMIT ?""",
            (user_id, *normalized, limit),
        ).fetchall()
    return [_attach_tags(conn, _row_to_memory(row)) for row in rows]


def by_emotion(
    conn: sqlite3.Connection,
    user_id: str,
    emotion: str,
    min_weight: float = 0.4,
    limit: int = 10,
) -> list[dict[str, Any]]:
    rows = conn.execute(
        """SELECT mi.*
           FROM memory_items mi
           JOIN memory_emotional_tags met ON mi.id = met.memory_id
           WHERE mi.user_id = ? AND mi.status = 'active'
             AND met.emotion = ? AND met.weight >= ?
           ORDER BY met.weight DESC, mi.importance DESC
           LIMIT ?""",
        (user_id, emotion.lower().strip(), min_weight, limit),
    ).fetchall()
    return [_attach_tags(conn, _row_to_memory(row)) for row in rows]


def by_salience(
    conn: sqlite3.Connection,
    user_id: str,
    limit: int = 10,
) -> list[dict[str, Any]]:
    rows = conn.execute(
        """SELECT * FROM memory_items
           WHERE user_id = ? AND status = 'active'
           ORDER BY importance DESC, updated_at DESC
           LIMIT ?""",
        (user_id, limit),
    ).fetchall()
    return [_attach_tags(conn, _row_to_memory(row)) for row in rows]


def top_for_prompt(
    conn: sqlite3.Connection,
    user_id: str,
    limit: int = 7,
    reference_emotion: str | None = None,
    reference_tags: list[str] | None = None,
) -> list[dict[str, Any]]:
    pool_size = min(60, max(limit * 4, limit + 5))
    memories = by_salience(conn, user_id, limit=pool_size)

    if reference_tags:
        boosted = by_tags(conn, user_id, reference_tags, require_all=False, limit=20)
        existing = {m["id"] for m in memories}
        for memory in boosted:
            if memory["id"] not in existing:
                memories.append(memory)

    now = datetime.now(timezone.utc)
    for memory in memories:
        memory["_score"] = _score(memory, now=now, reference_emotion=reference_emotion)
    memories.sort(key=lambda m: m["_score"], reverse=True)

    result = memories[:limit]
    for memory in result:
        memory["relevance_score"] = round(float(memory.pop("_score")), 4)
    return result


def multi_hop(
    conn: sqlite3.Connection,
    seed_memory_id: str,
    link_types: list[str] | None = None,
    max_depth: int = 2,
    max_results: int = 10,
) -> list[dict[str, Any]]:
    queue: deque[tuple[str, int]] = deque([(seed_memory_id, 0)])
    visited: set[str] = set()
    output: list[dict[str, Any]] = []

    while queue and len(output) < max_results:
        current_id, depth = queue.popleft()
        if current_id in visited or depth > max_depth:
            continue
        visited.add(current_id)

        if depth > 0:
            row = conn.execute(
                "SELECT * FROM memory_items WHERE id = ? AND status != 'obsolete'",
                (current_id,),
            ).fetchone()
            if row:
                output.append(_attach_tags(conn, _row_to_memory(row)))

        if depth >= max_depth:
            continue

        if link_types:
            placeholders = ",".join("?" * len(link_types))
            links = conn.execute(
                f"""SELECT to_memory_id FROM memory_links
                    WHERE from_memory_id = ? AND link_type IN ({placeholders})""",
                (current_id, *link_types),
            ).fetchall()
        else:
            links = conn.execute(
                "SELECT to_memory_id FROM memory_links WHERE from_memory_id = ?",
                (current_id,),
            ).fetchall()
        for link in links:
            nxt = link["to_memory_id"]
            if nxt not in visited:
                queue.append((nxt, depth + 1))
    return output


def get_active_goals(conn: sqlite3.Connection, user_id: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        """SELECT * FROM goals
           WHERE user_id = ? AND status NOT IN ('COMPLETED', 'ABANDONED')
           ORDER BY updated_at DESC""",
        (user_id,),
    ).fetchall()
    result: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["linked_memory_ids"] = json.loads(item.get("linked_memory_ids", "[]"))
        result.append(item)
    return result


def persona_trend(conn: sqlite3.Connection, user_id: str, last_n: int = 10) -> list[dict[str, Any]]:
    rows = conn.execute(
        """SELECT * FROM persona_snapshots
           WHERE user_id = ?
           ORDER BY created_at DESC
           LIMIT ?""",
        (user_id, last_n),
    ).fetchall()
    return list(reversed([dict(row) for row in rows]))


def latest_persona(conn: sqlite3.Connection, user_id: str) -> dict[str, Any] | None:
    row = conn.execute(
        """SELECT * FROM persona_snapshots
           WHERE user_id = ?
           ORDER BY created_at DESC
           LIMIT 1""",
        (user_id,),
    ).fetchone()
    return dict(row) if row else None


def persona_needs_steering(
    conn: sqlite3.Connection,
    user_id: str,
    sycophancy_threshold: float = 0.65,
    aggression_threshold: float = 0.65,
) -> tuple[bool, list[str]]:
    snapshot = latest_persona(conn, user_id)
    if not snapshot:
        return False, []
    flags: list[str] = []
    if snapshot.get("sycophancy", 0.0) > sycophancy_threshold:
        flags.append(f"sycophancy:{snapshot['sycophancy']:.2f}")
    if snapshot.get("aggression", 0.0) > aggression_threshold:
        flags.append(f"aggression:{snapshot['aggression']:.2f}")
    if snapshot.get("status_sensitivity", 0.0) > sycophancy_threshold:
        flags.append(f"status_sensitivity:{snapshot['status_sensitivity']:.2f}")
    return bool(flags), flags


def get_profile(conn: sqlite3.Connection, user_id: str) -> dict[str, Any] | None:
    row = conn.execute("SELECT * FROM assistant_profile WHERE user_id = ?", (user_id,)).fetchone()
    return dict(row) if row else None


def find_similar_active(
    conn: sqlite3.Connection,
    user_id: str,
    memory_type: str,
    tags: list[str],
    limit: int = 5,
) -> list[dict[str, Any]]:
    if not tags:
        rows = conn.execute(
            """SELECT * FROM memory_items
               WHERE user_id = ? AND memory_type = ? AND status = 'active'
               ORDER BY importance DESC, updated_at DESC
               LIMIT ?""",
            (user_id, memory_type, limit),
        ).fetchall()
        return [_attach_tags(conn, _row_to_memory(row)) for row in rows]

    matches = by_tags(conn, user_id, tags, require_all=False, limit=max(10, limit * 2))
    filtered = [m for m in matches if m.get("memory_type") == memory_type]
    return filtered[:limit]


def audit_log(
    conn: sqlite3.Connection,
    memory_id: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    if memory_id:
        rows = conn.execute(
            """SELECT * FROM memory_events
               WHERE memory_id = ?
               ORDER BY created_at DESC
               LIMIT ?""",
            (memory_id, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            """SELECT * FROM memory_events
               ORDER BY created_at DESC
               LIMIT ?""",
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def search_memories_by_text(
    conn: sqlite3.Connection,
    user_id: str,
    query: str,
    limit: int = 5,
) -> list[dict[str, Any]]:
    like_query = f"%{query.lower()}%"
    rows = conn.execute(
        """SELECT DISTINCT mi.*
           FROM memory_items mi
           LEFT JOIN memory_semantic_tags mst ON mi.id = mst.memory_id
           WHERE mi.user_id = ? AND mi.status = 'active'
             AND (
                 lower(mi.summary) LIKE ?
                 OR lower(COALESCE(mi.raw_content, '')) LIKE ?
                 OR lower(COALESCE(mst.tag, '')) LIKE ?
             )
           ORDER BY mi.importance DESC, mi.updated_at DESC
           LIMIT ?""",
        (user_id, like_query, like_query, like_query, limit),
    ).fetchall()
    return [_attach_tags(conn, _row_to_memory(row)) for row in rows]


def retrieve_relevant_memories(user_id: str, user_message: str, limit: int = 5) -> list[dict[str, Any]]:
    with get_connection(DB_PATH) as conn:
        text_hits = search_memories_by_text(conn, user_id, user_message, limit=limit)
        if text_hits:
            return text_hits
        return by_recency(conn, user_id, limit=limit)
