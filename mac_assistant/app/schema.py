from __future__ import annotations

import sqlite3
from pathlib import Path

DB_DIR = Path(__file__).resolve().parent.parent / "data"
DB_PATH = DB_DIR / "assistant.db"

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS memory_items (
    id               TEXT    PRIMARY KEY,
    user_id          TEXT    NOT NULL,
    memory_type      TEXT    NOT NULL,
    summary          TEXT    NOT NULL,
    raw_content      TEXT,
    source_type      TEXT    NOT NULL DEFAULT 'chat',
    source_ref       TEXT,
    confidence       REAL    NOT NULL DEFAULT 0.5,
    importance       REAL    NOT NULL DEFAULT 0.5,
    status           TEXT    NOT NULL DEFAULT 'active',
    evo_context_json      TEXT,
    proactive_hints_json  TEXT,
    persona_vector_json   TEXT,
    created_at       TEXT    NOT NULL,
    updated_at       TEXT    NOT NULL,
    last_accessed_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_memory_user_type
    ON memory_items (user_id, memory_type, status);
CREATE INDEX IF NOT EXISTS idx_memory_recency
    ON memory_items (user_id, updated_at DESC) WHERE status = 'active';
CREATE INDEX IF NOT EXISTS idx_memory_importance
    ON memory_items (user_id, importance DESC) WHERE status = 'active';
CREATE INDEX IF NOT EXISTS idx_memory_status
    ON memory_items (status);

CREATE TABLE IF NOT EXISTS memory_semantic_tags (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id  TEXT    NOT NULL REFERENCES memory_items(id) ON DELETE CASCADE,
    tag        TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_semtag_memory ON memory_semantic_tags (memory_id);
CREATE INDEX IF NOT EXISTS idx_semtag_tag ON memory_semantic_tags (tag);
CREATE UNIQUE INDEX IF NOT EXISTS idx_semtag_unique
    ON memory_semantic_tags (memory_id, tag);

CREATE TABLE IF NOT EXISTS memory_emotional_tags (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id  TEXT    NOT NULL REFERENCES memory_items(id) ON DELETE CASCADE,
    emotion    TEXT    NOT NULL,
    weight     REAL    NOT NULL DEFAULT 0.5
);
CREATE INDEX IF NOT EXISTS idx_emtag_memory ON memory_emotional_tags (memory_id);
CREATE INDEX IF NOT EXISTS idx_emtag_emotion ON memory_emotional_tags (emotion);
CREATE UNIQUE INDEX IF NOT EXISTS idx_emtag_unique
    ON memory_emotional_tags (memory_id, emotion);

CREATE TABLE IF NOT EXISTS memory_links (
    id              TEXT    PRIMARY KEY,
    from_memory_id  TEXT    NOT NULL REFERENCES memory_items(id),
    to_memory_id    TEXT    NOT NULL REFERENCES memory_items(id),
    link_type       TEXT    NOT NULL,
    weight          REAL    NOT NULL DEFAULT 1.0,
    created_at      TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_link_from ON memory_links (from_memory_id);
CREATE INDEX IF NOT EXISTS idx_link_to ON memory_links (to_memory_id);
CREATE INDEX IF NOT EXISTS idx_link_type ON memory_links (link_type);
CREATE UNIQUE INDEX IF NOT EXISTS idx_link_unique
    ON memory_links (from_memory_id, to_memory_id, link_type);

CREATE TABLE IF NOT EXISTS memory_events (
    id          TEXT    PRIMARY KEY,
    memory_id   TEXT,
    event_type  TEXT    NOT NULL,
    notes       TEXT,
    created_at  TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_event_memory ON memory_events (memory_id);
CREATE INDEX IF NOT EXISTS idx_event_type ON memory_events (event_type);
CREATE INDEX IF NOT EXISTS idx_event_time ON memory_events (created_at DESC);

CREATE TABLE IF NOT EXISTS goals (
    id                   TEXT    PRIMARY KEY,
    user_id              TEXT    NOT NULL,
    title                TEXT    NOT NULL,
    description          TEXT,
    status               TEXT    NOT NULL DEFAULT 'IN_PROGRESS',
    progress             REAL    NOT NULL DEFAULT 0.0,
    linked_memory_ids    TEXT    NOT NULL DEFAULT '[]',
    created_at           TEXT    NOT NULL,
    updated_at           TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_goals_user ON goals (user_id, status);

CREATE TABLE IF NOT EXISTS assistant_profile (
    user_id                           TEXT    PRIMARY KEY,
    assistant_name                    TEXT    NOT NULL DEFAULT 'OI',
    humor_mode                        TEXT    NOT NULL DEFAULT 'dry',
    proactive_level                   TEXT    NOT NULL DEFAULT 'medium',
    preferred_model                   TEXT    NOT NULL DEFAULT 'gpt-oss:20b',
    extraction_model                  TEXT    NOT NULL DEFAULT 'qwen3:8b',
    notification_style                TEXT    NOT NULL DEFAULT 'subtle',
    memory_rules_version              TEXT    NOT NULL DEFAULT '1.0',
    steering_enabled                  INTEGER NOT NULL DEFAULT 1,
    sycophancy_threshold              REAL    NOT NULL DEFAULT 0.65,
    aggression_threshold              REAL    NOT NULL DEFAULT 0.65,
    persona_snapshot_every_n_sessions INTEGER NOT NULL DEFAULT 5,
    max_memories_in_prompt            INTEGER NOT NULL DEFAULT 7,
    gist_age_days                     INTEGER NOT NULL DEFAULT 30,
    gist_count_limit                  INTEGER NOT NULL DEFAULT 50,
    notes                             TEXT,
    created_at                        TEXT    NOT NULL,
    updated_at                        TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS persona_snapshots (
    id                  TEXT    PRIMARY KEY,
    user_id             TEXT    NOT NULL,
    session_id          TEXT,
    sycophancy          REAL    NOT NULL DEFAULT 0.0,
    aggression          REAL    NOT NULL DEFAULT 0.0,
    hallucination_risk  REAL    NOT NULL DEFAULT 0.0,
    status_sensitivity  REAL    NOT NULL DEFAULT 0.0,
    openness            REAL    NOT NULL DEFAULT 0.5,
    mood_valence        REAL    NOT NULL DEFAULT 0.5,
    steering_applied    INTEGER NOT NULL DEFAULT 0,
    notes               TEXT,
    created_at          TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_persona_user_time
    ON persona_snapshots (user_id, created_at DESC);

CREATE TABLE IF NOT EXISTS embedding_cache (
    memory_id     TEXT    PRIMARY KEY REFERENCES memory_items(id) ON DELETE CASCADE,
    model_name    TEXT    NOT NULL,
    embedding     BLOB    NOT NULL,
    dimensions    INTEGER NOT NULL,
    created_at    TEXT    NOT NULL,
    updated_at    TEXT    NOT NULL
);
"""


def _configure_connection(conn: sqlite3.Connection) -> sqlite3.Connection:
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-32000")
    conn.execute("PRAGMA temp_store=MEMORY")
    return conn


def init_db(db_path: str | Path = DB_PATH) -> sqlite3.Connection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    _configure_connection(conn)
    conn.executescript(SCHEMA_SQL)
    conn.commit()
    return conn


def get_connection(db_path: str | Path = DB_PATH) -> sqlite3.Connection:
    db_path = Path(db_path)
    conn = sqlite3.connect(str(db_path))
    return _configure_connection(conn)


def list_tables(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    return [row["name"] for row in rows]


def row_counts(conn: sqlite3.Connection) -> dict[str, int]:
    counts: dict[str, int] = {}
    for table in list_tables(conn):
        counts[table] = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    return counts


if __name__ == "__main__":
    connection = init_db()
    print(f"Initialized database at: {DB_PATH}")
    for table_name, count in row_counts(connection).items():
        print(f"{table_name:<30} {count:>6}")
    connection.close()
