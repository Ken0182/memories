"""
mac_assistant/app/schema.py
─────────────────────────────────────────────────────────────────────────────
SQLite schema definition and database initialization.

All DDL lives here. Run this file directly to bootstrap the database:
    python -m mac_assistant.app.schema

Design decisions:
  - Pure SQLite, zero external DB dependencies for v1
  - JSON columns for variable-length structured fields (tags, vectors)
  - All timestamps stored as ISO-8601 TEXT for human readability
  - Soft-delete only: status column, never DROP rows
  - Indexes designed for the four main retrieval patterns:
      1. by type + status (category browsing)
      2. by updated_at DESC (recency)
      3. by importance DESC (salience)
      4. by user_id (multi-user ready)
─────────────────────────────────────────────────────────────────────────────
"""

import sqlite3
from pathlib import Path

# ── Database location ─────────────────────────────────────────────────────
_SCHEMA_DIR = Path(__file__).resolve().parent
DB_DIR  = _SCHEMA_DIR.parent / "data"
DB_PATH = DB_DIR / "assistant.db"


# ═══════════════════════════════════════════════════════════════════════════
# TABLE DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════

SCHEMA_SQL = """
-- ─────────────────────────────────────────────────────────────────────────
-- memory_items
-- Core memory record. One row = one durable fact / observation.
-- ─────────────────────────────────────────────────────────────────────────
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
    ON memory_items (user_id, updated_at DESC)
    WHERE status = 'active';

CREATE INDEX IF NOT EXISTS idx_memory_importance
    ON memory_items (user_id, importance DESC)
    WHERE status = 'active';

CREATE INDEX IF NOT EXISTS idx_memory_status
    ON memory_items (status);

-- ─────────────────────────────────────────────────────────────────────────
-- memory_semantic_tags
-- ─────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS memory_semantic_tags (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id  TEXT    NOT NULL REFERENCES memory_items(id) ON DELETE CASCADE,
    tag        TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_semtag_memory
    ON memory_semantic_tags (memory_id);

CREATE INDEX IF NOT EXISTS idx_semtag_tag
    ON memory_semantic_tags (tag);

CREATE UNIQUE INDEX IF NOT EXISTS idx_semtag_unique
    ON memory_semantic_tags (memory_id, tag);

-- ─────────────────────────────────────────────────────────────────────────
-- memory_emotional_tags
-- ─────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS memory_emotional_tags (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id  TEXT    NOT NULL REFERENCES memory_items(id) ON DELETE CASCADE,
    emotion    TEXT    NOT NULL,
    weight     REAL    NOT NULL DEFAULT 0.5
);

CREATE INDEX IF NOT EXISTS idx_emtag_memory
    ON memory_emotional_tags (memory_id);

CREATE INDEX IF NOT EXISTS idx_emtag_emotion
    ON memory_emotional_tags (emotion);

CREATE UNIQUE INDEX IF NOT EXISTS idx_emtag_unique
    ON memory_emotional_tags (memory_id, emotion);

-- ─────────────────────────────────────────────────────────────────────────
-- memory_links
-- ─────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS memory_links (
    id              TEXT    PRIMARY KEY,
    from_memory_id  TEXT    NOT NULL REFERENCES memory_items(id),
    to_memory_id    TEXT    NOT NULL REFERENCES memory_items(id),
    link_type       TEXT    NOT NULL,
    weight          REAL    NOT NULL DEFAULT 1.0,
    created_at      TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_link_from
    ON memory_links (from_memory_id);

CREATE INDEX IF NOT EXISTS idx_link_to
    ON memory_links (to_memory_id);

CREATE INDEX IF NOT EXISTS idx_link_type
    ON memory_links (link_type);

-- ─────────────────────────────────────────────────────────────────────────
-- memory_events
-- ─────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS memory_events (
    id          TEXT    PRIMARY KEY,
    memory_id   TEXT,
    event_type  TEXT    NOT NULL,
    notes       TEXT,
    created_at  TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_event_memory
    ON memory_events (memory_id);

CREATE INDEX IF NOT EXISTS idx_event_type
    ON memory_events (event_type);

CREATE INDEX IF NOT EXISTS idx_event_time
    ON memory_events (created_at DESC);

-- ─────────────────────────────────────────────────────────────────────────
-- goals
-- ─────────────────────────────────────────────────────────────────────────
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

CREATE INDEX IF NOT EXISTS idx_goals_user
    ON goals (user_id, status);

-- ─────────────────────────────────────────────────────────────────────────
-- assistant_profile
-- ─────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS assistant_profile (
    user_id               TEXT    PRIMARY KEY,
    humor_mode            TEXT    NOT NULL DEFAULT 'dry',
    proactive_level       TEXT    NOT NULL DEFAULT 'medium',
    preferred_model       TEXT    NOT NULL DEFAULT 'llama3.2',
    notification_style    TEXT    NOT NULL DEFAULT 'subtle',
    memory_rules_version  TEXT    NOT NULL DEFAULT '1.0',
    steering_enabled      INTEGER NOT NULL DEFAULT 1,
    sycophancy_threshold  REAL    NOT NULL DEFAULT 0.65,
    aggression_threshold  REAL    NOT NULL DEFAULT 0.65,
    created_at            TEXT    NOT NULL,
    updated_at            TEXT    NOT NULL
);

-- ─────────────────────────────────────────────────────────────────────────
-- persona_snapshots
-- ─────────────────────────────────────────────────────────────────────────
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

-- ─────────────────────────────────────────────────────────────────────────
-- embedding_cache (Phase 2 — future-ready)
-- ─────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS embedding_cache (
    memory_id     TEXT    PRIMARY KEY REFERENCES memory_items(id) ON DELETE CASCADE,
    model_name    TEXT    NOT NULL,
    embedding     BLOB    NOT NULL,
    dimensions    INTEGER NOT NULL,
    created_at    TEXT    NOT NULL,
    updated_at    TEXT    NOT NULL
);
"""


def init_db(db_path: str | Path = DB_PATH) -> sqlite3.Connection:
    """
    Create the database and all tables if they don't exist.
    Safe to call on every startup.
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-32000")
    conn.execute("PRAGMA temp_store=MEMORY")

    conn.executescript(SCHEMA_SQL)
    conn.commit()

    return conn


def get_connection(db_path: str | Path = DB_PATH) -> sqlite3.Connection:
    """Get a connection to an existing database."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def list_tables(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    return [r["name"] for r in rows]


def row_counts(conn: sqlite3.Connection) -> dict[str, int]:
    tables = list_tables(conn)
    return {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in tables}


def print_schema_summary(conn: sqlite3.Connection) -> None:
    print("\n╔══ OI Memory DB · Schema Summary ══════════════════════════╗")
    for table, count in row_counts(conn).items():
        print(f"║  {table:<30}  {count:>6} rows")
    print("╚═══════════════════════════════════════════════════════════╝\n")


if __name__ == "__main__":
    print(f"Initializing OI Memory DB at: {DB_PATH}")
    conn = init_db()
    print_schema_summary(conn)
    conn.close()
    print("✓ Database ready.")
