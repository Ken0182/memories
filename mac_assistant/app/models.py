"""
mac_assistant/app/models.py
─────────────────────────────────────────────────────────────────────────────
Pydantic v2 models for every memory object in the OI memory system.
Install once on your Mac:  pip install pydantic

These models are the single source of truth for:
  - What a memory candidate looks like when the LLM proposes it
  - What gets validated before any write to SQLite
  - What gets returned from retrieval queries

Design principles (from OI architecture docs):
  - Model PROPOSES → our code VALIDATES → our code STORES
  - Recency wins on contradiction (temporal consistency)
  - Emotional + semantic tags are first-class, not optional extras
  - Every memory links to others via typed relationships
  - Persona vectors are tracked separately from episodic memory
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import uuid4

# ── Pydantic v2 imports ───────────────────────────────────────────────────
# pip install pydantic
from pydantic import BaseModel, Field, field_validator, model_validator


# ═══════════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════════

class MemoryOperation(str, Enum):
    """
    The four operations from the OI extraction loop.
    Every memory candidate the LLM returns MUST declare one of these.

    ADD   — brand new concept, no existing match
    UPD   — enriches an existing memory (same entity, more info)
    DEL   — contradicts an existing memory (recency wins)
    NOP   — redundant / trivial, do not write to DB
    """
    ADD = "ADD"
    UPD = "UPD"
    DEL = "DEL"
    NOP = "NOP"


class MemoryType(str, Enum):
    """
    Layer 1 of the Cloud (Bulut) classification.
    Determines which retrieval bucket the memory lives in.
    """
    PREFERENCE      = "preference"       # likes, dislikes, habits
    FACT            = "fact"             # biographical, environmental facts
    GOAL            = "goal"             # long-term aspirations (Humanistic class)
    STRATEGY        = "strategy"         # evolutionary role patterns (Functionalist class)
    EMOTIONAL_STATE = "emotional_state"  # PAD-model tagged moments
    RELATIONSHIP    = "relationship"     # kin graph, social contacts
    SKILL           = "skill"            # competencies, learning arcs
    BELIEF          = "belief"           # values, worldview
    CONTEXT         = "context"          # environmental / situational
    PERSONA         = "persona"          # AI persona drift readings


class MemoryStatus(str, Enum):
    """
    Lifecycle status of a memory record.
    NEVER hard-delete — archive instead to preserve trend history.
    """
    ACTIVE   = "active"    # live, in retrieval pool
    OBSOLETE = "obsolete"  # contradicted by newer data, archived
    GIST     = "gist"      # compressed summary of older memories
    ARCHIVED = "archived"  # manually removed from active pool


class SourceType(str, Enum):
    """Where did this memory come from."""
    CHAT       = "chat"        # normal conversation turn
    FILE       = "file"        # file the user shared
    SCREEN     = "screen"      # Mac screen context (phase 2)
    AUDIO      = "audio"       # ambient audio sensor (phase 2)
    SYSTEM     = "system"      # OS event / calendar (phase 2)
    MANUAL     = "manual"      # user explicitly told assistant to remember
    INFERENCE  = "inference"   # AI inferred, not directly stated


class LinkType(str, Enum):
    """
    Types of relationships between memory nodes.
    Used in memory_links table for graph traversal.
    """
    REFINES      = "refines"       # new memory adds detail to old
    CONTRADICTS  = "contradicts"   # new memory conflicts with old
    CAUSED_BY    = "caused_by"     # causal chain
    SAME_ENTITY  = "same_entity"   # different facts about same person/thing
    SAME_THEME   = "same_theme"    # topically related
    TEMPORAL_SEQ = "temporal_seq"  # happened before/after


class SelectionType(str, Enum):
    """Evolutionary motivation layer (Functionalist class)."""
    INDIVIDUAL  = "individual"   # personal gain / status
    KIN         = "kin"          # protecting relatives / ingroup
    RECIPROCAL  = "reciprocal"   # social contract, tit-for-tat
    UNKNOWN     = "unknown"


class GoalStatus(str, Enum):
    IN_PROGRESS = "IN_PROGRESS"
    PASSING     = "PASSING"
    STALLED     = "STALLED"
    AT_RISK     = "AT_RISK"
    COMPLETED   = "COMPLETED"
    ABANDONED   = "ABANDONED"


# ═══════════════════════════════════════════════════════════════════════════
# SUB-MODELS
# ═══════════════════════════════════════════════════════════════════════════

class EmotionalTag(BaseModel):
    """
    PAD-inspired emotional label with weight.
    Stored in memory_emotional_tags table.
    """
    emotion: str = Field(..., min_length=1, max_length=64,
                         description="e.g. 'stress', 'playful', 'comfort', 'nostalgia'")
    weight: float = Field(..., ge=0.0, le=1.0,
                          description="Intensity of this emotion in this memory")

    model_config = {"json_schema_extra": {
        "example": {"emotion": "playful", "weight": 0.72}
    }}


class PersonaVector(BaseModel):
    """
    A single persona dimension reading.
    Stored in memory_items when memory_type == PERSONA.
    Used for drift detection and preventative steering.
    """
    sycophancy:         float = Field(0.0, ge=0.0, le=1.0)
    aggression:         float = Field(0.0, ge=0.0, le=1.0)
    hallucination_risk: float = Field(0.0, ge=0.0, le=1.0)
    status_sensitivity: float = Field(0.0, ge=0.0, le=1.0)
    openness:           float = Field(0.5, ge=0.0, le=1.0)
    mood_valence:       float = Field(0.5, ge=0.0, le=1.0)  # 0=negative, 1=positive

    @model_validator(mode="after")
    def check_warn_thresholds(self) -> PersonaVector:
        """Flag if any vector is above warn threshold (0.65)."""
        WARN = 0.65
        flags = []
        if self.sycophancy > WARN:
            flags.append(f"sycophancy:{self.sycophancy:.2f}")
        if self.aggression > WARN:
            flags.append(f"aggression:{self.aggression:.2f}")
        if self.status_sensitivity > WARN:
            flags.append(f"status_sensitivity:{self.status_sensitivity:.2f}")
        # Store flags as attribute for caller to read
        object.__setattr__(self, "_warn_flags", flags)
        return self

    @property
    def needs_steering(self) -> bool:
        return self.sycophancy > 0.80 or self.aggression > 0.80

    @property
    def warn_flags(self) -> list[str]:
        return getattr(self, "_warn_flags", [])


class ProactiveHints(BaseModel):
    """
    Suggestions written at deposition time for the action engine.
    Phase 2 (proactive OS actions) reads these.
    """
    music_mood:        Optional[str] = None   # e.g. "lo-fi focus"
    joke_opportunity: bool          = False
    joke_type:        Optional[str] = None   # e.g. "dry-dev-humor"
    vibe_action:      Optional[str] = None   # e.g. "dim-screen-warm"
    nudge_urgency:   float          = Field(0.0, ge=0.0, le=1.0)
    surface_memory_id: Optional[str] = None  # linked memory to surface


class EvoContext(BaseModel):
    """
    Evolutionary psychology layer stamp.
    Attached to memories that carry behavioral strategy signal.
    """
    selection_type:    SelectionType = SelectionType.UNKNOWN
    kin_active:        bool          = False
    tit_for_tat_score: Optional[float] = Field(None, ge=-1.0, le=1.0)
    status_signal:     Optional[str] = None  # "high" | "low" | "threatened"
    hormonal_proxy:    Optional[str] = None  # "cortisol_elevated" | "dopamine_anticipatory"


# ═══════════════════════════════════════════════════════════════════════════
# CORE MEMORY CANDIDATE
# This is what the LLM returns. It gets validated then written to DB.
# ═══════════════════════════════════════════════════════════════════════════

class MemoryCandidate(BaseModel):
    """
    What the LLM proposes after reading a conversation turn.
    The extractor.py module sends this to memory_store.py for validation.

    IMPORTANT:
    - The LLM fills this in JSON
    - Our code validates it with this model
    - Only validated candidates reach SQLite
    - NOP candidates are discarded immediately
    """

    # ── Required by LLM ────────────────────────────────────────────────
    operation:   MemoryOperation
    memory_type: MemoryType
    summary:     str = Field(..., min_length=5, max_length=512,
                             description="Concise, self-contained fact. No pronouns — use 'User'.")
    confidence:  float = Field(..., ge=0.0, le=1.0,
                               description="How certain are you this is durable/accurate?")
    importance:  float = Field(..., ge=0.0, le=1.0,
                               description="How useful will this be for future sessions?")

    # ── Optional enrichment ────────────────────────────────────────────
    raw_content:   Optional[str]             = None   # original text that generated this
    semantic_tags: list[str]                 = Field(default_factory=list)
    emotional_tags: list[EmotionalTag]       = Field(default_factory=list)
    source_type:   SourceType                = SourceType.CHAT
    source_ref:    Optional[str]             = None   # message_id, file_path, etc.

    # ── Relationships ──────────────────────────────────────────────────
    updates_memory_id:     Optional[str] = None   # for UPD: which memory to enrich
    contradicts_memory_id: Optional[str] = None   # for DEL: which memory to obsolete
    linked_entity_ids:     list[str]     = Field(default_factory=list)

    # ── Optional deep layers ───────────────────────────────────────────
    evo_context:     Optional[EvoContext]     = None
    proactive_hints: Optional[ProactiveHints] = None
    persona_vector:  Optional[PersonaVector]  = None   # only for memory_type=PERSONA

    # ── Validators ─────────────────────────────────────────────────────

    @field_validator("summary")
    @classmethod
    def summary_must_be_third_person(cls, v: str) -> str:
        """Memories must be self-contained — reject first-person."""
        first_person = ["i ", "i'm ", "i've ", "my ", "i'd ", "i'll "]
        low = v.lower()
        if any(low.startswith(fp) for fp in first_person):
            raise ValueError(
                "Memory summary must be third-person (start with 'User', not 'I'). "
                f"Got: '{v[:60]}'"
            )
        return v

    @field_validator("semantic_tags")
    @classmethod
    def limit_tags(cls, v: list[str]) -> list[str]:
        if len(v) > 12:
            raise ValueError("Maximum 12 semantic tags per memory.")
        return [t.lower().strip().replace(" ", "-") for t in v]

    @model_validator(mode="after")
    def upd_requires_target(self) -> MemoryCandidate:
        if self.operation == MemoryOperation.UPD and not self.updates_memory_id:
            raise ValueError("UPD operation requires updates_memory_id.")
        if self.operation == MemoryOperation.DEL and not self.contradicts_memory_id:
            raise ValueError("DEL operation requires contradicts_memory_id.")
        return self

    @model_validator(mode="after")
    def nop_must_be_low_importance(self) -> MemoryCandidate:
        if self.operation == MemoryOperation.NOP and self.importance > 0.3:
            raise ValueError(
                "NOP operations should have importance <= 0.3. "
                "If it's important, use ADD or UPD instead."
            )
        return self

    model_config = {"json_schema_extra": {
        "example": {
            "operation": "ADD",
            "memory_type": "preference",
            "summary": "User prefers dry, human-like humor in casual contexts.",
            "raw_content": "I like little dry jokes when things are chill.",
            "semantic_tags": ["humor", "dry-humor", "casual-mode"],
            "emotional_tags": [
                {"emotion": "playful", "weight": 0.72},
                {"emotion": "comfort", "weight": 0.41}
            ],
            "confidence": 0.88,
            "importance": 0.81,
            "source_type": "chat",
        }
    }}


# ═══════════════════════════════════════════════════════════════════════════
# EXTRACTION RESULT (wrapper for LLM response)
# ═══════════════════════════════════════════════════════════════════════════

class MemoryExtractionResult(BaseModel):
    """Wrapper for the LLM extraction response."""
    candidate_memories: list[MemoryCandidate] = Field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════
# STORED MEMORY (what comes back from DB)
# ═══════════════════════════════════════════════════════════════════════════

class StoredMemory(BaseModel):
    """
    Full memory record as stored in SQLite.
    Returned by retrieval.py queries.
    Injected into prompts by prompt_builder.py.
    """
    id:               str
    user_id:          str
    memory_type:      MemoryType
    summary:          str
    raw_content:      Optional[str]
    source_type:      SourceType
    source_ref:       Optional[str]
    confidence:       float
    importance:       float
    status:           MemoryStatus
    semantic_tags:    list[str]             = Field(default_factory=list)
    emotional_tags:   list[EmotionalTag]    = Field(default_factory=list)
    evo_context:      Optional[EvoContext] = None
    proactive_hints:   Optional[ProactiveHints] = None
    persona_vector:   Optional[PersonaVector]  = None
    created_at:       datetime
    updated_at:       datetime
    last_accessed_at: Optional[datetime]   = None

    # Computed at retrieval time
    relevance_score: Optional[float] = None   # recency × salience × emotional match

    def to_prompt_line(self) -> str:
        """
        Compact string representation for injection into LLM system prompt.
        Keeps token cost low.
        """
        tags = ", ".join(self.semantic_tags[:5]) if self.semantic_tags else "—"
        emotions = ", ".join(
            f"{e.emotion}({e.weight:.1f})" for e in self.emotional_tags[:3]
        ) if self.emotional_tags else "—"
        return (
            f"[{self.memory_type.value.upper()}] {self.summary} "
            f"| tags: {tags} | emotion: {emotions} "
            f"| confidence: {self.confidence:.2f}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# MEMORY LINK
# ═══════════════════════════════════════════════════════════════════════════

class MemoryLink(BaseModel):
    """
    Directional relationship between two memory nodes.
    Enables multi-hop reasoning: A → refines → B → caused_by → C
    """
    id:             str = Field(default_factory=lambda: str(uuid4()))
    from_memory_id: str
    to_memory_id:   str
    link_type:      LinkType
    weight:         float = Field(1.0, ge=0.0, le=1.0)
    created_at:     datetime = Field(default_factory=datetime.utcnow)


# ═══════════════════════════════════════════════════════════════════════════
# MEMORY EVENT (audit log)
# ═══════════════════════════════════════════════════════════════════════════

class MemoryEvent(BaseModel):
    """
    Immutable audit trail. Every ADD / UPD / DEL / NOP is logged here.
    Used for trend analysis, debugging, and temporal reasoning.
    """
    id:          str = Field(default_factory=lambda: str(uuid4()))
    memory_id:   Optional[str] = None   # None for NOP events
    event_type:  MemoryOperation
    notes:       Optional[str] = None
    created_at:  datetime = Field(default_factory=datetime.utcnow)


# ═══════════════════════════════════════════════════════════════════════════
# GOAL TRACKER (Humanistic class — progress log)
# ═══════════════════════════════════════════════════════════════════════════

class Goal(BaseModel):
    """
    Long-term aspiration tracked across sessions.
    Written to user-needs.json by the Initializer Agent.
    """
    id:                 str = Field(default_factory=lambda: str(uuid4()))
    user_id:            str
    title:              str
    description:        Optional[str]  = None
    status:             GoalStatus     = GoalStatus.IN_PROGRESS
    progress:           float          = Field(0.0, ge=0.0, le=1.0)
    linked_memory_ids:  list[str]      = Field(default_factory=list)
    created_at:         datetime = Field(default_factory=datetime.utcnow)
    updated_at:         datetime = Field(default_factory=datetime.utcnow)

    def to_log_line(self) -> str:
        bar = "█" * int(self.progress * 10) + "░" * (10 - int(self.progress * 10))
        return f"[{self.status.value:<12}] {bar} {self.progress*100:.0f}%  {self.title}"


# ═══════════════════════════════════════════════════════════════════════════
# ASSISTANT PROFILE
# ═══════════════════════════════════════════════════════════════════════════

class AssistantProfile(BaseModel):
    """
    Stable per-user configuration for the assistant.
    Stored in assistant_profile table. Read on every session start.
    """
    user_id:              str
    humor_mode:           str   = "dry"       # dry | warm | none
    proactive_level:      str   = "medium"    # low | medium | high
    preferred_model:      str   = "llama3.2"  # Ollama model name
    notification_style:   str   = "subtle"    # subtle | bold | none
    memory_rules_version: str   = "1.0"
    steering_enabled:     bool  = True
    sycophancy_threshold: float = Field(0.65, ge=0.0, le=1.0)
    aggression_threshold: float = Field(0.65, ge=0.0, le=1.0)
    created_at:  datetime = Field(default_factory=datetime.utcnow)
    updated_at:  datetime = Field(default_factory=datetime.utcnow)


# ═══════════════════════════════════════════════════════════════════════════
# SESSION CONTEXT (ephemeral, not stored)
# ═══════════════════════════════════════════════════════════════════════════

class SessionContext(BaseModel):
    """
    What the Initializer Agent assembles at session start.
    Injected into the first system prompt. NOT stored in DB.
    """
    user_id:            str
    session_id:         str = Field(default_factory=lambda: str(uuid4()))
    started_at:         datetime = Field(default_factory=datetime.utcnow)
    top_memories:       list[StoredMemory]   = Field(default_factory=list)
    active_goals:       list[Goal]           = Field(default_factory=list)
    persona_snapshot:   Optional[PersonaVector] = None
    steering_needed:    bool = False
    relationship_state: str = "normal"  # normal | stressed | elevated-trust

    def to_system_prompt_block(self) -> str:
        """
        Renders the full context block for injection into the system prompt.
        Keep this tight — every token costs.
        """
        lines = [
            "═══ ASSISTANT MEMORY CONTEXT ═══",
            f"Session: {self.session_id[:8]} | User: {self.user_id}",
            f"Relationship state: {self.relationship_state}",
            "",
        ]

        if self.steering_needed and self.persona_snapshot:
            flags = self.persona_snapshot.warn_flags
            lines += [
                f"⚠ PERSONA DRIFT — {', '.join(flags)}",
                "→ Apply preventative steering. Maintain factual accuracy.",
                "",
            ]

        if self.active_goals:
            lines.append("ACTIVE GOALS:")
            for g in self.active_goals:
                lines.append(f"  {g.to_log_line()}")
            lines.append("")

        if self.top_memories:
            lines.append("RELEVANT MEMORIES:")
            for m in self.top_memories:
                lines.append(f"  • {m.to_prompt_line()}")
            lines.append("")

        lines.append("═══════════════════════════════")
        return "\n".join(lines)
