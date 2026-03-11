from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, model_validator


class MemoryOperation(str, Enum):
    ADD = "ADD"
    UPD = "UPD"
    DEL = "DEL"
    NOP = "NOP"


class MemoryType(str, Enum):
    PREFERENCE = "preference"
    FACT = "fact"
    GOAL = "goal"
    STRATEGY = "strategy"
    EMOTIONAL_STATE = "emotional_state"
    RELATIONSHIP = "relationship"
    SKILL = "skill"
    BELIEF = "belief"
    CONTEXT = "context"
    PERSONA = "persona"


class MemoryStatus(str, Enum):
    ACTIVE = "active"
    OBSOLETE = "obsolete"
    GIST = "gist"
    ARCHIVED = "archived"


class SourceType(str, Enum):
    CHAT = "chat"
    FILE = "file"
    SCREEN = "screen"
    AUDIO = "audio"
    SYSTEM = "system"
    MANUAL = "manual"
    INFERENCE = "inference"


class LinkType(str, Enum):
    REFINES = "refines"
    CONTRADICTS = "contradicts"
    CAUSED_BY = "caused_by"
    SAME_ENTITY = "same_entity"
    SAME_THEME = "same_theme"
    TEMPORAL_SEQ = "temporal_seq"


class SelectionType(str, Enum):
    INDIVIDUAL = "individual"
    KIN = "kin"
    RECIPROCAL = "reciprocal"
    UNKNOWN = "unknown"


class GoalStatus(str, Enum):
    IN_PROGRESS = "IN_PROGRESS"
    PASSING = "PASSING"
    STALLED = "STALLED"
    AT_RISK = "AT_RISK"
    COMPLETED = "COMPLETED"
    ABANDONED = "ABANDONED"


class EmotionalTag(BaseModel):
    emotion: str = Field(..., min_length=1, max_length=64)
    weight: float = Field(..., ge=0.0, le=1.0)

    @field_validator("emotion")
    @classmethod
    def normalize_emotion(cls, v: str) -> str:
        return v.lower().strip().replace(" ", "_")


class PersonaVector(BaseModel):
    sycophancy: float = Field(0.0, ge=0.0, le=1.0)
    aggression: float = Field(0.0, ge=0.0, le=1.0)
    hallucination_risk: float = Field(0.0, ge=0.0, le=1.0)
    status_sensitivity: float = Field(0.0, ge=0.0, le=1.0)
    openness: float = Field(0.5, ge=0.0, le=1.0)
    mood_valence: float = Field(0.5, ge=0.0, le=1.0)

    @property
    def needs_steering(self) -> bool:
        return self.sycophancy > 0.80 or self.aggression > 0.80

    def warn_flags(
        self,
        sycophancy_threshold: float = 0.65,
        aggression_threshold: float = 0.65,
    ) -> list[str]:
        flags: list[str] = []
        if self.sycophancy > sycophancy_threshold:
            flags.append(f"sycophancy:{self.sycophancy:.2f}")
        if self.aggression > aggression_threshold:
            flags.append(f"aggression:{self.aggression:.2f}")
        if self.status_sensitivity > sycophancy_threshold:
            flags.append(f"status_sensitivity:{self.status_sensitivity:.2f}")
        return flags


class ProactiveHints(BaseModel):
    music_mood: str | None = None
    joke_opportunity: bool = False
    joke_type: str | None = None
    vibe_action: str | None = None
    nudge_urgency: float = Field(0.0, ge=0.0, le=1.0)
    surface_memory_id: str | None = None


class EvoContext(BaseModel):
    selection_type: SelectionType = SelectionType.UNKNOWN
    kin_active: bool = False
    tit_for_tat_score: float | None = Field(None, ge=-1.0, le=1.0)
    status_signal: str | None = None
    hormonal_proxy: str | None = None


class MemoryCandidate(BaseModel):
    operation: MemoryOperation
    memory_type: MemoryType
    summary: str = Field(..., min_length=5, max_length=512)
    confidence: float = Field(..., ge=0.0, le=1.0)
    importance: float = Field(..., ge=0.0, le=1.0)

    raw_content: str | None = None
    semantic_tags: list[str] = Field(default_factory=list)
    emotional_tags: list[EmotionalTag] = Field(default_factory=list)
    source_type: SourceType = SourceType.CHAT
    source_ref: str | None = None

    updates_memory_id: str | None = None
    contradicts_memory_id: str | None = None
    linked_entity_ids: list[str] = Field(default_factory=list)

    evo_context: EvoContext | None = None
    proactive_hints: ProactiveHints | None = None
    persona_vector: PersonaVector | None = None

    @field_validator("summary")
    @classmethod
    def summary_must_be_third_person(cls, v: str) -> str:
        low = v.lower().strip()
        first_person = ("i ", "i'm ", "i've ", "my ", "i'd ", "i'll ")
        if any(low.startswith(fp) for fp in first_person):
            raise ValueError("summary must be third-person and start with 'User'.")
        return v.strip()

    @field_validator("semantic_tags")
    @classmethod
    def normalize_tags(cls, v: list[str]) -> list[str]:
        cleaned: list[str] = []
        for tag in v:
            t = tag.lower().strip().replace(" ", "-")
            if t and t not in cleaned:
                cleaned.append(t)
        if len(cleaned) > 12:
            raise ValueError("Maximum 12 semantic tags per memory.")
        return cleaned

    @model_validator(mode="after")
    def validate_operation_contract(self) -> MemoryCandidate:
        if self.operation == MemoryOperation.UPD and not self.updates_memory_id:
            raise ValueError("UPD requires updates_memory_id.")
        if self.operation == MemoryOperation.DEL and not self.contradicts_memory_id:
            raise ValueError("DEL requires contradicts_memory_id.")
        if self.operation == MemoryOperation.NOP and self.importance > 0.3:
            raise ValueError("NOP must have importance <= 0.3.")
        if self.memory_type == MemoryType.PERSONA and not self.persona_vector:
            raise ValueError("memory_type=persona requires persona_vector.")
        return self


class MemoryExtractionResult(BaseModel):
    candidate_memories: list[MemoryCandidate] = Field(default_factory=list)


class StoredMemory(BaseModel):
    id: str
    user_id: str
    memory_type: MemoryType
    summary: str
    raw_content: str | None = None
    source_type: SourceType = SourceType.CHAT
    source_ref: str | None = None
    confidence: float
    importance: float
    status: MemoryStatus
    semantic_tags: list[str] = Field(default_factory=list)
    emotional_tags: list[EmotionalTag] = Field(default_factory=list)
    evo_context: EvoContext | None = None
    proactive_hints: ProactiveHints | None = None
    persona_vector: PersonaVector | None = None
    created_at: datetime
    updated_at: datetime
    last_accessed_at: datetime | None = None
    relevance_score: float | None = None

    def to_prompt_line(self) -> str:
        tags = ", ".join(self.semantic_tags[:5]) if self.semantic_tags else "-"
        emotions = (
            ", ".join(f"{e.emotion}({e.weight:.1f})" for e in self.emotional_tags[:3])
            if self.emotional_tags
            else "-"
        )
        return (
            f"[{self.memory_type.value.upper()}] {self.summary} | tags: {tags} "
            f"| emotions: {emotions} | confidence: {self.confidence:.2f}"
        )


class MemoryLink(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    from_memory_id: str
    to_memory_id: str
    link_type: LinkType
    weight: float = Field(1.0, ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class MemoryEvent(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    memory_id: str | None = None
    event_type: MemoryOperation
    notes: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Goal(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str
    title: str = Field(..., min_length=3, max_length=120)
    description: str | None = None
    status: GoalStatus = GoalStatus.IN_PROGRESS
    progress: float = Field(0.0, ge=0.0, le=1.0)
    linked_memory_ids: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_log_line(self) -> str:
        full = int(self.progress * 10)
        bar = "█" * full + "░" * (10 - full)
        return f"[{self.status.value:<12}] {bar} {self.progress * 100:.0f}% {self.title}"


class AssistantProfile(BaseModel):
    user_id: str
    assistant_name: str = "OI"
    humor_mode: str = "dry"
    proactive_level: str = "medium"
    preferred_model: str = "gpt-oss:20b"
    extraction_model: str = "qwen3:8b"
    notification_style: str = "subtle"
    memory_rules_version: str = "1.0"
    steering_enabled: bool = True
    sycophancy_threshold: float = Field(0.65, ge=0.0, le=1.0)
    aggression_threshold: float = Field(0.65, ge=0.0, le=1.0)
    persona_snapshot_every_n_sessions: int = Field(5, ge=1, le=100)
    max_memories_in_prompt: int = Field(7, ge=1, le=50)
    gist_age_days: int = Field(30, ge=1, le=3650)
    gist_count_limit: int = Field(50, ge=5, le=10000)
    notes: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SessionContext(BaseModel):
    user_id: str
    session_id: str = Field(default_factory=lambda: str(uuid4()))
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    top_memories: list[StoredMemory] = Field(default_factory=list)
    active_goals: list[Goal] = Field(default_factory=list)
    persona_snapshot: PersonaVector | None = None
    steering_needed: bool = False
    relationship_state: str = "normal"

    def to_system_prompt_block(self) -> str:
        lines = [
            "=== ASSISTANT MEMORY CONTEXT ===",
            f"Session: {self.session_id[:8]} | User: {self.user_id}",
            f"Relationship state: {self.relationship_state}",
            "",
        ]
        if self.steering_needed and self.persona_snapshot:
            flags = self.persona_snapshot.warn_flags()
            lines.extend(
                [
                    f"PERSONA DRIFT: {', '.join(flags) if flags else 'detected'}",
                    "Apply preventative steering while preserving factuality.",
                    "",
                ]
            )
        if self.active_goals:
            lines.append("ACTIVE GOALS:")
            lines.extend(f"  {g.to_log_line()}" for g in self.active_goals)
            lines.append("")
        if self.top_memories:
            lines.append("RELEVANT MEMORIES:")
            lines.extend(f"  - {m.to_prompt_line()}" for m in self.top_memories)
            lines.append("")
        lines.append("===============================")
        return "\n".join(lines)


def candidate_from_raw(payload: dict[str, Any]) -> MemoryCandidate:
    """Safe helper used by extraction pipeline."""
    return MemoryCandidate.model_validate(payload)
