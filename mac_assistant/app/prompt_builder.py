from __future__ import annotations

from typing import Any

from mac_assistant.app.models import SessionContext
from mac_assistant.app.rules import build_extraction_prompt as build_rules_extraction_prompt


def build_chat_prompt(
    user_message: str,
    profile: dict[str, Any] | None,
    memories: list[dict[str, Any]],
    session_context: SessionContext | None = None,
) -> str:
    profile_lines: list[str] = []
    if profile:
        for key in sorted(profile.keys()):
            if key in {"created_at", "updated_at"}:
                continue
            profile_lines.append(f"- {key}: {profile[key]}")

    memory_lines = [
        (
            f"- [{memory['memory_type']}] {memory['summary']}"
            f" (importance={float(memory.get('importance', 0.0)):.2f},"
            f" relevance={float(memory.get('relevance_score', 0.0)):.2f})"
        )
        for memory in memories
    ]

    parts = [
        "You are a local Mac assistant.",
        "Be practical, useful, calm, and factually accurate.",
        "Use humor only when appropriate to user context.",
        "",
        "Assistant profile:",
        "\n".join(profile_lines) if profile_lines else "- none",
        "",
    ]

    if session_context:
        parts += [session_context.to_system_prompt_block(), ""]

    parts += [
        "Relevant long-term memories:",
        "\n".join(memory_lines) if memory_lines else "- none",
        "",
        "User message:",
        user_message,
    ]
    return "\n".join(parts).strip()


def build_extraction_prompt(
    user_message: str,
    existing_memories: list[dict[str, Any]] | None = None,
) -> str:
    return build_rules_extraction_prompt(user_message, existing_memories=existing_memories)
