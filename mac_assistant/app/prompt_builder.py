"""
mac_assistant/app/prompt_builder.py
─────────────────────────────────────────────────────────────────────────────
Builds prompts for the Mac Assistant.

  - build_chat_prompt: chat + memories for LLM context
  - build_extraction_prompt: delegates to rules.build_extraction_prompt
  - SessionContext.to_system_prompt_block: full session bootstrap (see models.py)
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from typing import Optional

from mac_assistant.app.rules import build_extraction_prompt as _build_extraction_prompt
from mac_assistant.app.retrieval import get_profile
from mac_assistant.app.schema import get_connection


def build_chat_prompt(user_message: str,
                      memories: list[dict],
                      user_id: str = "default",
                      profile: Optional[dict] = None) -> str:
    """
    Build the chat prompt with assistant profile + relevant memories.
    """
    if profile is None:
        conn = get_connection()
        try:
            profile = get_profile(conn, user_id) or {}
        finally:
            conn.close()

    profile_lines = []
    if profile:
        for key in ("humor_mode", "proactive_level", "preferred_model", "notification_style"):
            if key in profile and profile[key] is not None:
                profile_lines.append(f"- {key}: {profile[key]}")

    memory_lines = []
    for mem in memories:
        mtype = mem.get("memory_type", "fact")
        summary = mem.get("summary", "")
        importance = mem.get("importance", 0.5)
        memory_lines.append(f"- [{mtype}] {summary} (importance={importance:.2f})")

    prompt = f"""
You are a local Mac assistant.
Be practical, useful, and calm.
Use humor only when appropriate.
Be serious for study and important tasks.

Assistant profile:
{chr(10).join(profile_lines) if profile_lines else "- none"}

Relevant long-term memories:
{chr(10).join(memory_lines) if memory_lines else "- none"}

User message:
{user_message}
""".strip()

    return prompt


def build_extraction_prompt(conversation_turn: str,
                            existing_memories: list[dict] | None = None) -> str:
    """Delegates to rules.build_extraction_prompt."""
    return _build_extraction_prompt(conversation_turn, existing_memories)
