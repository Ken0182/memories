from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import requests

from mac_assistant.app.extractor import extract_memory_candidates
from mac_assistant.app.memory_store import dispatch_candidates, upsert_profile
from mac_assistant.app.prompt_builder import build_chat_prompt
from mac_assistant.app.retrieval import search_memories_by_text, top_for_prompt
from mac_assistant.app.schema import DB_PATH, get_connection, init_db

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "gpt-oss:20b"
PROFILE_PATH = Path(__file__).resolve().parent.parent / "config" / "assistant_profile.json"


def call_ollama(prompt: str, model: str = DEFAULT_MODEL, timeout: int = 120) -> str:
    response = requests.post(
        OLLAMA_URL,
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=timeout,
    )
    response.raise_for_status()
    return str(response.json().get("response", "")).strip()


def _load_profile_defaults() -> dict[str, Any]:
    if not PROFILE_PATH.exists():
        return {"user_id": "default"}
    with PROFILE_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if "user_id" not in data:
        data["user_id"] = "default"
    return data


def setup_defaults(user_id: str = "default") -> None:
    profile = _load_profile_defaults()
    profile["user_id"] = user_id
    with get_connection(DB_PATH) as conn:
        upsert_profile(conn, profile)


def chat_once(user_id: str, user_message: str) -> tuple[str, int]:
    with get_connection(DB_PATH) as conn:
        profile = conn.execute(
            "SELECT * FROM assistant_profile WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        profile_dict = dict(profile) if profile else {"user_id": user_id}

        text_hits = search_memories_by_text(conn, user_id, user_message, limit=5)
        memories = text_hits or top_for_prompt(conn, user_id, limit=7)
        prompt = build_chat_prompt(user_message, profile_dict, memories)
        reply = call_ollama(prompt, model=profile_dict.get("preferred_model", DEFAULT_MODEL))

        extraction = extract_memory_candidates(
            user_message,
            existing_memories=memories,
            model=profile_dict.get("extraction_model", DEFAULT_MODEL),
        )
        written = dispatch_candidates(conn, user_id, extraction.candidate_memories)
        return reply, len(written)


def repl(user_id: str = "default") -> None:
    init_db()
    setup_defaults(user_id)
    print("Mac Assistant Memory Capsule")
    print("Type 'memories' to inspect stored memories, or 'quit' to exit.\n")

    while True:
        user_message = input("You: ").strip()
        if not user_message:
            continue
        if user_message.lower() == "quit":
            break
        if user_message.lower() == "memories":
            with get_connection(DB_PATH) as conn:
                rows = conn.execute(
                    """SELECT id, memory_type, summary, status, importance, updated_at
                       FROM memory_items ORDER BY updated_at DESC LIMIT 50"""
                ).fetchall()
                for row in rows:
                    print(dict(row))
                print()
            continue

        try:
            reply, saved = chat_once(user_id, user_message)
            print(f"\nAssistant: {reply}\n")
            print(f"[saved {saved} memory item(s)]\n")
        except requests.RequestException as exc:
            print(f"\nAssistant backend error: {exc}\n")


if __name__ == "__main__":
    repl()
