"""
mac_assistant/app/main.py
─────────────────────────────────────────────────────────────────────────────
REPL entry point for the Mac Assistant with OI memory capsule.

Usage:
  python -m mac_assistant.app.main

Or from project root:
  PYTHONPATH=. python -m mac_assistant.app.main
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import requests

from mac_assistant.app.config import load_user_needs
from mac_assistant.app.extractor import extract_and_resolve
from mac_assistant.app.memory_store import dispatch_candidate, upsert_profile
from mac_assistant.app.prompt_builder import build_chat_prompt
from mac_assistant.app.retrieval import list_all_memories, retrieve_relevant_memories
from mac_assistant.app.schema import get_connection, init_db

OLLAMA_URL = "http://localhost:11434/api/generate"
USER_ID = "default"


def call_ollama(prompt: str, model: str | None = None) -> str:
    model = model or load_user_needs().get("preferred_model", "qwen3:8b")
    response = requests.post(
        OLLAMA_URL,
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=120,
    )
    response.raise_for_status()
    return response.json().get("response", "")


def setup_defaults():
    """Bootstrap assistant profile from user_needs.json if missing."""
    opts = load_user_needs()
    conn = get_connection()
    try:
        profile = {
            "user_id": opts.get("user_id", USER_ID),
            "humor_mode": opts.get("humor_mode", "dry"),
            "proactive_level": opts.get("proactive_level", "medium"),
            "preferred_model": opts.get("preferred_model", "llama3.2"),
            "notification_style": opts.get("notification_style", "subtle"),
            "steering_enabled": opts.get("steering_enabled", True),
        }
        upsert_profile(conn, profile)
    finally:
        conn.close()


def chat_once(user_message: str) -> tuple[str, int]:
    """
    One chat turn: retrieve memories, build prompt, get reply, extract & store.
    Returns (reply_text, num_memories_saved).
    """
    conn = get_connection()

    try:
        memories = retrieve_relevant_memories(USER_ID, user_message, limit=5, conn=conn)
        prompt = build_chat_prompt(user_message, memories, user_id=USER_ID, profile=None)
        reply = call_ollama(prompt)

        valid, _ = extract_and_resolve(user_message, user_id=USER_ID, conn=conn)
        saved = 0
        for candidate in valid:
            if candidate.get("operation") != "NOP":
                mid = dispatch_candidate(conn, USER_ID, candidate)
                if mid:
                    saved += 1

        return reply, saved
    finally:
        conn.close()


def repl():
    init_db()
    setup_defaults()

    print("Mac Assistant (OI Memory Capsule) V1")
    print("Commands: 'memories' | 'quit'")
    print()

    while True:
        try:
            user_message = input("You: ").strip()
        except EOFError:
            break

        if not user_message:
            continue

        if user_message.lower() == "quit":
            break

        if user_message.lower() == "memories":
            conn = get_connection()
            try:
                for mem in list_all_memories(conn, user_id=USER_ID):
                    print(mem)
            finally:
                conn.close()
            print()
            continue

        reply, saved = chat_once(user_message)
        print(f"\nAssistant: {reply}\n")
        print(f"[saved {saved} new memory item(s)]\n")


if __name__ == "__main__":
    repl()
