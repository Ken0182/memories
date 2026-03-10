# Mac Assistant — OI Memory Capsule

A Memory capsule for MacBook-implemented AI, following the OI (Open Interface) architecture.

## Features

- **ADD / UPD / DEL / NOP** — Full extraction loop with temporal consistency
- **Memory types** — preference, fact, goal, strategy, emotional_state, relationship, skill, belief, context, persona
- **Emotional + semantic tags** — First-class PAD-style tagging
- **Memory links** — Typed graph (refines, contradicts, caused_by, same_entity, same_theme, temporal_seq)
- **Persona drift** — Sycophancy, aggression, status_sensitivity tracking
- **Goal tracking** — Humanistic class with progress logs
- **Gisting** — Compress old memories to save token budget
- **Multi-hop retrieval** — Follow link chains from seed memories

## Installation

```bash
pip install -r requirements.txt
```

Requires Ollama running locally with a model (e.g. `qwen3:8b`, `llama3.2`).

## Usage

```bash
# From workspace root
python -m mac_assistant.app.main
```

Or bootstrap the database only:

```bash
python -m mac_assistant.app.schema
```

## Layout

```
mac_assistant/
├── app/
│   ├── models.py       # Pydantic v2 models
│   ├── schema.py       # SQLite DDL
│   ├── memory_store.py # All writes (ADD/UPD/DEL/NOP)
│   ├── retrieval.py    # All reads
│   ├── rules.py        # Extraction policy, validation
│   ├── prompt_builder.py
│   ├── extractor.py    # LLM extraction
│   └── main.py        # REPL entry point
├── data/
│   ├── assistant.db   # SQLite DB (created on first run)
│   ├── user_needs.json
│   └── memory_rules.json
└── requirements.txt
```

## Configuration

- **user_needs.json** — Assistant profile, thresholds, model names
- **memory_rules.json** — Extraction policy, do_not_store, always_store, temporal_consistency

## Design Principles

1. **Model PROPOSES → code VALIDATES → code STORES**
2. **Recency wins on contradiction**
3. **Never hard-delete — archive obsolete**
4. **Persona vectors tracked separately from episodic memory**
