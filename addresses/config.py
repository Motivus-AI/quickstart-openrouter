from __future__ import annotations

import os


def openrouter_model_id() -> str:
    return os.environ.get(
        "OPENROUTER_MODEL",
        "openrouter:anthropic/claude-haiku-4.5",
    )


def require_openrouter_api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        msg = "Set OPENROUTER_API_KEY (see .env.example)."
        raise RuntimeError(msg)
    return key
