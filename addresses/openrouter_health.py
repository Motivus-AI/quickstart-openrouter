from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from addresses.config import openrouter_model_id, require_openrouter_api_key

CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
MODELS_URL = "https://openrouter.ai/api/v1/models"
MODELS_USER_URL = "https://openrouter.ai/api/v1/models/user"
PRIVACY_URL = "https://openrouter.ai/settings/privacy"


def _raw_model_slug() -> str:
    model = openrouter_model_id()
    if model.startswith("openrouter:"):
        return model.removeprefix("openrouter:")
    return model


def _request_headers(api_key: str) -> dict[str, str]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if site_url := os.environ.get("OPENROUTER_SITE_URL"):
        headers["HTTP-Referer"] = site_url
    if site_name := os.environ.get("OPENROUTER_SITE_NAME"):
        headers["X-OpenRouter-Title"] = site_name
    return headers


def _fetch_json(url: str, api_key: str) -> dict:
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {api_key}"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


def _cheapest_anthropic_for_account(api_key: str) -> str | None:
    """Best-effort: cheapest Anthropic slug allowed by account + ZDR filters."""
    try:
        catalog = {
            m["id"]: m for m in _fetch_json(MODELS_URL, api_key).get("data", [])
        }
        allowed = _fetch_json(MODELS_USER_URL, api_key).get("data", [])
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError):
        return None

    priced: list[tuple[float, str]] = []
    for entry in allowed:
        mid = entry.get("id", "")
        if not mid.startswith("anthropic/"):
            continue
        meta = catalog.get(mid)
        if not meta:
            continue
        try:
            prompt = float(meta["pricing"]["prompt"])
        except (KeyError, TypeError, ValueError):
            continue
        priced.append((prompt, mid))

    if not priced:
        return None
    priced.sort(key=lambda x: x[0])
    return priced[0][1]


def check_openrouter_connectivity() -> tuple[bool, str]:
    """
    Probe chat/completions with the configured model.

    Returns (ok, human-readable message).
    """
    api_key = require_openrouter_api_key()
    model = _raw_model_slug()
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with exactly: ok"}],
        "max_tokens": 16,
    }
    req = urllib.request.Request(
        CHAT_URL,
        data=json.dumps(payload).encode(),
        headers=_request_headers(api_key),
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.load(resp)
        text = data["choices"][0]["message"]["content"]
        return True, f"OpenRouter OK (model={model!r}, reply={text!r})"
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            err = json.loads(body).get("error", {})
            message = err.get("message", body)
        except json.JSONDecodeError:
            message = body
        return False, _format_failure(api_key, model, exc.code, message)
    except urllib.error.URLError as exc:
        return False, f"Network error talking to OpenRouter: {exc.reason}"


def _format_failure(api_key: str, model: str, status: int, message: str) -> str:
    if "guardrail" in message.lower() or "data policy" in message.lower():
        hint = _cheapest_anthropic_for_account(api_key)
        suggest = (
            f"\n\nWith your current ZDR settings, try:\n"
            f"  OPENROUTER_MODEL=openrouter:{hint}\n"
            if hint
            else ""
        )
        return (
            f"OpenRouter rejected model {model!r} (HTTP {status}): {message}\n\n"
            "Your organization requires Zero Data Retention (ZDR). Older/cheaper "
            "models like claude-3-haiku often have no ZDR route; they will always 404.\n"
            f"1. Open {PRIVACY_URL} (or switch to a ZDR-compatible model){suggest}"
            "2. Re-run: uv run python main.py --check\n\n"
            "See: https://openrouter.ai/docs/guides/features/zdr"
        )
    if status == 404 and "No endpoints found" in message:
        return (
            f"Model {model!r} is not available on your account (HTTP 404).\n"
            f"Pick another slug in OPENROUTER_MODEL or browse "
            "https://openrouter.ai/models\n\n"
            f"API message: {message}"
        )
    return f"OpenRouter error for model {model!r} (HTTP {status}): {message}"
