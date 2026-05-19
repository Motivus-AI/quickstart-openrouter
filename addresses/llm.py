from __future__ import annotations

import os

import httpx
import openrouter
from langchain_openrouter import ChatOpenRouter

from addresses.config import openrouter_model_id, require_openrouter_api_key

# ChatOpenRouter `timeout` is in milliseconds.
DEFAULT_TIMEOUT_MS = 180_000


def _raw_model_slug() -> str:
    model = openrouter_model_id()
    return model.removeprefix("openrouter:") if model.startswith("openrouter:") else model


def _attribution_headers() -> dict[str, str]:
    headers: dict[str, str] = {}
    if site_url := os.environ.get("OPENROUTER_SITE_URL"):
        headers["HTTP-Referer"] = site_url
    if site_name := os.environ.get("OPENROUTER_SITE_NAME"):
        headers["X-OpenRouter-Title"] = site_name
    return headers


def create_chat_model() -> ChatOpenRouter:
    """
    OpenRouter chat model with a correctly configured HTTP client.

    langchain-openrouter builds httpx clients without read timeout when
    attribution headers are set; that causes ReadTimeout on slow routes.
    """
    timeout_ms = int(os.environ.get("OPENROUTER_TIMEOUT_MS", DEFAULT_TIMEOUT_MS))
    timeout_sec = timeout_ms / 1000
    http_timeout = httpx.Timeout(timeout_sec, connect=30.0)
    headers = _attribution_headers()

    http_client = httpx.Client(
        timeout=http_timeout,
        headers=headers or None,
        follow_redirects=True,
    )
    async_client = httpx.AsyncClient(
        timeout=http_timeout,
        headers=headers or None,
        follow_redirects=True,
    )
    sdk = openrouter.OpenRouter(
        api_key=require_openrouter_api_key(),
        client=http_client,
        async_client=async_client,
        timeout_ms=timeout_ms,
    )

    return ChatOpenRouter(
        model=_raw_model_slug(),
        temperature=0.2,
        max_tokens=int(os.environ.get("OPENROUTER_MAX_TOKENS", "1024")),
        max_retries=int(os.environ.get("OPENROUTER_MAX_RETRIES", "0")),
        client=sdk,
        # Prevent _build_client() from replacing our httpx client.
        app_url=None,
        app_title=None,
    )
