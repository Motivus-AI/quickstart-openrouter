"""
Minimal OpenRouter HTTP API call (no LangGraph).
https://openrouter.ai/docs/quickstart
"""

import json
import os
import urllib.request

API_URL = "https://openrouter.ai/api/v1/chat/completions"


def main() -> None:
    api_key = os.environ["OPENROUTER_API_KEY"]
    payload = {
        "model": os.environ.get("OPENROUTER_RAW_MODEL", "anthropic/claude-sonnet-4"),
        "messages": [
            {
                "role": "user",
                "content": "In one sentence, what is a postal address book?",
            }
        ],
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if site_url := os.environ.get("OPENROUTER_SITE_URL"):
        headers["HTTP-Referer"] = site_url
    if site_name := os.environ.get("OPENROUTER_SITE_NAME"):
        headers["X-OpenRouter-Title"] = site_name

    req = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode(),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.load(resp)

    print(data["choices"][0]["message"]["content"])


if __name__ == "__main__":
    main()
