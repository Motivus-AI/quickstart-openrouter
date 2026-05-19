#!/usr/bin/env python3
"""CLI for the address-book LangGraph agent (OpenRouter)."""

from __future__ import annotations

import argparse
import sys
import uuid

from dotenv import load_dotenv
from langchain.messages import HumanMessage

from addresses.config import require_openrouter_api_key
from addresses.graph import build_address_agent
from addresses.openrouter_health import check_openrouter_connectivity


def print_last_message(result: dict) -> None:
    msg = result["messages"][-1]
    content = getattr(msg, "content", None) or msg
    print(content)


def _handle_openrouter_error(exc: BaseException) -> int:
    text = str(exc)
    if "ReadTimeout" in type(exc).__name__ or "read operation timed out" in text.lower():
        print(
            "OpenRouter request timed out (network or provider slow).\n"
            "Try again, or increase OPENROUTER_TIMEOUT_MS (default 180000).\n"
            "Quick check: uv run python main.py --check\n",
            file=sys.stderr,
        )
        print(f"Details: {text}", file=sys.stderr)
        return 3
    if "guardrail" in text.lower() or "data policy" in text.lower():
        print(
            "OpenRouter blocked the request due to account privacy / guardrail settings.\n"
            "Fix: https://openrouter.ai/settings/privacy\n"
            "Then run: uv run python main.py --check\n",
            file=sys.stderr,
        )
        print(f"Details: {text}", file=sys.stderr)
        return 2
    print(f"OpenRouter API error: {text}", file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Address book agent (LangGraph + OpenRouter)",
    )
    parser.add_argument("message", nargs="*", help="User message for the agent")
    parser.add_argument(
        "--thread",
        default="addresses-cli",
        help="Conversation thread id (checkpoint memory)",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run a scripted demo flow",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify OPENROUTER_API_KEY and model access before running the agent",
    )
    args = parser.parse_args(argv)

    try:
        require_openrouter_api_key()
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1

    if args.check:
        ok, msg = check_openrouter_connectivity()
        print(msg)
        return 0 if ok else 2

    if not args.demo and not args.message:
        parser.print_help()
        return 0

    thread_id = f"demo-{uuid.uuid4().hex[:8]}" if args.demo else args.thread
    graph, _store = build_address_agent()
    config = {"configurable": {"thread_id": thread_id}}

    prompts = (
        [
            "Save my home address: label Home, street 742 Evergreen Terrace, "
            "city Springfield, state IL, postal_code 62704, country US.",
            "Add office: label Office, street 1 Market St, number 1, "
            "city San Francisco, state CA, postal_code 94105.",
            "List all my addresses.",
            "What is the mailing format for Home?",
        ]
        if args.demo
        else [" ".join(args.message)]
    )

    try:
        for i, content in enumerate(prompts, start=1):
            print(f"\n>>> [{i}/{len(prompts)}] {content}\n", flush=True)
            print("… calling OpenRouter (agent + tools)", flush=True)
            result = graph.invoke(
                {"messages": [HumanMessage(content=content)]},
                config=config,
            )
            print_last_message(result)
            sys.stdout.flush()
    except Exception as exc:
        name = type(exc).__name__
        if "NotFound" in name or "OpenRouter" in name or "Timeout" in name:
            return _handle_openrouter_error(exc)
        raise

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
