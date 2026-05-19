"""
Minimal LangGraph Graph API agent (calculator), OpenRouter model.
https://docs.langchain.com/oss/python/langgraph/quickstart
"""

from __future__ import annotations

import operator
import os
from typing import Annotated, Literal

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain.tools import tool
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

load_dotenv()


@tool
def add(a: int, b: int) -> int:
    """Add a and b."""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiply a and b."""
    return a * b


class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int


def main() -> None:
    model = init_chat_model(
        os.environ.get("OPENROUTER_MODEL", "openrouter:anthropic/claude-haiku-4.5"),
        temperature=0,
        timeout=120_000,
        max_retries=1,
    )
    tools = [add, multiply]
    by_name = {t.name: t for t in tools}
    bound = model.bind_tools(tools)

    def llm_call(state: MessagesState) -> dict:
        return {
            "messages": [
                bound.invoke(
                    [
                        SystemMessage(
                            content="You are a helpful calculator assistant."
                        )
                    ]
                    + state["messages"]
                )
            ],
            "llm_calls": state.get("llm_calls", 0) + 1,
        }

    def tool_node(state: MessagesState) -> dict:
        out: list[ToolMessage] = []
        for tc in state["messages"][-1].tool_calls:
            out.append(
                ToolMessage(
                    content=str(by_name[tc["name"]].invoke(tc["args"])),
                    tool_call_id=tc["id"],
                )
            )
        return {"messages": out}

    def should_continue(state: MessagesState) -> Literal["tools", "__end__"]:
        return "tools" if state["messages"][-1].tool_calls else END

    builder = StateGraph(MessagesState)
    builder.add_node("agent", llm_call)
    builder.add_node("tools", tool_node)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", should_continue, ["tools", END])
    builder.add_edge("tools", "agent")
    graph = builder.compile()

    result = graph.invoke(
        {"messages": [HumanMessage(content="What is 3 plus 4, then times 2?")]}
    )
    for m in result["messages"]:
        m.pretty_print()


if __name__ == "__main__":
    main()
