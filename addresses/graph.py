from __future__ import annotations

from typing import Literal

from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from addresses.config import openrouter_model_id
from addresses.state import AgentState
from addresses.store import AddressStore
from addresses.tools import create_address_tools

SYSTEM_PROMPT = """You are a postal address book assistant.

## Tools

- `add_address`: save a new address (all fields when possible).
- `list_addresses`: show every saved entry with ids.
- `search_addresses`: find entries by text.
- `format_address`: mailing layout for one id.
- `delete_address`: remove an entry by id.

Be concise. Confirm ids before deleting when ambiguous.
Never invent addresses—always use tools."""


def build_address_agent(
    store: AddressStore | None = None,
    *,
    checkpointer: InMemorySaver | None = None,
) -> tuple[CompiledStateGraph, AddressStore]:
    """
    LangGraph ReAct agent (Graph API) backed by OpenRouter.

    Follows https://docs.langchain.com/oss/python/langgraph/quickstart
    """
    store = store or AddressStore()
    tools = create_address_tools(store)
    tools_by_name = {t.name: t for t in tools}

    # ChatOpenRouter uses timeout in milliseconds (120_000 = 2 min).
    # timeout=120 would mean 120ms and triggers long SDK retries.
    model = init_chat_model(
        openrouter_model_id(),
        temperature=0.2,
        timeout=120_000,
        max_tokens=4096,
        max_retries=1,
    )
    model_with_tools = model.bind_tools(tools)

    def llm_call(state: AgentState) -> dict:
        response = model_with_tools.invoke(
            [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
        )
        return {
            "messages": [response],
            "llm_calls": state.get("llm_calls", 0) + 1,
        }

    def tool_node(state: AgentState) -> dict:
        last = state["messages"][-1]
        results: list[ToolMessage] = []
        for tool_call in last.tool_calls:
            tool = tools_by_name[tool_call["name"]]
            observation = tool.invoke(tool_call["args"])
            results.append(
                ToolMessage(content=str(observation), tool_call_id=tool_call["id"])
            )
        return {"messages": results}

    def should_continue(state: AgentState) -> Literal["tools", END]:
        if state["messages"][-1].tool_calls:
            return "tools"
        return END

    builder = StateGraph(AgentState)
    builder.add_node("agent", llm_call)
    builder.add_node("tools", tool_node)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", should_continue, ["tools", END])
    builder.add_edge("tools", "agent")

    graph = builder.compile(checkpointer=checkpointer or InMemorySaver())
    return graph, store
