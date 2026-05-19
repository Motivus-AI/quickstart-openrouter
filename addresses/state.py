from __future__ import annotations

import operator
from typing import Annotated

from langchain.messages import AnyMessage
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """LangGraph state: messages accumulate across the ReAct loop."""

    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int
