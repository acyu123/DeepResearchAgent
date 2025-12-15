"""LangGraph single-node graph template.

Returns a predefined response. Replace logic and configuration as needed.
"""

from __future__ import annotations

from typing import Any, Dict

from langgraph.graph import StateGraph
from langgraph.runtime import Runtime
from langchain_openai import ChatOpenAI

from agent.state import (
    Context,
    InputState,
    State,
)

from agent.utils import (
    get_llm_api_key,
    get_search_api_key,
)


async def call_model(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Process input and returns output.

    Can use runtime context to alter behavior.
    """
    print('---- call model')
    print(get_llm_api_key())
    print(state)
    
    llm = ChatOpenAI(
        model="gpt-4.1-mini",
        # stream_usage=True,
        # temperature=None,
        # max_tokens=None,
        # timeout=None,
        # reasoning_effort="low",
        # max_retries=2,
        api_key=get_llm_api_key(),
        # base_url="...",
        # organization="...",
        # other params...
    )
    
    ai_msg = llm.invoke(state['messages'])
    print(ai_msg)
    
    return {
        "messages": [ai_msg],
        "changeme": ai_msg.text,
    }


# Define the graph
graph = (
    StateGraph(State, input=InputState, context_schema=Context)
    .add_node(call_model)
    .add_edge("__start__", "call_model")
    .compile(name="Deep Research Agent")
)
