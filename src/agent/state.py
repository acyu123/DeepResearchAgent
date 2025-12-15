from __future__ import annotations

from dataclasses import dataclass

from langgraph.graph import MessagesState
from typing_extensions import TypedDict


class Context(TypedDict):
    """Context parameters for the agent.

    Set these when creating assistants OR when invoking the graph.
    See: https://langchain-ai.github.io/langgraph/cloud/how-tos/configuration_cloud/
    """

    my_configurable_param: str

@dataclass
class InputState(MessagesState):
    """Input state for the agent containing 'messages'.

    Defines the initial structure of incoming data.
    See: https://langchain-ai.github.io/langgraph/concepts/low_level/#state
    """

@dataclass
class State(MessagesState):
    """Main state containing messages and research data.

    Defines the initial structure of incoming data.
    See: https://langchain-ai.github.io/langgraph/concepts/low_level/#state
    """

    changeme: str = "example"
    