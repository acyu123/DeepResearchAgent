from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing_extensions import Annotated

from langgraph.graph import MessagesState
from typing_extensions import TypedDict


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
    
    topic: str = field(default=None)  # Research topic
    clarification_messages: Annotated[list, operator.add] = field(default_factory=list)
    needs_clarification: bool = field(default=False)
    queries: list = field(default=list)
    search_results: list = field(default=list)
    source: dict = field(default_factory=dict)
    notes: Annotated[list, operator.add] = field(default_factory=list)
    needs_followup: bool = field(default=False)
    follow_up_question: str = field(default=None)
    num_followup_attempts: int = field(default=0)
    