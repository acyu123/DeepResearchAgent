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
    
    topic: str = field(default=None)  # Research topic obtained from the input message
    clarification_messages: Annotated[list, operator.add] = field(default_factory=list) # List of clarification questions and answers
    needs_clarification: bool = field(default=False) # True if the research topic is unclear
    queries: list = field(default=list) # List of search queries generated based on the research topic and clarification messages
    search_results: list = field(default=list) # The search results for each query, containing the title, url, and raw content
    source: dict = field(default_factory=dict) # An individual search result to send to the summarizer node
    notes: Annotated[list, operator.add] = field(default_factory=list) # summaries of each search result in point form
    needs_followup: bool = field(default=False) # True if the summary notes are insufficient for writing a report on the research topic
    follow_up_question: str = field(default=None) # A follow-up question to fill knowledge gaps with additional research on the topic
    num_followup_attempts: int = field(default=0) # Counts the number of times the agent followed up on insufficient information
    final_report: str = field(default=None) # The final report to be outputted
    research_notes: str = field(default=None) # The research notes used to generate the final report
    