"""LangGraph single-node graph template.

Returns a predefined response. Replace logic and configuration as needed.
"""

from __future__ import annotations

from typing import Any, Dict
import json

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from agent.state import (
    InputState,
    State,
)

from agent.utils import (
    call_llm,
    get_search_api_key,
    get_current_date,
    format_clarification_messages,
    get_search_results,
)

from agent.prompts import (
    clarification_prompt,
    query_generation_prompt,
)

from agent.config import (
    Configuration,
)


async def clarification(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """Process input and returns output.

    Can use runtime context to alter behavior.
    """
    print('---- clarification')
    print(state)
    
    configurable = Configuration.from_runnable_config(config)
    
    new_clarification_messages = []
    
    if 'topic' not in state or not state['topic']:
        topic = state['messages'][0].text
    else:
        new_clarification_messages.append(state['messages'][-1])
        topic = state['topic']
    
    
    if len(state['clarification_messages'])+1 >= configurable.max_clarification_retries * 2:
        return {
            'clarification_messages': new_clarification_messages,
            "topic": topic, 
            'needs_clarification': False,
        }
    
    clarification_messages = format_clarification_messages(state['clarification_messages'] + new_clarification_messages)
    
    prompt = clarification_prompt.format(
        date=get_current_date(), 
        user_prompt=topic,
        messages=clarification_messages
    )
    
    print(prompt)
    
    
    response = call_llm(prompt, return_json=True)
    print(response)
    
    
    if response['needs_clarification']:
        # End with clarifying question for user
        print('needs_clarification')
        new_clarification_messages.append(AIMessage(content=response['clarification_question']))
        return {
            'messages': [AIMessage(content=response['clarification_question'])],
            'clarification_messages': new_clarification_messages,
            "topic": topic, 
            'needs_clarification': True,
        }

    return {
        "topic": topic, 
        'needs_clarification': False
    }

def route_clarification(
    state: State, config: RunnableConfig
):
    print(state)
    return 'needs_clarification' in state and state['needs_clarification']
    

async def query_generation(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """Process input and returns output.

    Can use runtime context to alter behavior.
    """
    print('---- query_generation')
    print(state)
    
    configurable = Configuration.from_runnable_config(config)
    print(configurable)
    
    prompt = query_generation_prompt.format(
        num_queries=configurable.num_queries,
        date=get_current_date(), 
        user_prompt=state['topic'],
        messages=format_clarification_messages(state['clarification_messages'])
    )
    
    print(prompt)
    
    response = call_llm(prompt, return_json=True)
    print(response)
    
    return {
        "queries": response['search_queries'],
    }

async def search_results_extraction(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """Process input and returns output.

    Can use runtime context to alter behavior.
    """
    print('---- search_results_extraction')
    print(state)
    
    configurable = Configuration.from_runnable_config(config)
    
    for query in state['queries']:
        get_search_results(query, configurable.num_results_per_query)
    
    return {
        "messages": [ai_msg],
        "changeme": ai_msg.text,
    }

async def summarize(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """Process input and returns output.

    Can use runtime context to alter behavior.
    """
    print('---- search_results_extraction')
    print(state)
    
    return {
        "messages": [ai_msg],
        "changeme": ai_msg.text,
    }

async def final_report(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """Process input and returns output.

    Can use runtime context to alter behavior.
    """
    print('---- final_report')
    print(state)

    return {
        "messages": [ai_msg],
        "changeme": ai_msg.text,
    }

async def followup(state: State, config: RunnableConfig) -> Command[Literal["query_generation", "__end__"]]:
    """Process input and returns output.

    Can use runtime context to alter behavior.
    """
    print('---- followup')
    print(state)
    
    
    response = call_llm(prompt)
    
    if response.should_followup:
        # End with clarifying question for user
        return Command(
            goto="query_generation", 
            update={"messages": [AIMessage(content=response.question)], "topic": topic}
        )
    else:
        # Proceed to research with verification message
        return Command(
            goto=END, 
            update={"messages": []}
        )



# Define the graph
graph = (
    StateGraph(State, input=InputState, config_schema=Configuration)
    .add_node(clarification)
    .add_node(query_generation)
    .add_node(search_results_extraction)
    .add_node(summarize)
    .add_node(final_report)
    .add_node(followup)
    .add_edge("__start__", "clarification")
    .add_conditional_edges("clarification", route_clarification, {True: END, False: "query_generation"})
    .add_edge("query_generation", "search_results_extraction")
    .add_edge("search_results_extraction", "summarize")
    .add_edge("summarize", "final_report")
    .add_edge("final_report", "followup")
    .add_edge("followup", END)
    .compile(name="Deep Research Agent")
)
