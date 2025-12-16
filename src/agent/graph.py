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
from langgraph.types import Command, Send, Overwrite

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
    format_research_notes,
)

from agent.prompts import (
    clarification_prompt,
    query_generation_prompt,
    notes_prompt,
    followup_prompt,
    report_generation_prompt,
)

from agent.config import (
    Configuration,
)


async def clarification(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """Process input and returns output.

    Can use runtime context to alter behavior.
    """
    print('---- clarification')
    
    configurable = Configuration.from_runnable_config(config)
    
    if 'needs_clarification' not in state or not state['needs_clarification']:
        if 'topic' in state and 'final_report' in state:
            return {
                "follow_up_question": state['messages'][-1].text,
                "needs_followup": True,
            }
    
    new_clarification_messages = []
    
    if 'topic' not in state or not state['topic']:
        topic = state['messages'][0].text
    else:
        new_clarification_messages.append(state['messages'][-1])
        topic = state['topic']
    
    
    if len(state['clarification_messages'])+1 >= configurable.max_clarification_retries * 2:
        if new_clarification_messages:
            return {
                'clarification_messages': new_clarification_messages,
                "topic": topic, 
                'needs_clarification': False,
            }
        else:
            return {
                "topic": topic, 
                'needs_clarification': False,
            }
    
    clarification_messages = format_clarification_messages(state['clarification_messages'] + new_clarification_messages)
    
    prompt = clarification_prompt.format(
        date=get_current_date(), 
        user_prompt=topic,
        messages=clarification_messages
    )
    
    response = call_llm(prompt, return_json=True)
    
    
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
    return 'needs_clarification' in state and state['needs_clarification']
    

async def query_generation(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """Process input and returns output.

    Can use runtime context to alter behavior.
    """
    print('---- query_generation')
    
    configurable = Configuration.from_runnable_config(config)
    
    topic = state['topic']
    if 'needs_followup' in state and state['needs_followup']:
        topic = state['follow_up_question']
    
    prompt = query_generation_prompt.format(
        num_queries=configurable.num_queries,
        date=get_current_date(),
        user_prompt=topic,
        messages=format_clarification_messages(state['clarification_messages'])
    )
    
    
    response = call_llm(prompt, return_json=True)
    
    queries = response['search_queries']
    
    return {
        "queries": queries[:configurable.num_queries],
        "needs_followup": False,
    }

async def search_results_extraction(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """Process input and returns output.

    Can use runtime context to alter behavior.
    """
    print('---- search_results_extraction')
    
    configurable = Configuration.from_runnable_config(config)
    
    search_results = []
    for query in state['queries']:
        search_results.extend(get_search_results(query, configurable.num_results_per_query))
    
    return {
        "search_results": search_results,
    }

# Conditional edge function to create summarize workers that each summarize a search result
def assign_workers(state: State):
    """Assign a worker to each search result"""

    # Kick off section writing in parallel via Send() API
    return [Send("summarize", {
        "source": s, 
        "topic": state['topic'],
    }) for s in state["search_results"]]

async def summarize(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """Process input and returns output.

    Can use runtime context to alter behavior.
    """
    print('---- search_results_extraction')
    
    source = state['source']
    
    prompt = notes_prompt.format(
        user_prompt=state['topic'],
        search_result=source['content']
    )
    
    response = call_llm(prompt, return_json=False)
    
    return {
        "notes": [{'title': source['title'], 'url': source['url'], 'notes': response.text}],
    }


async def followup(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """Process input and returns output.

    Can use runtime context to alter behavior.
    """
    print('---- followup')
    
    notes = format_research_notes(state['notes'])
    
    prompt = followup_prompt.format(
        user_prompt=state['topic'],
        summary=notes,
    )
    
    response = call_llm(prompt, return_json=True)
    
    
    num_followup = 0
    if 'num_followup_attempts' in state:
        num_followup = state['num_followup_attempts']
    if response['needs_followup']:
        num_followup += 1
    
    return {
        "needs_followup": response['needs_followup'],
        "follow_up_question": response['follow_up_question'],
        "num_followup_attempts": num_followup,
    }
    
def route_followup(
    state: State, config: RunnableConfig
):
    configurable = Configuration.from_runnable_config(config)
    
    if configurable.max_followup_retries == 0:
        return False
    if 'num_followup_attempts' in state and state['num_followup_attempts'] > configurable.max_followup_retries:
        return False
        
    return 'needs_followup' in state and state['needs_followup']
    
    
async def final_report(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """Process input and returns output.

    Can use runtime context to alter behavior.
    """
    print('---- final_report')
    
    notes = format_research_notes(state['notes'])
    
    prompt = report_generation_prompt.format(
        user_prompt=state['topic'],
        research_notes=notes,
    )
    
    response = call_llm(prompt, return_json=False)
    

    return {
        "messages": [response.text],
        "final_report": response.text,
    }





# Define the graph
graph = (
    StateGraph(State, input=InputState, config_schema=Configuration)
    .add_node(clarification)
    .add_node(query_generation)
    .add_node(search_results_extraction)
    .add_node(summarize)
    .add_node(followup)
    .add_node(final_report)
    
    .add_edge("__start__", "clarification")
    .add_conditional_edges("clarification", route_clarification, {True: END, False: "query_generation"})
    .add_edge("query_generation", "search_results_extraction")
    .add_conditional_edges("search_results_extraction", assign_workers, ["summarize"])
    .add_edge("summarize", "followup")
    .add_conditional_edges("followup", route_followup, {True: "query_generation", False: "final_report"})
    .add_edge("final_report", END)
    
    .compile(name="Deep Research Agent")
)
