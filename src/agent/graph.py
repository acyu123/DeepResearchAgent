from __future__ import annotations

from typing import Any, Dict, List
import json
from pydantic import BaseModel, Field

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


# The structured output from the LLM in the clarification node
class ClarificationOutput(BaseModel):
    needs_clarification: bool = Field(description="True if the user request is unclear. False if the user request is clear")
    clarification_question: str = Field(description="A question that would most improve the quality of the research")

# The structured output from the LLM in the query_generation node
class QueryGenerationOutput(BaseModel):
    search_queries: List[str] = Field(description="A list of strings, where each string is a search query")

# The structured output from the LLM in the followup node
class FollowupOutput(BaseModel):
    needs_followup: bool = Field(description="True if the information is insufficient. False if the information is enough for writing a report about the topic")
    follow_up_question: str = Field(description="A question that would most improve the quality of the research")


async def clarification(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """
    LangGraph node that analyzes the research topic and clarification messages to determine if the research topic is clear enough. 
    Otherwise ask clarification questions until it's clear.
    
    Args:
        state: Current agent state containing user messages
        config: Runtime configuration with model settings and preferences
        
    Returns:
        Dictionary with the state updates:
            topic: The research topic obtained from the messages
            needs_clarification: Determines whether to end with a clarifying question or proceed to generate search queries
            clarification_messages: The updated list of clarification questions and answers obtained from the messages
    """
    print('---- clarification')
    
    configurable = Configuration.from_runnable_config(config)
    
    new_clarification_messages = [] 
    
    if 'topic' not in state or not state['topic']:
        topic = state['messages'][-1].text # If the topic isn't set, the latest message is the research topic
    else:
        new_clarification_messages.append(state['messages'][-1]) # the latest message is the answer to the previous clarification question
        topic = state['topic']
    
    # check if the number of clarification attempts exceeds the maximum number of retries
    if len(state['clarification_messages'])+1 >= configurable.max_clarification_retries * 2:
        # indicate that no more clarification is needed
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
    
    # convert the list of clarification messages into a single string
    clarification_messages = format_clarification_messages(state['clarification_messages'] + new_clarification_messages)
    
    # format the LLM prompt with the required information
    prompt = clarification_prompt.format(
        date=get_current_date(), 
        user_prompt=topic,
        messages=clarification_messages
    )
    
    # call the LLM to obtain the structured output
    response = call_llm(configurable.model, prompt, structure=ClarificationOutput)
    
    if response.needs_clarification:
        # append the clarification question to the messages, and the list of clarification messages
        new_clarification_messages.append(AIMessage(content=response.clarification_question))
        # indicate that clarification is needed
        return {
            'messages': [AIMessage(content=response.clarification_question)],
            'clarification_messages': new_clarification_messages,
            "topic": topic, 
            'needs_clarification': True,
        }

    # indicate that no clarification is needed
    return {
        "topic": topic, 
        'needs_clarification': False
    }

def route_clarification(
    state: State, config: RunnableConfig
):
    """
    Determines whether to end with a clarifying question or proceed to generate search queries
    Returns True if clarification is needed
    """
    return 'needs_clarification' in state and state['needs_clarification']
    

async def query_generation(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """
    LangGraph node that generates a list of search queries based on the research topic or the follow-up question
    
    Uses an LLM to create optimized search queries for web research based on the research topic. 
    
    Args:
        state: Current agent state containing user messages
        config: Runtime configuration with model settings and preferences
        
    Returns:
        Dictionary with the state updates:
            queries: List of generated search queries
            needs_followup: Reset to False after follow-up is processed
    """
    print('---- query_generation')
    
    configurable = Configuration.from_runnable_config(config)
    
    
    topic = state['topic']
    
    # set the follow-up question as the query generation topic if follow-up is needed
    if 'needs_followup' in state and state['needs_followup']:
        topic = state['follow_up_question']
    
    # format the LLM prompt with the required information
    prompt = query_generation_prompt.format(
        num_queries=configurable.num_queries,
        date=get_current_date(),
        user_prompt=topic,
        messages=format_clarification_messages(state['clarification_messages'])
    )
    
    # call the LLM to obtain the structured output
    response = call_llm(configurable.model, prompt, structure=QueryGenerationOutput)
    
    queries = response.search_queries
    
    return {
        "queries": queries[:configurable.num_queries],
        "needs_followup": False,
        "follow_up_question": "",
    }

async def search_results_extraction(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """
    LangGraph node that calls the search API with the search queries to obtain a list of raw content from each source
    
    Args:
        state: Current agent state containing user messages
        config: Runtime configuration with model settings and preferences
        
    Returns:
        Dictionary with the state updates:
            search_results: List of search results aggregated from all of the queries, containing the title, url, and raw content
    """
    print('---- search_results_extraction')
    
    configurable = Configuration.from_runnable_config(config)
    
    # for each search query, add the search results to the list of all the search results
    search_results = []
    for query in state['queries']:
        search_results.extend(get_search_results(query, configurable.num_results_per_query))
    
    return {
        "search_results": search_results,
    }

def assign_workers(state: State):
    """
    Assign a summarize node to each search result
    """

    # run the summarize nodes in parallel for each search result
    return [Send("summarize", {
        "source": s, 
        "topic": state['topic'],
    }) for s in state["search_results"]]

async def summarize(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """
    LangGraph node that converts the raw content of a search result into a summary in point-form.
    
    Args:
        state: Current agent state containing user messages
        config: Runtime configuration with model settings and preferences
        
    Returns:
        Dictionary with the state updates:
            notes: The summary for the current search result in point-form, along with the corresponding title and url
    """
    print('---- summarize')
    
    configurable = Configuration.from_runnable_config(config)
    
    source = state['source']
    
    # format the LLM prompt with the required information
    prompt = notes_prompt.format(
        search_result=source['content']
    )
    
    # call the LLM to obtain a summary in point-form from the raw content of the search result
    response = call_llm(configurable.model, prompt)
    
    return {
        "notes": [{'title': source['title'], 'url': source['url'], 'notes': response.text}],
    }


async def followup(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """
    LangGraph node that determines if the summarized search results contain enough information to write a report on the research topic.
    If there's not enough information, return a follow-up question
    
    Args:
        state: Current agent state containing user messages
        config: Runtime configuration with model settings and preferences
        
    Returns:
        Dictionary with the state updates:
            needs_followup: Determines whether to restart the research process with a follow-up question or proceed to generate the final report
            follow_up_question: A single question to fill knowledge gaps in the current information
            num_followup_attempts: Counts the number of follow-up attempts
    """
    print('---- followup')
    
    configurable = Configuration.from_runnable_config(config)
    
    # check if the number of follow-up attempts exceeds the maximum number of retries
    if configurable.max_followup_retries == 0 or 'num_followup_attempts' in state and state['num_followup_attempts'] >= configurable.max_followup_retries:
        return {
            "needs_followup": False,
        }
    
    notes = format_research_notes(state['notes'])
    
    # format the LLM prompt with the required information
    prompt = followup_prompt.format(
        user_prompt=state['topic'],
        summary=notes,
    )
    
    # call the LLM to obtain the structured output
    response = call_llm(configurable.model, prompt, structure=FollowupOutput)
    
    # obtain and increment the number of follow-up attempts
    num_followup = 0
    if 'num_followup_attempts' in state:
        num_followup = state['num_followup_attempts']
    if response.needs_followup:
        num_followup += 1
    
    return {
        "needs_followup": response.needs_followup,
        "follow_up_question": response.follow_up_question,
        "num_followup_attempts": num_followup,
    }
    
def route_followup(
    state: State, config: RunnableConfig
):
    """
    Determines whether to return to the query_generation node with a follow-up question, or proceed to the final_report node
    Returns True if follow-up is needed
    """
    return 'needs_followup' in state and state['needs_followup']
    
    
async def final_report(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """
    LangGraph node that generates the final report using the summarized search results
    
    Args:
        state: Current agent state containing user messages
        config: Runtime configuration with model settings and preferences
        
    Returns:
        Dictionary with the state updates:
            messages: The list of messages between the user and AI agent with the final report added as an AI message
            final_report: The final report formatted with Markdown
            research_notes: A snapshot of the notes used for generating the final report, used for debug and evaluation purposes
            topic: Clears the topic
            notes: Clears the list of research notes
            clarification_messages: Clears the list of clarification messages
            num_followup_attempts: Reset to 0
    """
    print('---- final_report')
    
    configurable = Configuration.from_runnable_config(config)
    
    notes = format_research_notes(state['notes'])
    
    # format the LLM prompt with the required information
    prompt = report_generation_prompt.format(
        user_prompt=state['topic'],
        research_notes=notes,
    )
    
    # call the LLM to obtain the final report
    response = call_llm(configurable.model, prompt)
    
    # return the final output and clear the intermediate fields
    return {
        "messages": [response],
        "final_report": response.text,
        "research_notes": format_research_notes(state['notes']),
        "topic": "",
        "notes": Overwrite([]),
        "clarification_messages": Overwrite([]),
        "num_followup_attempts": 0,
    }


# Define the graph
graph_builder = StateGraph(State, input=InputState, config_schema=Configuration)

# add all nodes in the graph
graph_builder.add_node(clarification)
graph_builder.add_node(query_generation)
graph_builder.add_node(search_results_extraction)
graph_builder.add_node(summarize)
graph_builder.add_node(followup)
graph_builder.add_node(final_report)
  
# add the edges between the nodes  
graph_builder.add_edge("__start__", "clarification")
graph_builder.add_conditional_edges("clarification", route_clarification, {True: END, False: "query_generation"})
graph_builder.add_edge("query_generation", "search_results_extraction")
graph_builder.add_conditional_edges("search_results_extraction", assign_workers, ["summarize"])
graph_builder.add_edge("summarize", "followup")
graph_builder.add_conditional_edges("followup", route_followup, {True: "query_generation", False: "final_report"})
graph_builder.add_edge("final_report", END)
    
graph = graph_builder.compile(name="Deep Research Agent")
