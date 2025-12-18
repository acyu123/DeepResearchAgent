
import os
from datetime import datetime
import json

from langchain_openai import ChatOpenAI

from tavily import TavilyClient


def get_current_date():
    """
    Obtain the current date to use in LLM prompts
    
    Return:
        The current date in format: Month Day, Year
    """
    return datetime.now().strftime("%B %d, %Y")


def format_clarification_messages(messages):
    """
    Converts a list of messages into a single string where each line is a message text
    
    Args:
        messages: A list of messages containing the clarification questions and answers
        
    Return:
        A single string with a message on each line
    """
    
    clarification_messages = 'None'
    if messages:
        clarification_messages = '\n'.join([m.text for m in messages])
        
    return clarification_messages


def call_llm(model, prompt, structure=None):
    """
    Calls the OpenAI api with the given prompt and returns the LLM output
    
    Args:
        model: The name of the OpenAI GPT model to use
        prompt: The prompt to input into the LLM
        structure: The class that defines the fields of the structured output. If not provided, the output would be unstructured text
        
    Return:
        The response from the LLM, either a structured object or an unstructured AI message
    """
    
    llm = ChatOpenAI(
        model=model,
        # stream_usage=True,
        # temperature=None,
        # max_tokens=None,
        # timeout=None,
        # reasoning_effort="low",
        # max_retries=2,
        api_key=os.getenv("OPENAI_API_KEY"),
        # base_url="...",
        # organization="...",
        # other params...
    )
    
    if structure:
        response = llm.with_structured_output(structure).invoke(prompt) # return structured output
    else:
        response = llm.invoke(prompt) # return text output
            
    return response


def get_search_results(query, max_results):
    """
    Calls the search API with the query to return the title, url, and raw content of the results
    
    Args:
        query: The search query
        max_results: The maximum number of results to return
        
    Return:
        A list of search results for the query, containing the title, url, and raw content
    """
    
    client = TavilyClient(os.getenv("TAVILY_API_KEY"))
    
    # obtain more than max_results results from the search API
    response = client.search(
        query=query,
        max_results=max_results*2,
        include_raw_content="text"
    )
    
    # obtain only max_results results where the raw content is available
    results = [r for r in response['results'] if r['raw_content']]
    results = results[:max_results]
    
    return [{
        'query': query, 
        'title': r['title'], 
        'url': r['url'], 
        'content': r['raw_content']
    } for r in results]


def format_research_notes(notes):
    """
    Converts the notes for each source into a single string where each section starts with the title and url.
    
    Args:
        notes: The list of research notes for each source, with the corresponding title and url
        
    Returns:
        A single string where each section starts with the title and url
    """
    result = ""
    for source in notes:
        # clean up empty lines
        content = source['notes']
        lines = content.splitlines()
        content = '\n'.join([line for line in lines if line.strip()])
    
        # add the notes in the format:
        # [title] ([url]): 
        # [content]
        result += '\n'+source['title']+' ('+source['url']+'):'+'\n'+content+'\n'
    return result
