
import os
from datetime import datetime
import json

from langchain_openai import ChatOpenAI

from tavily import TavilyClient


def get_search_api_key():
    return os.getenv("TAVILY_API_KEY")

def get_llm():
    return 

def get_current_date():
    return datetime.now().strftime("%B %d, %Y")

def format_clarification_messages(messages):
    clarification_messages = 'None'
    if messages:
        clarification_messages = '\n'.join([m.text for m in messages])
        
    return clarification_messages

def call_llm(prompt, return_json=False):
    llm = ChatOpenAI(
        model="gpt-4.1-nano",
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
    
    response = llm.invoke(prompt)
    
    if return_json:
        try:
            response = json.loads(response.text)
        except Exception as e:
            print(response)
            response = llm.invoke(prompt)
            response = json.loads(response.text)
            
    return response

def get_search_results(query, max_results):
    client = TavilyClient(os.getenv("TAVILY_API_KEY"))
    response = client.search(
        query=query,
        max_results=max_results*2,
        include_raw_content="text"
    )
    
    results = [r for r in response['results'] if r['raw_content']]
    results = results[:max_results]
    print(results)
    
    return [{'title': r['title'], 'url': r['url'], 'content': r['raw_content']} for r in results]
