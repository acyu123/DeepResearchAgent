
import os

def get_llm_api_key():
    return os.getenv("OPENAI_API_KEY")
    
def get_search_api_key():
    return os.getenv("TAVILY_API_KEY")

