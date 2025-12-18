from langsmith import Client
from dotenv import load_dotenv
import asyncio
from langgraph.checkpoint.memory import MemorySaver
import uuid
from pydantic import BaseModel, Field
import os
from langchain_openai import ChatOpenAI

from agent.graph import graph_builder
from agent.prompts import evaluator_prompt
from agent.utils import (
    call_llm,
    format_clarification_messages,
    format_research_notes,
)

load_dotenv(".env")

client = Client(api_key=os.getenv("LANGSMITH_API_KEY"))
dataset_name = "Deep Research Bench"
dataset_limit = 10 # number of examples to evaluate on

# config values
metadata = {
    "model": 'gpt-4.1-nano', # The OpenAI model to use
    "max_clarification_retries": 0, # Maximum number of times to ask for clarification
    "num_queries": 3, # Number of queries to generate
    "num_results_per_query": 2, # Maximum number of results to fetch per query
    "max_followup_retries": 1, # Maximum number of times to followup
}

class EvaluationScores(BaseModel):
    grounding_and_accuracy: int = Field(description="Are claims in the report supported by the provided research notes")
    coverage_and_depth: int = Field(description="Does the report adequately address the core aspects of the user topic")
    synthesis_and_reasoning: int = Field(description="Are relationships, trends, or trade-offs clearly explained")
    structure_and_clarity: int = Field(description="Are sections logically ordered and clearly labeled")
    usefulness_to_user: int = Field(description="Would this report meaningfully help a user understand the topic or make decisions")


async def data_generator():
    """
    Limit and offset the number of examples from the dataset
    """
    examples = client.list_examples(dataset_name=dataset_name, offset=50, limit=dataset_limit)
    for example in examples:
        yield example


async def target(inputs: dict):
    """
    Compile the agent graph and run with the provided input to obtain the final state
    
    Args:
        inputs: The input messages to the agent
        
    Return:
        The final state after running the agent
    """
    graph = graph_builder.compile(checkpointer=MemorySaver())
    config = {
        "configurable": {
            "thread_id": str(uuid.uuid4()),
        }
    }
    
    config["configurable"].update(metadata)
    
    # NOTE: We do not use MCP tools to stay consistent
    final_state = await graph.ainvoke(
        {"messages": [{"role": "user", "content": inputs["messages"][0]["content"]}]},
        config
    )
    return final_state


def evaluator(inputs: dict, outputs: dict):
    """
    Use an LLM as evaluator to obtain scores based on the input messages and the final state outputted by the agent
    
    Args:
        inputs: The input messages to the agent
        outputs: The final state after running the agent
        
    Return:
        The evaluation scores for the quality of the result
    """
    messages = inputs["messages"]
    topic = messages[0]["content"]
    
    clarification = 'None'
    if len(messages) > 1:
        clarification = format_clarification_messages(messages[1:])

    prompt = evaluator_prompt.format(
        user_prompt = topic,
        messages = clarification,
        research_notes = outputs['research_notes'],
        final_report = outputs["final_report"],
    )
    
    eval_result = ChatOpenAI(
        model="gpt-4.1-nano",
    ).with_structured_output(EvaluationScores).invoke(prompt)
    
    return [
        {"key": "grounding_and_accuracy", "score": eval_result.grounding_and_accuracy / 5},
        {"key": "coverage_and_depth", "score": eval_result.coverage_and_depth / 5},
        {"key": "synthesis_and_reasoning", "score": eval_result.synthesis_and_reasoning / 5},
        {"key": "structure_and_clarity", "score": eval_result.structure_and_clarity / 5},
        {"key": "usefulness_to_user", "score": eval_result.usefulness_to_user / 5},
    ]


async def main():
    return await client.aevaluate(
        target,
        data=data_generator(),
        evaluators=[evaluator],
        experiment_prefix=f"DeepResearcher GPT-4.1-nano, Tavily Search",
        max_concurrency=10,
        metadata=metadata,
    )

if __name__ == "__main__":
    results = asyncio.run(main())
    print(results)