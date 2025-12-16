import os
from enum import Enum
from dataclasses import dataclass, fields
from typing import Any, Optional, Dict, Literal

from langchain_core.runnables import RunnableConfig

@dataclass(kw_only=True)
class Configuration:
    
    max_clarification_retries: int = 0 # Maximum number of times to ask for clarification
    num_queries: int = 1 # Number of queries to generate
    num_results_per_query: int = 3 # Maximum number of results to fetch per query
    max_followup_retries: int = 1 # Maximum number of times to followup

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )
        values: dict[str, Any] = {
            f.name: os.environ.get(f.name.upper(), configurable.get(f.name))
            for f in fields(cls)
            if f.init
        }
        return cls(**{k: v for k, v in values.items() if v})
