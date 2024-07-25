"""
This file describes mapper optimizations.
"""

from motion.agent import Agent, OpenAILLM
from typing import List, Tuple, Any
from motion.executor import Operation


def try_gleaning(
    operation: Operation,
    sample_data: List[Tuple[Any, Any]],
    errors: List[Tuple[str, Any, int]],
):
    # Keep querying an LLM
    pass
