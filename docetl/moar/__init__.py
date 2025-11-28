"""
MCTS (Monte Carlo Tree Search) module for DocETL optimization.

This module provides Monte Carlo Tree Search optimization for DocETL pipelines
using Pareto frontier analysis and multi-objective optimization.
"""

# Default list of available models for MOAR optimization
# Defined here before imports to avoid circular import issues
AVAILABLE_MODELS = [
    # "gpt-5",
    # "gpt-5-mini",
    # "gpt-5-nano",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "gpt-4o",
    "gpt-4o-mini",
    # "gemini-2.5-pro",
    # "gemini-2.5-flash",
    # "gemini-2.5-flash-lite"
]

from .MOARSearch import MOARSearch  # noqa: E402
from .Node import Node  # noqa: E402
from .ParetoFrontier import ParetoFrontier  # noqa: E402

__all__ = [
    "MOARSearch",
    "Node", 
    "ParetoFrontier",
    "AVAILABLE_MODELS"
]