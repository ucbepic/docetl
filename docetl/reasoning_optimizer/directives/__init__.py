from typing import Dict, List, Tuple
from .base import (
    Directive, 
    DirectiveTestCase, 
    TestResult, 
    MAX_DIRECTIVE_INSTANTIATION_ATTEMPTS,
    DEFAULT_MODEL,
    DEFAULT_MAX_TPM,
    AVAILABLE_MODELS,
    DEFAULT_OUTPUT_DIR
)
from .chaining import ChainingDirective
from .gleaning import GleaningDirective
from .change_model import ChangeModelDirective
from .change_model_acc import ChangeModelAccDirective
from .change_model_cost import ChangeModelCostDirective
from .doc_summarization import DocSummarizationDirective
from .isolating_subtasks import IsolatingSubtasksDirective
from .doc_compression import DocCompressionDirective
from .deterministic_doc_compression import DeterministicDocCompressionDirective
from .reduce_gleaning import ReduceGleaningDirective
from .reduce_chaining import ReduceChainingDirective
from .operator_fusion import OperatorFusionDirective
from .doc_chunking import DocumentChunkingDirective
from .doc_chunking_topk import DocumentChunkingTopKDirective
from .chunk_header_summary import ChunkHeaderSummaryDirective
from .take_head_tail import TakeHeadTailDirective
from .clarify_instructions import ClarifyInstructionsDirective
from .swap_with_code import SwapWithCodeDirective
from .map_reduce_fusion import MapReduceFusionDirective
from .hierarchical_reduce import HierarchicalReduceDirective
from .cascade_filtering import CascadeFilteringDirective
from .arbitrary_rewrite import ArbitraryRewriteDirective

# Registry of all available directives
ALL_DIRECTIVES = [
    ChainingDirective(),
    GleaningDirective(), 
    ReduceGleaningDirective(),
    ReduceChainingDirective(),
    # ChangeModelAccDirective(),
    ChangeModelCostDirective(),
    DocSummarizationDirective(),
    IsolatingSubtasksDirective(),
    DocCompressionDirective(),
    DeterministicDocCompressionDirective(),
    OperatorFusionDirective(),
    DocumentChunkingDirective(),
    DocumentChunkingTopKDirective(),
    ChunkHeaderSummaryDirective(),
    TakeHeadTailDirective(),
    ClarifyInstructionsDirective(),
    SwapWithCodeDirective(),
    MapReduceFusionDirective(),
    HierarchicalReduceDirective(),
    CascadeFilteringDirective(),
    ArbitraryRewriteDirective(),
]

ALL_COST_DIRECTIVES = [
    ReduceChainingDirective(),
    DocSummarizationDirective(),
    DocCompressionDirective(),
    DeterministicDocCompressionDirective(),
    DocumentChunkingTopKDirective(),
    OperatorFusionDirective(),
    TakeHeadTailDirective(),
    MapReduceFusionDirective(),
    ChangeModelCostDirective(),
    CascadeFilteringDirective(),
    ArbitraryRewriteDirective(),
]

MODEL_STATS = {
    "cuad": {
        "gpt-5": {"acc": 0.7200038754634324, "cost": 2.282243},
        "gpt-5-mini": {"acc": 0.6835748934962984, "cost": 0.605154},
        "gpt-5-nano": {"acc": 0.6463552293414165, "cost": 0.231796},
        "gpt-4.1": {"acc": 0.7366652765882619, "cost": 2.026396},
        "gpt-4.1-mini": {"acc": 0.6914865483894942, "cost": 0.380290},
        "gpt-4.1-nano": {"acc": 0.36382364868308986, "cost": 0.080214},
        "gpt-4o": {"acc": 0.6214280458632053, "cost": 2.210165},
        "gpt-4o-mini": {"acc": 0.4311917773936343, "cost": 0.137882},
        "gemini-2.5-pro": {"acc": 0.7174635945681389, "cost": 2.820552},
        "gemini-2.5-flash": {"acc": 0.7170929338182611, "cost": 1.032156},
        "gemini-2.5-flash-lite": {"acc": 0.6087973440184378, "cost": 0.097669},
    }
}

DIRECTIVE_GROUPS = {
    "compression": [
        DocCompressionDirective(),
        DocSummarizationDirective(),
        DeterministicDocCompressionDirective(),
    ],
    "chunking": [
        DocumentChunkingDirective(),
        DocumentChunkingTopKDirective(),
    ]
}

MULTI_INSTANCE_DIRECTIVES = [
    DocumentChunkingDirective(),
    DocumentChunkingTopKDirective(),
    ChangeModelAccDirective(),
    ChangeModelCostDirective(),
    DeterministicDocCompressionDirective(),
    TakeHeadTailDirective(),
    CascadeFilteringDirective(),
]   

# Create a mapping from directive names to directive instances
DIRECTIVE_REGISTRY = {directive.name: directive for directive in ALL_DIRECTIVES}

def get_all_directive_strings() -> str:
    """
    Generate string descriptions for all available directives for use in prompts.
    
    Returns:
        str: Formatted string containing all directive descriptions
    """
    return "\n".join([directive.to_string_for_plan() for directive in ALL_DIRECTIVES])


def get_all_cost_directive_strings() -> str:
    return "\n".join([directive.to_string_for_plan() for directive in ALL_COST_DIRECTIVES])


def instantiate_directive(
    directive_name: str,
    operators: List[Dict],
    target_ops: List[str], 
    agent_llm: str,
    message_history: list,
    global_default_model: str = None,
    dataset: str = None,
    **kwargs
) -> Tuple[List[Dict], list]:
    """
    Centralized method to instantiate any directive by name.
    
    Args:
        directive_name: Name of the directive to instantiate
        operators: List of pipeline operators
        target_ops: List of target operation names
        agent_llm: LLM model to use
        message_history: Conversation history
        **kwargs: Additional arguments to pass to directive
    
    Returns:
        Tuple of (new_ops_list, updated_message_history)
        
    Raises:
        ValueError: If directive_name is not recognized
    """
    if message_history is None:
        message_history = []
        
    if directive_name not in DIRECTIVE_REGISTRY:
        available = list(DIRECTIVE_REGISTRY.keys())
        raise ValueError(f"Unknown directive '{directive_name}'. Available: {available}")
    
    directive = DIRECTIVE_REGISTRY[directive_name]
    return directive.instantiate(
        operators=operators,
        target_ops=target_ops,
        agent_llm=agent_llm,
        message_history=message_history,
        global_default_model = global_default_model,
        dataset=dataset,
        **kwargs
    )

__all__ = [
    "Directive",
    "DirectiveTestCase", 
    "TestResult",
    "MAX_DIRECTIVE_INSTANTIATION_ATTEMPTS",
    "DEFAULT_MODEL",
    "DEFAULT_MAX_TPM", 
    "AVAILABLE_MODELS",
    "DEFAULT_OUTPUT_DIR",
    "ChainingDirective",
    "GleaningDirective",
    "ReduceGleaningDirective",
    "ReduceChainingDirective",
    "ChangeModelDirective",
    "DocSummarizationDirective",
    "IsolatingSubtasksDirective",
    "DocCompressionDirective",
    "DeterministicDocCompressionDirective",
    "OperatorFusionDirective",
    "DocumentChunkingDirective",
    "DocumentChunkingTopKDirective",
    "ChunkHeaderSummaryDirective",
    "TakeHeadTailDirective",
    "ClarifyInstructionsDirective",
    "SwapWithCodeDirective",
    "MapReduceFusionDirective",
    "HierarchicalReduceDirective",
    "CascadeFilteringDirective",
    "ArbitraryRewriteDirective",
    "ALL_DIRECTIVES",
    "DIRECTIVE_REGISTRY", 
    "get_all_directive_strings",
    "instantiate_directive"
]