"""
Utility functions for MCTS implementation.

This module contains helper functions and utilities used by the MCTS algorithm
but not core to the MCTS structure itself.
"""

import json
import math
import os
import time
from typing import List, Optional

from pydantic import BaseModel

from docetl.reasoning_optimizer.directives import (
    DIRECTIVE_GROUPS,
    Directive,
    get_all_cost_directive_strings,
    get_all_directive_strings,
)
from docetl.reasoning_optimizer.load_data import load_input_doc
from docetl.reasoning_optimizer.op_descriptions import *  # noqa: F403, F405

# Maximum number of tokens we will allow in the prompt we send to the model.
# The Azure GPT-5 family allows 272,000 tokens.
MAX_CONTEXT_TOKENS = 270_000


class ExpandResponseFormat(BaseModel):
    directive: str
    operators: List[str]


def count_tokens(messages):
    """Count estimated tokens in messages list."""
    # messages should be a list of dicts, each with a "content" key
    total_chars = sum(
        len(m.get("content", "")) for m in messages if isinstance(m, dict)
    )
    return max(1, total_chars // 4)


def trim_history(history: list, keep_system_first: bool = True) -> list:
    """Trim the conversation history in-place so its estimated token count
    (via ``count_tokens``) does not exceed ``MAX_CONTEXT_TOKENS``.

    We always keep the very first system message and the first user message so the
    assistant retains the global instructions and the initial query context. After
    that we drop the oldest messages until the budget is satisfied. Returns the
    trimmed history list.
    """

    # Determine starting index to preserve the initial system message and first user message
    start_idx = 0
    if keep_system_first and history:
        if history[0].get("role") == "system":
            start_idx = 1
            # Find the first user message after the system message
            for i in range(1, len(history)):
                if history[i].get("role") == "user":
                    start_idx = i + 1
                    break
        elif history[0].get("role") == "user":
            # If first message is user, keep it and find the next user message
            start_idx = 1
            for i in range(1, len(history)):
                if history[i].get("role") == "user":
                    start_idx = i + 1
                    break

    # Drop oldest messages (just after the preserved block) until within limit
    while len(history) > start_idx + 1 and count_tokens(history) > MAX_CONTEXT_TOKENS:
        history.pop(start_idx)

    return history


def get_directive_group(directive_name: str) -> str:
    """
    Get the group name for a directive.

    Args:
        directive_name: Name of the directive

    Returns:
        Group name if found, None otherwise
    """
    for group_name, directives in DIRECTIVE_GROUPS.items():
        for directive in directives:
            if directive.name == directive_name:
                return group_name
    return None


def get_excluded_directives_for_operation(node, op_name: str) -> set:
    """Get compression directives to exclude for code_map and extract operations."""
    op_type = node.op_dict[op_name].get("type")
    compression_exclusions = set()
    if op_type in ["code_map", "extract"]:
        compression_exclusions = set(DIRECTIVE_GROUPS.get("compression", []))
    return compression_exclusions


def is_action_applicable(node, action: Directive) -> bool:
    """Check if an action is applicable to a node."""
    return True


def update_pipeline(orig_config, new_ops_list, target_ops):
    """
    Update the pipeline configuration with new operations.

    Args:
        orig_config (dict): The original pipeline configuration
        new_ops_list (list): The entire pipeline operations list (not a subset)
        target_ops (list): List of target operation names to replace

    Returns:
        dict: Updated pipeline configuration
    """
    if new_ops_list is not None:
        op_names = [op.get("name") for op in new_ops_list if "name" in op]

    # Update the pipeline steps to use the new operation names
    if "pipeline" in orig_config and "steps" in orig_config["pipeline"]:
        for step in orig_config["pipeline"]["steps"]:
            if "operations" in step:
                new_ops = []
                for op in step["operations"]:
                    if op == target_ops[0]:
                        new_ops.extend(op_names)
                step["operations"] = new_ops

    return orig_config


def fix_models(parsed_yaml):
    """No-op: Model names should be specified correctly in the YAML."""
    pass


def is_fully_explored(node, max_children_multiplier: float = 1.0) -> bool:
    """Check if a node has been fully explored based on visit count."""
    if node.parent is None:
        return True
    allowed_children = max(
        2, 1 + math.floor(math.sqrt(float(node.visits)) * max_children_multiplier)
    )

    # Not only check the number of children, but also ensure all children have been simulated
    # A child is considered simulated if it has been visited at least once (visits > 0)
    # This ensures that children created but not yet simulated won't cause the parent to be considered fully explored
    if len(node.children) < allowed_children:
        return False

    # Check if all children have been simulated (visited at least once)
    # This prevents selecting children that were just created but not yet simulated
    for child in node.children:
        if child.visits == 0:
            # This child hasn't been selected/simulated yet, so parent is not fully explored
            return False

    return True


def should_continue_search(
    iteration: int,
    max_iterations: int,
    start_time: float,
    max_time: Optional[float] = None,
) -> bool:
    """Determine if search should continue based on iteration count and time."""
    if iteration >= max_iterations:
        return False

    if max_time is not None:
        elapsed_time = time.time() - start_time
        if elapsed_time >= max_time:
            return False

    return True


def calculate_ucb1(
    node, parent_visits: int, exploration_constant: float = math.sqrt(2)
) -> float:
    """Calculate UCB1 value for node selection."""
    if node.visits == 0:
        return float("inf")

    exploitation = node.value / node.visits
    exploration = exploration_constant * math.sqrt(
        math.log(parent_visits) / node.visits
    )
    return exploitation + exploration


def print_tree_visits_and_values(node=None, depth=0, file_handle=None, console=None):
    """Print tree structure with visit counts and values."""
    if node is None:
        return

    indent = "  " * depth
    node_info = f"{indent}Node ID: {node.get_id()}, Visits: {node.visits}, Value: {node.value:.4f}"

    if file_handle:
        file_handle.write(node_info + "\n")
    else:
        if console is None:
            from docetl.console import DOCETL_CONSOLE

            console = DOCETL_CONSOLE
        console.log(node_info)

    for child in node.children:
        print_tree_visits_and_values(child, depth + 1, file_handle, console)


def log_tree_to_file(root_node, iteration_num, output_dir="./outputs", console=None):
    """Log the tree structure to a file."""
    log_file_path = os.path.join(output_dir, f"moar_tree_iteration_{iteration_num}.txt")

    with open(log_file_path, "w") as f:
        f.write(f"MOAR Tree Structure - Iteration {iteration_num}\n")
        f.write("=" * 50 + "\n")
        print_tree_visits_and_values(root_node, file_handle=f, console=console)


def create_expansion_prompt_acc(
    node,
    action_options,
    input_query,
    available_actions,
    action_cost_changes,
    action_accuracy_changes,
    action_counts,
    sample_input,
    root_node,
    yaml_file_path,
    dataset=None,
    node_accuracies=None,
    model_stats=None,
    available_models=None,
) -> tuple[str, str]:
    """Create expansion prompt for accuracy optimization."""

    # Use provided available_models list, or extract from model_stats as fallback
    if available_models:
        available_models_list = available_models
    elif model_stats:
        available_models_list = list(model_stats.keys())
    else:
        available_models_list = []

    availabel_actions_str = ""
    for item in action_options:
        op_name = item[0]
        action_name = item[1]
        action_str = f"Operator: {op_name}, Rewrite directive: {action_name}\n"
        availabel_actions_str += action_str

    action_stats = []
    for action in available_actions:
        cost_change = action_cost_changes.get(action, 0)
        accuracy_change = action_accuracy_changes.get(action, 0)
        count = action_counts.get(action, 0)

        if count > 0:
            avg_cost_change = cost_change / count
            avg_accuracy_change = accuracy_change / count
            action_stats.append(
                f"- {action.name}: {count} uses, avg change in cost: {avg_cost_change:+.2f}, avg change in accuracy: {avg_accuracy_change:+.4f}"
            )
        else:
            action_stats.append(
                f"- {action.name}: {count} uses, avg change in cost: Unknown (never tried), avg change in accuracy: Unknown (never tried)"
            )

    action_stats_str = "\n".join(action_stats)

    input_schema = load_input_doc(yaml_file_path)

    user_message = f"""
    I have a set of operations used to process long documents, along with a list of possible rewrite directives aimed at improving the quality of the query result.
    Given a query pipeline made up of these operations, recommend one specific rewrite directive (specify by its name) that would improve accuracy and specify which operators (specify by their names) in the pipeline the directive should be applied to.
    Make sure that your chosen directive is in the provided list of rewrite directives.

    Pipeline:
    Pipelines in DocETL are the core structures that define the flow of data processing. A pipeline consists of five main components: \n
    - Default Model: The language model to use for the pipeline. Limit your choice of model to {available_models_list} \n
    - System Prompts: A description of your dataset and the "persona" you'd like the LLM to adopt when analyzing your data. \n
    - Datasets: The input data sources for your pipeline. \n
    - Operators: The processing steps that transform your data. \n
    - Pipeline Specification: The sequence of steps and the output configuration. \n

    Operators:
    Operators form the building blocks of data processing pipelines. Below is the list of operators:
    {op_map.to_string()}\n
    {op_extract.to_string()}\n
    {op_parallel_map.to_string()}\n
    {op_filter.to_string()}\n
    {op_reduce.to_string()}\n
    {op_split.to_string()}\n
    {op_gather.to_string()}\n
    {op_unnest.to_string()}\n
    {op_sample.to_string()}\n
    {op_resolve.to_string()}\n

    Rewrite directives:
    {get_all_directive_strings()}\n

    Your valid choice of operation and rewrite directive combination. Only choose one of these:\n
    {availabel_actions_str}

    Action Performance History:
    Based on previous executions across DIFFERENT query pipelines, here's how each action has performed:\n
    {action_stats_str}

    Note: These statistics come from applying actions to various other query pipelines, not the current one. Use this as general guidance about action effectiveness, but consider that performance may vary significantly for your specific pipeline structure and data.

    Model Performance Reference:
    If you are considering a model change directive, here are model statistics on this specific dataset: \n {str(model_stats if model_stats else {})}
    These show the accuracy (acc) and cost for each model. Only reference this when evaluating model change options.

    Selection Strategy:
    Consider the current query pipeline, which directive can best improve the accuracy.
    Prioritize exploration of untested actions while balancing with exploitation of proven performers:
    - Actions with 0 uses have unknown potential, so you should explore them if applicable.
    - fusion directives are helpful
    - Consider both immediate improvement and learning about the action space

    {node.get_memo_for_llm(root_node, node_accuracies)}

    Make sure you read every rewrite directive carefully.
    Make sure you only choose from the valid choices above and avoid already used combinations or approaches too similar to what has already been tried in the current optimization path.

    Input document schema with token statistics: {input_schema} \n
    Input data sample: {json.dumps(sample_input, indent=2)[:5000]} \n
    The original query in YAML format using our operations: {input_query} \n
    The original query result: {json.dumps(node.sample_result, indent=2)[:3000]} \n
    """

    # Create a condensed version for message history (without full operator/directive descriptions)
    condensed_user_message = f"""
    Recommend one specific rewrite directive for accuracy optimization.

    Valid choices:
    {availabel_actions_str}

    Action Performance History:
    {action_stats_str}

    Current pipeline: {input_query}
    """

    return user_message, condensed_user_message


def create_expansion_prompt_cost(
    node,
    action_options,
    input_query,
    available_actions,
    action_cost_changes,
    action_accuracy_changes,
    action_counts,
    sample_input,
    root_node,
    yaml_file_path,
    dataset=None,
    node_accuracies=None,
    model_stats=None,
    available_models=None,
) -> tuple[str, str]:
    """Create expansion prompt for cost optimization."""

    # Use provided available_models list, or extract from model_stats as fallback
    if available_models:
        available_models_list = available_models
    elif model_stats:
        available_models_list = list(model_stats.keys())
    else:
        available_models_list = []

    availabel_actions_str = ""
    for item in action_options:
        op_name = item[0]
        action_name = item[1]
        action_str = f"Operator: {op_name}, Rewrite directive: {action_name}\n"
        availabel_actions_str += action_str

    action_stats = []
    for action in available_actions:
        cost_change = action_cost_changes.get(action, 0)
        accuracy_change = action_accuracy_changes.get(action, 0)
        count = action_counts.get(action, 0)

        if count > 0:
            avg_cost_change = cost_change / count
            avg_accuracy_change = accuracy_change / count
            action_stats.append(
                f"- {action.name}: {count} uses, avg change in cost: {avg_cost_change:+.2f}, avg change in accuracy: {avg_accuracy_change:+.4f}"
            )
        else:
            action_stats.append(
                f"- {action.name}: {count} uses, avg change in cost: Unknown (never tried), avg change in accuracy: Unknown (never tried)"
            )

    action_stats_str = "\n".join(action_stats)

    input_schema = load_input_doc(yaml_file_path)

    user_message = f"""
    I have a set of operations used to process long documents, along with a list of possible rewrite directives designed to improve the cost effectiveness of the pipeline, while maintaining similar or better accuracy.
    Given a query pipeline composed of these operations, recommend one specific rewrite directive (identified by its name from the provided list) that would improve cost effectiveness. Also, specify which operator(s) (by name) in the pipeline the directive should be applied to.
    Make sure your recommended directive is selected from the provided list.

    Pipeline:
    Pipelines in DocETL are the core structures that define the flow of data processing. A pipeline consists of five main components: \n
    - Default Model: The language model to use for the pipeline. Limit your choice of model to {available_models_list} \n
    - System Prompts: A description of your dataset and the "persona" you'd like the LLM to adopt when analyzing your data. \n
    - Datasets: The input data sources for your pipeline. \n
    - Operators: The processing steps that transform your data. \n
    - Pipeline Specification: The sequence of steps and the output configuration. \n

    Operators:
    Operators form the building blocks of data processing pipelines. Below is the list of operators:
    {op_map.to_string()}\n
    {op_extract.to_string()}\n
    {op_parallel_map.to_string()}\n
    {op_filter.to_string()}\n
    {op_reduce.to_string()}\n
    {op_split.to_string()}\n
    {op_gather.to_string()}\n
    {op_unnest.to_string()}\n
    {op_sample.to_string()}\n
    {op_resolve.to_string()}\n

    Rewrite directives:
    {get_all_cost_directive_strings()}\n

    Your valid choice of operation and rewrite directive combination. Only choose one of these:\n
    {availabel_actions_str}

    Action Performance History:
    Based on previous executions across DIFFERENT query pipelines, here's how each action has performed:\n
    {action_stats_str}

    Note: These statistics come from applying actions to various other query pipelines, not the current one. Use this as general guidance about action effectiveness, but consider that performance may vary significantly for your specific pipeline structure and data.

    Model Performance Reference:
    If you are considering a model change directive, here are model statistics on this specific dataset: \n {str(model_stats if model_stats else {})}
    These show the accuracy (acc) and cost for each model. Only reference this when evaluating model change options.

    Selection Strategy:
    Consider the current query pipeline, which directive can best improve cost effectiveness.
    Prioritize exploration of untested actions while balancing with exploitation of proven performers:
    - Actions with 0 uses have unknown potential, so you should explore them if applicable.
    - fusion directives are helpful
    - Consider both immediate improvement and learning about the action space

    {node.get_memo_for_llm(root_node, node_accuracies)}

    Make sure you only choose from the valid choices above and avoid already used combinations or approaches too similar to what has already been tried in the current optimization path.

    Input document schema with token statistics: {input_schema} \n
    Input data sample: {json.dumps(sample_input, indent=2)[:5000]} \n
    The original query in YAML format using our operations: {input_query} \n
    The original query result: {json.dumps(node.sample_result, indent=2)[:3000]} \n
    """

    condensed_user_message = f"""
    Recommend one specific rewrite directive for cost optimization.

    Valid choices:
    {availabel_actions_str}

    Action Performance History:
    {action_stats_str}

    Current pipeline: {input_query}
    """

    return user_message, condensed_user_message
