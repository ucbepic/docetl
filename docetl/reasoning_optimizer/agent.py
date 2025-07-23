import json
import os
import threading
import time
from typing import Dict

import litellm
import yaml
from op_descriptions import *
from pydantic import BaseModel

from docetl.reasoning_optimizer.directives import (
    ChainingDirective,
    ChangeModelDirective,
    GleaningDirective,
)
from docetl.reasoning_optimizer.load_data import load_input_doc
from docetl.utils import load_config

# argparse removed - use experiments/reasoning/run_baseline.py for CLI

# Global dictionary of rate limiters per model
model_rate_limiters: Dict[str, "TokenRateLimiter"] = {}
# Use environment variable or default to current directory
data_dir = os.environ.get("EXPERIMENT_DATA_DIR", "./data/")


def get_rate_limiter(model: str, max_tpm: int) -> "TokenRateLimiter":
    if model not in model_rate_limiters:
        model_rate_limiters[model] = TokenRateLimiter(max_tpm)
    return model_rate_limiters[model]


class TokenRateLimiter:
    def __init__(self, max_tpm):
        self.max_tpm = max_tpm
        self.tokens_used = 0
        self.lock = threading.Lock()
        self.reset_time = time.time() + 60  # 60 seconds window

    def allow(self, tokens):
        with self.lock:
            now = time.time()
            if now >= self.reset_time:
                self.tokens_used = 0
                self.reset_time = now + 60
            if self.tokens_used + tokens > self.max_tpm:
                return False
            self.tokens_used += tokens
            return True

    def wait_for_slot(self, tokens):
        while not self.allow(tokens):
            time_to_wait = max(0, self.reset_time - time.time())
            time.sleep(time_to_wait)


def count_tokens(messages):
    # messages should be a list of dicts, each with a "content" key
    total_chars = sum(
        len(m.get("content", "")) for m in messages if isinstance(m, dict)
    )
    return max(1, total_chars // 4)


class ResponseFormat(BaseModel):
    directive: str
    operators: List[str]


def get_openai_response(
    input_query,
    input_schema,
    input_data_sample,
    model="o3",
    max_tpm=5000000,
    message_history=[],
    curr_plan_output="",
    iteration=1,
):
    """
    The first LLM call. Generates a rewrite plan given the rewrite directives.
    """

    if iteration == 1:

        user_message = f"""
        I have a set of operations used to process long documents, along with a list of possible rewrite directives aimed at improving the quality of the query result.
        Given a query pipeline made up of these operations, recommend one specific rewrite directive (specify by its name) that would improve accuracy and pecify which operators (specify by the names) in the pipeline the directive should be applied to.
        Make sure that your cosen directive is in the provided list of rewrite directives.
        Pipeline:
        Pipelines in DocETL are the core structures that define the flow of data processing. A pipeline consists of five main components: \n
        - Default Model: The language model to use for the pipeline. Limit your choice of model to gpt-4.1-nano, gpt-4o-mini, gpt-4o, gpt-4.1 \n
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
        {ChainingDirective().to_string_for_plan()}\n
        {GleaningDirective().to_string_for_plan()}\n
        {ChangeModelDirective().to_string_for_plan()}\n

        Input document schema with token statistics: {input_schema} \n
        Input data sample: {json.dumps(input_data_sample, indent=2)[:5000]} \n
        The original query in YAML format using our operations: {input_query} \n
        Sample of the result from executing the original query: {json.dumps(curr_plan_output, indent=2)} \n
        """
    else:
        user_message = f"""
        Given the previously rewritten pipeline, recommend one specific rewrite directive (specify by its name) that would improve accuracy and pecify which operator (specify by the name) in the pipeline the directive should be applied to.
        Make sure that your cosen directive is in the provided list of rewrite directives.
         Rewrite directives:
        {ChainingDirective().to_string_for_plan()}\n
        {GleaningDirective().to_string_for_plan()}\n
        {ChangeModelDirective().to_string_for_plan()}\n

        Input data sample: {json.dumps(input_data_sample, indent=2)[:5000]} \n
        The original query in YAML format using our operations: {input_query} \n
        Sample of the result from executing the previously rewritten query: {json.dumps(curr_plan_output, indent=2)} \n
        """

    message_history.extend(
        [
            {
                "role": "system",
                "content": "You are an expert query optimization agent for document processing pipelines. Your role is to analyze user queries and apply rewrite directives to create more accurate execution plans. Your output must follow the structured output format.",
            },
            {"role": "user", "content": user_message},
        ]
    )
    messages = message_history

    # Enforce rate limit for the specified model
    if max_tpm > 0:
        limiter = get_rate_limiter(model, max_tpm)
        tokens = count_tokens(messages)
        limiter.wait_for_slot(tokens)

    response = litellm.completion(
        model=model,
        messages=messages,
        api_key=os.environ.get("AZURE_API_KEY"),
        api_base=os.environ.get("AZURE_API_BASE"),
        api_version=os.environ.get("AZURE_API_VERSION"),
        azure=True,
        reasoning_effort="high",
        response_format=ResponseFormat,
    )
    # response = litellm.completion(
    #     model=model,
    #     messages=messages,
    #     api_key=os.environ["GEMINI_API_KEY"],
    #     # reasoning_effort = "high",
    #     response_format=ResponseFormat
    # )
    assistant_response = response.choices[0].message.content

    # Add user and assistant messages to message_history as dicts
    message_history.append({"role": "assistant", "content": assistant_response})
    return assistant_response, message_history


def update_yaml_operations(input_file_path, output_file_path, new_operations):
    """
    Load a YAML file, replace the operations section, and save to a new file.

    Args:
        input_file_path (str): Path to the original YAML file
        output_file_path (str): Path where the modified YAML will be saved
        new_operations (list): List of operation dictionaries to replace the original operations
    """
    # Load the original YAML file
    with open(input_file_path, "r") as file:
        config = yaml.safe_load(file)

    # Replace the operations section
    config["operations"] = new_operations

    # Write the modified config to a new YAML file
    with open(output_file_path, "w") as file:
        yaml.dump(
            config, file, default_flow_style=False, allow_unicode=True, sort_keys=False
        )

    print(f"Modified YAML saved to: {output_file_path}")


def update_pipeline(orig_config, new_ops_list, target_ops):
    """
    Update the pipeline configuration with new operations.

    Args:
        orig_config (dict): The original pipeline configuration
        new_ops_list (list): List of new operations to add
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


def update_sample(new_ops_list, target_ops, orig_operators):
    """
    Update sample settings in new operations based on original operators.

    Args:
        new_ops_list (list): List of new operations to update
        target_ops (list): List of target operation names
        orig_operators (list): List of original operators

    Returns:
        list: Updated new operations list with sample settings
    """
    # Build a mapping from op name to op config in orig_operators
    op_name_to_config = {op.get("name"): op for op in orig_operators if "name" in op}

    # For each op in new_ops_list, if the corresponding op in orig_operators has 'sample', add it

    sample_size = -1
    for target_op_name in target_ops:
        target_op = op_name_to_config[target_op_name]
        if "sample" in target_op:
            sample_size = target_op["sample"]

    print("SAMPLE SIZE: ", sample_size)

    for op in new_ops_list:
        if sample_size != -1:
            op["sample"] = sample_size

    return new_ops_list


def save_message_history(message_history, filepath):
    """
    Save message history to a JSON file.

    Args:
        message_history (list): List of message dictionaries
        filepath (str): Path to save the message history
    """
    with open(filepath, "w") as f:
        json.dump(message_history, f, indent=2)
    print(f"Message history saved to: {filepath}")


def load_message_history(filepath):
    """
    Load message history from a JSON file.

    Args:
        filepath (str): Path to the message history file

    Returns:
        list: List of message dictionaries, or empty list if file doesn't exist
    """
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            return json.load(f)
    return []


def run_single_iteration(
    yaml_path, model, max_tpm, message_history, iteration_num, orig_output_sample
):
    """
    Run a single iteration of the optimization process.

    Args:
        yaml_path (str): Path to the YAML file
        model (str): Model name
        max_tpm (int): Tokens per minute limit
        message_history (list): Cumulative message history
        iteration_num (int): Current iteration number

    Returns:
        tuple: (output_file_path, updated_message_history)
    """
    print(f"\n=== Running Iteration {iteration_num} ===")
    print(f"Input file: {yaml_path}")

    # Parse input yaml file to get the list of operations
    orig_config = load_config(yaml_path)
    orig_operators = orig_config["operations"]

    sample_data_path = os.path.join(data_dir, "CUAD_random_sample.json")
    with open(sample_data_path, "r") as f:
        random_sample = json.load(f)

    with open(yaml_path, "r") as f:
        input_query = f.read()

    input_schema = load_input_doc(yaml_path)

    reply, message_history = get_openai_response(
        input_query,
        input_schema,
        random_sample,
        model=model,
        max_tpm=max_tpm,
        message_history=message_history,
        curr_plan_output=orig_output_sample,
        iteration=iteration_num,
    )
    print("Agent:", reply)

    # Parse agent response
    try:
        parsed = json.loads(reply)
        directive = parsed.get("directive")
        target_ops = parsed.get("operators")
        print(f"Directive: {directive}, Target ops: {target_ops}")
    except Exception as e:
        print(f"Failed to parse agent response: {e}")
        return None, message_history

    new_ops_list = None
    if directive == "chaining":
        new_ops_list, message_history = ChainingDirective().instantiate(
            operators=orig_operators,
            target_ops=target_ops,
            agent_llm=model,
            message_history=message_history,
        )
        orig_config["operations"] = new_ops_list

        orig_config = update_pipeline(orig_config, new_ops_list, target_ops)
        new_ops_list = update_sample(new_ops_list, target_ops, orig_operators)
        orig_config["operations"] = new_ops_list

    elif directive == "gleaning":
        new_ops_list, message_history = GleaningDirective().instantiate(
            operators=orig_operators,
            target_ops=target_ops,
            agent_llm=model,
            message_history=message_history,
        )
        orig_config["operations"] = new_ops_list

    elif directive == "change model":
        new_ops_list, message_history = ChangeModelDirective().instantiate(
            operators=orig_operators,
            target_ops=target_ops,
            agent_llm=model,
            message_history=message_history,
        )
        orig_config["operations"] = new_ops_list

    output_file_path = os.path.join(
        data_dir,
        "agent_optimized_plan/group10",
        f"CUAD-map_opt_iter_{iteration_num}.yaml",
    )

    # Ensure every operator's model starts with 'azure/'
    for op in orig_config.get("operations", []):
        if "model" in op and not str(op["model"]).startswith("azure/"):
            op["model"] = "azure/" + str(op["model"])

    # Add bypass_cache: true at the top level
    orig_config["bypass_cache"] = True

    # Save the modified config
    with open(output_file_path, "w") as file:
        yaml.dump(
            orig_config,
            file,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        )

    print(f"Modified YAML saved to: {output_file_path}")

    return output_file_path, message_history


# agent.py is now a pure module - no experiment code
# Use experiments/reasoning/run_baseline.py to run experiments
