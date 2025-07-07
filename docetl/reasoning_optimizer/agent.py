import random
from docetl.reasoning_optimizer.instantiate_schemas import ChangeModelConfig
from docetl.reasoning_optimizer.prompts import PromptLibrary
import litellm
import os
import threading
import time
import yaml  
import json
import sys
from pydantic import BaseModel
from typing import Dict
from docetl.reasoning_optimizer.load_data import load_input_doc
from op_descriptions import *
from ChainingDirective import *
from GleaningDirective import *
from ChangeModelDirective import *
from docetl.utils import load_config
import argparse

# Global dictionary of rate limiters per model
model_rate_limiters: Dict[str, 'TokenRateLimiter'] = {}

def get_rate_limiter(model: str, max_tpm: int) -> 'TokenRateLimiter':
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
    total_chars = sum(len(m.get("content", "")) for m in messages if isinstance(m, dict))
    return max(1, total_chars // 4)


class ResponseFormat(BaseModel):
    directive: str
    operators: List[str]

def get_openai_response(input_query, input_schema, input_data_sample, model="o3", max_tpm=5000000, message_history=[]):
    """
    The first LLM call. Generates a rewrite plan given the rewrite directives. 
    """

    user_message = f"""
    I have a set of operations used to process long documents, along with a list of possible rewrite directives aimed at improving accuracy and cost-effectiveness.
    Given a pipeline made up of these operations, recommend one specific rewrite directiven (specify by its name) that would improve both accuracy and cost-effectiveness and pecify which operators (specify by the names) in the pipeline the directive should be applied to.

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
    

    Input document schema with token statistics: {input_schema}
    Input data sample: {json.dumps(input_data_sample, indent=2)[:5000]}
    User query in YAML format using our operations: {input_query}
    """

    messages = message_history + [
        {"role": "system", "content": "You are an expert query optimization agent for document processing pipelines. Your role is to analyze user queries and apply rewrite directives to create more accurate and cost-effective execution plans. Your output must follow the structured output format."},
        {"role": "user", "content": user_message}
    ]
    
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
        reasoning_effort = "high",
        response_format=ResponseFormat
    )
    assistant_response = response.choices[0].message.content

    # Add user and assistant messages to message_history as dicts
    message_history.extend([
        {"role": "system", "content": "You are an expert query optimization agent for document processing pipelines. Your role is to analyze user queries and apply rewrite directives to create more accurate and cost-effective execution plans. Your output must follow the structured output format."},
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": assistant_response}
    ])
    return assistant_response

def update_yaml_operations(input_file_path, output_file_path, new_operations):
    """
    Load a YAML file, replace the operations section, and save to a new file.
    
    Args:
        input_file_path (str): Path to the original YAML file
        output_file_path (str): Path where the modified YAML will be saved
        new_operations (list): List of operation dictionaries to replace the original operations
    """
    # Load the original YAML file
    with open(input_file_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Replace the operations section
    config['operations'] = new_operations
    
    # Write the modified config to a new YAML file
    with open(output_file_path, 'w') as file:
        yaml.dump(config, file, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    print(f"Modified YAML saved to: {output_file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chat with OpenAI with per-model token rate limiting.")
    parser.add_argument("--model", type=str, default="o3", help="Model name")
    parser.add_argument("--max_tpm", type=int, default=5000000, help="Token per minute limit for the model")
    parser.add_argument("--yaml_path", type=str, help="Path to the YAML file")
    args = parser.parse_args()

    # Parse input yaml file to get the list of operations
    orig_config = load_config(args.yaml_path)
    orig_operators = orig_config["operations"]

    with open('/Users/lindseywei/Documents/DocETL-optimizer/reasoning-optimizer/CUAD_random_sample.json', 'r') as f:
        random_sample = json.load(f)

    with open(args.yaml_path, "r") as f:
        input_query = f.read()
    
    input_schema = load_input_doc(args.yaml_path)
    
    message_history = []
    reply = get_openai_response(input_query, input_schema, random_sample, model=args.model, max_tpm=args.max_tpm, message_history=message_history)    
    print("Agent:", reply)

    # Parse agent response 
    try:
        parsed = json.loads(reply)
        directive = parsed.get("directive")
        target_ops = parsed.get("operators")
    except Exception as e:
        print(f"Failed to parse agent response: {e}")

    new_ops_list = None
    if directive == "chaining":
        new_ops_list = ChainingDirective().instantiate(operators=orig_operators, target_ops=target_ops, agent_llm=args.model, message_history=message_history)
        print("new_ops_list:")
        print(new_ops_list)
        if new_ops_list is not None:
            op_names = [op.get("name") for op in new_ops_list if "name" in op]
            print("Operation names in new_ops_list:", op_names)
        orig_config["operations"] = new_ops_list

        # Update the pipeline steps to use the new operation names
        if "pipeline" in orig_config and "steps" in orig_config["pipeline"]:
            for step in orig_config["pipeline"]["steps"]:
                if "operations" in step and step["operations"] == target_ops:
                    step["operations"] = op_names

    elif directive == "gleaning":
        new_ops_list = GleaningDirective().instantiate(operators=orig_operators, target_ops=target_ops, agent_llm=args.model, message_history=message_history)
        print("new_ops_list:")
        print(new_ops_list)
        orig_config["operations"] = new_ops_list
        
    elif directive == "change model":
        new_ops_list = ChangeModelDirective().instantiate(operators=orig_operators, target_ops=target_ops, agent_llm=args.model, message_history=message_history)
        print("new_ops_list:")
        print(new_ops_list)
        orig_config["operations"] = new_ops_list


    # Dump yaml file
    output_file_path = f"{args.yaml_path}_agent_opt_v4.yaml"
    with open(output_file_path, 'w') as file:
        yaml.dump(orig_config, file, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    print(f"Modified YAML saved to: {output_file_path}")
  
    