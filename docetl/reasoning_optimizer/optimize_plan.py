import argparse
import json
import os
import threading
import time
from typing import Dict

import litellm
from pydantic import BaseModel

from docetl.reasoning_optimizer.load_data import load_input_doc
from docetl.reasoning_optimizer.prompts import PromptLibrary

# Global dictionary of rate limiters per model
model_rate_limiters: Dict[str, "TokenRateLimiter"] = {}


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


class Step(BaseModel):
    operation: str
    description: str
    parameter: list[str]


class Rewritten_plan(BaseModel):
    plan_name: str
    rewrites_used: list[str]
    plan_description: str
    stepwise_pipeline: list[Step]
    reason: str


class ResponseFormat(BaseModel):
    yaml_file: str
    plan: Rewritten_plan


def get_openai_response(
    input_query, input_schema, input_data_sample, model="o3", max_tpm=5000000
):
    """
    The first LLM call. Generates a rewrite plan given the rewrite directives.
    """

    user_message = f"""
    I need one optimized query plan for a given user query on document processing.
    I have a set of LLM-powered operations for processing long documents and rewrite directives to improve accuracy and cost-effectiveness. Multiple rewrite directives can be combined iteratively or sequentially in optimization plans.
    You can choose to use the additional techniques Metadata Extraction and Header Extraction along with the rewrite directives.
    You also need to carefully choose the parameters for each operation as the parameters can significantly impact the accuracy and cost-effectiveness of the plan.

    Return a YAML file with the optimized query plan. Each operation in the rewritten plan should have the required parameters specified. The parameter required for each operation is specified in the operation description.

    Make sure you apply existing rewrite directives and use only existing operations provided below. Ensure that your YAML file is valid.

    Pipeline:
    Pipelines in DocETL are the core structures that define the flow of data processing. A pipeline consists of five main components: \n
    - Default Model: The language model to use for the pipeline. Limit your choice of model to gpt-4.1-nano, gpt-4o-mini, gpt-4o, gpt-4.1 \n
    - System Prompts: A description of your dataset and the "persona" you'd like the LLM to adopt when analyzing your data. \n
    - Datasets: The input data sources for your pipeline. \n
    - Operators: The processing steps that transform your data. \n
    - Pipeline Specification: The sequence of steps and the output configuration. \n

    Operation descriptions:
    Operators form the building blocks of data processing pipelines. All operators share the following common attributes: \n
    - name: A unique identifier for the operator. \n
    - type: Specifies the type of operation (e.g., "map", "reduce", "filter"). \n
    LLM-based operators (including Map, Reduce, Resolve, Filter, and Extract) have additional attributes:\n
    - prompt: A Jinja2 template that defines the instruction for the language model. \n
    - output: Specifies the schema for the output from the LLM call. \n
    - model (optional): Allows specifying a different model from the pipeline default. Limit your choice of model to gpt-4.1-nano, gpt-4o-mini, gpt-4o, gpt-4.1\n
    Additional parameters required by each operation are specified in the operation's description.
    {PromptLibrary.map_operator()}\n
    {PromptLibrary.reduce_operator()}\n
    {PromptLibrary.resolve_operator()}\n
    {PromptLibrary.split_operator()}\n
    {PromptLibrary.gather_operator()}\n
    {PromptLibrary.filter_operator()}\n
    {PromptLibrary.extract_operator()}\n

    Rewrite directives:
    {PromptLibrary.document_chunking(), PromptLibrary.multi_level_agg(), PromptLibrary.chaining(), PromptLibrary.reordering()}

    Additional techniques:
    {PromptLibrary.metadata_extraction(), PromptLibrary.header_extraction()}

    Input document schema with token statistics: {input_schema}
    Input data sample: {json.dumps(input_data_sample, indent=2)[:5000]}
    User query in YAML format using our operations: {input_query}
    """

    messages = [
        {
            "role": "system",
            "content": "You are an expert query optimization agent for document processing pipelines. Your role is to analyze user queries and apply rewrite directives to create more accurate and cost-effective execution plans. Your output must follow the structured output format.",
        },
        {"role": "user", "content": user_message},
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
        reasoning_effort="high",
        response_format=ResponseFormat,
    )
    return response.choices[0].message["content"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Chat with OpenAI with per-model token rate limiting."
    )
    parser.add_argument("--model", type=str, default="o3", help="Model name")
    parser.add_argument(
        "--max_tpm",
        type=int,
        default=5000000,
        help="Token per minute limit for the model",
    )
    parser.add_argument("--yaml_path", type=str, help="Path to the YAML file")
    args = parser.parse_args()

    data_dir = os.environ.get("EXPERIMENT_DATA_DIR", "./data/")
    sample_data_path = os.path.join(data_dir, "CUAD_random_sample.json")
    with open(sample_data_path, "r") as f:
        random_sample = json.load(f)

    with open(args.yaml_path, "r") as f:
        input_query = f.read()

    input_schema = load_input_doc(args.yaml_path)
    reply = get_openai_response(
        input_query, input_schema, random_sample, model=args.model, max_tpm=args.max_tpm
    )
    with open("o3_CUAD_opt_plan_v3.yaml", "w", encoding="utf-8") as f:
        f.write(reply.get)

    print("AI:", reply)
