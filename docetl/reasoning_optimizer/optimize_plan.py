from docetl.reasoning_optimizer.prompts import PromptLibrary
import litellm
import os
import threading
import time
import yaml  
from pydantic import BaseModel
from typing import Dict
from docetl.reasoning_optimizer.load_data import load_input_doc
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

class CalendarEvent(BaseModel):
    name: str
    date: str
    participants: list[str]

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

def get_openai_response(input_query, input_schema, model="gpt-4.1-nano", max_tpm=5000000):
    """
    Calls litellm.completion to generate a response to the user's message using Azure OpenAI.
    Enforces per-model rate limits if applicable.
    Assumes AZURE_API_KEY, AZURE_API_BASE, and AZURE_API_VERSION are set in environment variables.
    """

    user_message = f"""
    We have a set of LLM-powered operations for processing long documents and rewrite directives to improve accuracy and cost-effectiveness. Multiple rewrite directives can be combined iteratively or sequentially in optimization plans.
    Task: Apply rewrite directives to suggest 1 optimized query plan that improves both accuracy and cost-effectiveness over the original user query.
    Operations: 
    {PromptLibrary.map_operator(), PromptLibrary.reduce_operator(), PromptLibrary.resolve_operator(), 
    PromptLibrary.gather_operator(), PromptLibrary.filter_operator(), PromptLibrary.extract_operator()}
    
    Rewrite directives: 
    {PromptLibrary.document_chunking(), PromptLibrary.multi_level_agg(), PromptLibrary.chaining(), PromptLibrary.reordering()}

    Input document schema with token statistics: {input_schema}
    User query in YAML format using our operations: {input_query}
    """

    messages = [
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
        response_format = Rewritten_plan
    )
    return response.choices[0].message['content']


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chat with OpenAI with per-model token rate limiting.")
    parser.add_argument("--model", type=str, default="gpt-4.1-nano", help="Model name (default: gpt-4.1-nano)")
    parser.add_argument("--max_tpm", type=int, default=5000000, help="Token per minute limit for the model")
    parser.add_argument("--yaml_path", type=str, help="Path to the YAML file")
    args = parser.parse_args()

    with open(args.yaml_path, "r") as f:
        input_query = f.read()
    
    input_schema = load_input_doc(args.yaml_path)
    reply = get_openai_response(input_query, input_schema, model=args.model, max_tpm=args.max_tpm)
    print("AI:", reply)

