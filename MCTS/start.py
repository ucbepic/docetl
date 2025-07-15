import litellm
import os
import threading
import time
import yaml  
import json
import argparse
from pydantic import BaseModel
from docetl.reasoning_optimizer.load_data import load_input_doc

# Global rate limiter 
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

gpt41_rate_limiter = TokenRateLimiter(5_000_000) # 5,000,000 tokens per minute for GPT-4.1 family of models

def count_tokens(messages):
    # messages should be a list of dicts, each with a "content" key
    total_chars = sum(len(m.get("content", "")) for m in messages if isinstance(m, dict))
    return max(1, total_chars // 4)

class CalendarEvent(BaseModel):
    name: str
    date: str
    participants: list[str]

def get_openai_response(user_message, model="o3"):
    """
    Calls litellm.completion to generate a response to the user's message using Azure OpenAI.
    Assumes AZURE_API_KEY, AZURE_API_BASE, and AZURE_API_VERSION are set in environment variables.
    """
    messages = [
        {"role": "system", "content": "You are an expert query optimization agent for document processing pipelines. Your role is to analyze user queries and apply rewrite directives to create more accurate and cost-effective execution plans."},
        {"role": "user", "content": user_message}
    ]

    response = litellm.completion(
        model=model,
        messages=messages,
        api_key=os.environ.get("AZURE_API_KEY"),
        api_base=os.environ.get("AZURE_API_BASE"),
        api_version=os.environ.get("AZURE_API_VERSION"),
        azure=True,
        reasoning_effort="high",
    )
    # response.json
    # response.complete_response
    # print(response.model_dump_json(indent=2))
    # response.choices[0].message['content']
    print(response)
    return response


def save_response_content_only(response, filename):
    try:
        content = response.choices[0].message.content
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Response content saved to: {filename}")
        return filename
    except (AttributeError, IndexError) as e:
        print(f"Error extracting content: {e}")
        return None

def save_response_detail(response, filename):
    try:
        content = response.complete_response
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Complete response saved to: {filename}")
        return filename
    except (AttributeError, IndexError) as e:
        print(f"Error extracting content: {e}")
        return None


def save_response_json(response, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("=== LLM Response Details ===\n")
        f.write(f"Model: {response.model if hasattr(response, 'model') else 'Unknown'}\n")
        f.write("\n=== Full JSON Response ===\n")
        json.dump(response.json, f, indent=2, ensure_ascii=False)
    
    print(f"Detailed response saved to: {filename}")
    return filename




if __name__ == "__main__":
    yaml_path = "/Users/lindseywei/Documents/DocETL-optimizer/reasoning-optimizer/CUAD-map.yaml"
    input_schema = load_input_doc(yaml_path)
    print(input_schema)
    with open('/Users/lindseywei/Documents/DocETL-optimizer/reasoning-optimizer/CUAD_random_sample.json', 'r') as f:
        random_sample = json.load(f)

    with open(yaml_path, "r") as f:
        input_query = f.read()
    user_input = f"""
    Learn DocETL's operations and rewrite directives in this document: https://ucbepic.github.io/docetl/concepts/operators/
    Task: Apply rewrite directives to suggest 1 optimized query plan that improves both accuracy and cost-effectiveness over the original user query. Write the optimized plan in YAML.
    Make sure you apply existing rewrie directives and use only existing operations in DocETL. 
    Input document schema with token statistics: {input_schema}
    Input data sample: {json.dumps(random_sample, indent=2)[:5000]}
    User query in YAML format using our operations: {input_query}
    """
    res = get_openai_response(user_input, model="o3")
    save_response_content_only(res, "res_content.txt")
    #save_response_detail(res, "res_detail.txt")






