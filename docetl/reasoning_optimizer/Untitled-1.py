import litellm
import os
import threading
import time
import yaml  
from pydantic import BaseModel

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

def get_openai_response(user_message, model="gpt-4.1-nano"):
    """
    Calls litellm.completion to generate a response to the user's message using Azure OpenAI.
    Enforces GPT-4.1 rate limits if applicable.
    Assumes AZURE_API_KEY, AZURE_API_BASE, and AZURE_API_VERSION are set in environment variables.
    """
    messages = [
        {"role": "system", "content": "Extract the event information."},
        {"role": "user", "content": user_message}
    ]
    
    # Enforce rate limit for GPT-4.1 models
    if "gpt-4.1" in model:
        tokens = count_tokens(messages)
        gpt41_rate_limiter.wait_for_slot(tokens)

    response = litellm.completion(
        model=model,
        messages=messages,
        api_key=os.environ.get("AZURE_API_KEY"),
        api_base=os.environ.get("AZURE_API_BASE"),
        api_version=os.environ.get("AZURE_API_VERSION"),
        azure=True,
        response_format = CalendarEvent
    )
    return response.choices[0].message['content']




if __name__ == "__main__":
    user_input = input("You: ")
    reply = get_openai_response(user_input, model="gpt-4.1-nano")
    print("AI:", reply)




