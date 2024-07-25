from contextlib import contextmanager
from litellm import completion


class LLMCallTracker:
    def __init__(self):
        self.last_prompt = None
        self.last_response = None

    @contextmanager
    def track_call(self):
        self.last_prompt = None
        self.last_response = None
        yield self

    def completion(self, *args, **kwargs):
        self.last_prompt = kwargs.get("messages", [])
        self.last_response = completion(*args, **kwargs)
        return self.last_response
