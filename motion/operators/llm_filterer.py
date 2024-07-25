from motion.operators.base_operator import Operator
from motion.llm_profiler import LLMCallTracker
from typing import List, Any, Tuple, Dict
from abc import ABC, abstractmethod
from motion.types import RK, RV, K, V


class LLMFilterer(Operator, ABC):
    def __init__(self, model: str, **llm_kwargs):
        self.tracker = LLMCallTracker()
        self.model = model
        self.llm_kwargs = llm_kwargs

    @abstractmethod
    def generate_prompt(self, key: K, value: V) -> list:
        pass

    @abstractmethod
    def process_response(self, response: Any, **prompt_kwargs) -> bool:
        pass

    def filter(self, key: K, value: V) -> bool:
        with self.tracker.track_call():
            prompt = self.generate_prompt(key, value)
            response = self.tracker.completion(
                messages=prompt, model=self.model, **self.llm_kwargs
            )
            return self.process_response(response, key=key, value=value)

    def execute(self, key: K, value: V) -> Tuple[bool, Dict]:
        result = self.filter(key, value)
        return result, {
            "prompt": self.tracker.last_prompt,
            "response": self.tracker.last_response,
        }

    def validate(self, key: K, value: V, output: bool) -> bool:
        return True

    def correct(self, key: K, value: V, output: bool) -> bool:
        return output
