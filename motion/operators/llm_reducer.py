from motion.operators.base_operator import Operator
from motion.llm_profiler import LLMCallTracker
from typing import List, Any, Tuple, Dict
from abc import ABC, abstractmethod
from motion.types import RK, RV, K, V


class LLMReducer(Operator, ABC):
    def __init__(self, model: str, **llm_kwargs):
        self.tracker = LLMCallTracker()
        self.model = model
        self.llm_kwargs = llm_kwargs

    @abstractmethod
    def generate_prompt(self, key: K, values: List[V]) -> list:
        pass

    @abstractmethod
    def process_response(self, response: Any, **prompt_kwargs) -> RV:
        pass

    def reduce(self, key: K, values: List[V]) -> RV:
        with self.tracker.track_call():
            prompt = self.generate_prompt(key, values)
            response = self.tracker.completion(
                messages=prompt, model=self.model, **self.llm_kwargs
            )
        return self.process_response(response, key=key, values=values)

    def execute(self, key: K, values: List[V]) -> Tuple[RV, Dict]:
        result = self.reduce(key, values)
        return result, {
            "prompt": self.tracker.last_prompt,
            "response": self.tracker.last_response,
        }

    def validate(self, key: K, input_values: List[V], output_value: RV) -> bool:
        return True

    def correct(self, key: K, input_values: List[V], output_value: RV) -> Tuple[K, V]:
        return key, output_value
