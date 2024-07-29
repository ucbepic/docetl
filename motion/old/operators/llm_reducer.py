from motion.operators.base_operator import Operator
from litellm import completion
from typing import List, Any, Tuple, Dict
from abc import ABC, abstractmethod
from motion.types import RK, RV, K, V


class LLMReducer(Operator, ABC):
    def __init__(self, model: str, **llm_kwargs):
        self.model = model
        self.llm_kwargs = llm_kwargs
        super().__init__()

    @abstractmethod
    def generate_prompt(self, key: K, values: List[V]) -> list:
        pass

    def process_response(self, response: Any, **prompt_kwargs) -> RV:
        return response.choices[0].message.content

    def execute(self, key: K, values: List[V]) -> Tuple[RV, Dict]:
        prompt = self.generate_prompt(key, values)
        response = completion(messages=prompt, model=self.model, **self.llm_kwargs)
        return self.process_response(response, key=key, values=values), {
            "prompt": prompt,
            "response": response,
        }

    def validate(self, key: K, input_values: List[V], output_value: RV) -> None:
        pass

    def correct(self, key: K, input_values: List[V], output_value: RV) -> Tuple[K, V]:
        return key, output_value
