from motion.operators.base_operator import Operator
from litellm import completion
from typing import List, Any, Tuple, Dict
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from motion.types import RK, RV, K, V


class LLMMapper(Operator, ABC):
    def __init__(self, model: str, **llm_kwargs):
        self.model = model
        self.llm_kwargs = llm_kwargs

    @abstractmethod
    def generate_prompt(self, key: K, value: V) -> list:
        pass

    @abstractmethod
    def process_response(self, response: Any, **prompt_kwargs) -> Tuple[RK, RV]:
        pass

    def execute(self, key: K, value: V) -> Tuple[Tuple[RK, RV], Dict]:
        prompt = self.generate_prompt(key, value)
        response = completion(messages=prompt, model=self.model, **self.llm_kwargs)
        return self.process_response(response, key=key, value=value), {
            "prompt": prompt,
            "response": response,
        }

    def validate(
        self, input_key: K, input_value: V, output_key: RK, output_value: RV
    ) -> bool:
        return True

    def correct(
        self, input_key: K, input_value: V, output_key: RK, output_value: RV
    ) -> Tuple[K, V]:
        return output_key, output_value


class LLMFlatMapper(Operator, ABC):
    def __init__(self, model: str, **llm_kwargs):
        self.model = model
        self.llm_kwargs = llm_kwargs

    @abstractmethod
    def generate_prompt(self, key: K, value: V) -> list:
        pass

    @abstractmethod
    def process_response(self, response: Any, **prompt_kwargs) -> List[Tuple[RK, RV]]:
        pass

    def execute(self, key: K, value: V) -> Tuple[List[Tuple[RK, RV]], Dict]:
        prompt = self.generate_prompt(key, value)
        response = completion(messages=prompt, model=self.model, **self.llm_kwargs)
        return self.process_response(response, key=key, value=value), {
            "prompt": prompt,
            "response": response,
        }

    def validate(self, key: K, value: V, mapped_kv_pairs: List[Tuple[RK, RV]]) -> bool:
        return True

    def correct(
        self, key: K, value: V, mapped_kv_pairs: List[Tuple[RK, RV]]
    ) -> List[Tuple[RK, RV]]:
        return mapped_kv_pairs


class LLMParallelFlatMapper(Operator, ABC):
    @abstractmethod
    def get_mappers(self) -> List[LLMMapper]:
        pass

    def map(self, key: K, value: V) -> List[Tuple[RK, RV]]:
        mappers = self.get_mappers()
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(lambda m: m.execute(key, value), mappers))
        return results

    def execute(self, key: K, value: V) -> Tuple[List[Tuple[RK, RV]], Dict]:
        results = self.map(key, value)
        return_values = [result[0] for result in results]
        prompts = [result[1]["prompt"] for result in results]
        responses = [result[1]["response"] for result in results]
        return return_values, {"prompts": prompts, "responses": responses}

    def validate(self, key: K, value: V, mapped_kv_pairs: List[Tuple[RK, RV]]) -> bool:
        return True

    def correct(
        self, key: K, value: V, mapped_kv_pairs: List[Tuple[RK, RV]]
    ) -> List[Tuple[RK, RV]]:
        return mapped_kv_pairs
