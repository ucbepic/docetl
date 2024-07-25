from abc import ABC, abstractmethod
from typing import List, Any, Tuple, Set, Optional, Callable, Dict
from motion.types import ValidatorAction, K, V
from concurrent.futures import ThreadPoolExecutor
from motion.llm_profiler import LLMCallTracker


class Operator(ABC):
    on_fail: ValidatorAction = ValidatorAction.WARN

    def get_description(self) -> Optional[str]:
        return None  # Default implementation returns None


class LLMMapper(Operator, ABC):
    def __init__(self, model: str, **llm_kwargs):
        self.tracker = LLMCallTracker()
        self.model = model
        self.llm_kwargs = llm_kwargs

    @abstractmethod
    def generate_prompt(self, key: K, value: V) -> list:
        """Generate the prompt for the LLM call. Prompt should be a list of messages"""
        pass

    @abstractmethod
    def process_response(self, response: Any, **prompt_kwargs) -> Tuple[K, V]:
        """Process the LLM response and return the mapped key-value pair."""
        pass

    def map(self, key: K, value: V) -> Tuple[K, V]:
        with self.tracker.track_call():
            prompt = self.generate_prompt(key, value)
            response = self.tracker.completion(
                messages=prompt, model=self.model, **self.llm_kwargs
            )
            return self.process_response(response, key=key, value=value)

    def execute(self, key: K, value: V) -> Tuple[Tuple[K, V], Dict]:
        result = self.map(key, value)
        return result, {
            "prompt": self.tracker.last_prompt,
            "response": self.tracker.last_response,
        }

    def validate(
        self, input_key: K, input_value: V, output_key: K, output_value: V
    ) -> bool:
        return True  # Default implementation always returns True

    def correct(
        self, input_key: K, input_value: V, output_key: K, output_value: V
    ) -> Tuple[K, V]:
        return (
            output_key,
            output_value,
        )  # Default implementation returns original output


class LLMFlatMapper(LLMMapper, ABC):
    def __init__(self, model: str, **llm_kwargs):
        self.tracker = LLMCallTracker()
        self.model = model
        self.llm_kwargs = llm_kwargs

    @abstractmethod
    def generate_prompt(self, key: K, value: V) -> list:
        """Generate the prompt for the LLM call. Prompt should be a list of messages"""
        pass

    @abstractmethod
    def process_response(self, response: Any, **prompt_kwargs) -> List[Tuple[K, V]]:
        """Process the LLM response and return a list of mapped key-value pairs."""
        pass

    def map(self, key: K, value: V) -> List[Tuple[K, V]]:
        with self.tracker.track_call():
            prompt = self.generate_prompt(key, value)
            response = self.tracker.completion(
                messages=prompt, model=self.model, **self.llm_kwargs
            )
            return self.process_response(response, key=key, value=value)

    def execute(self, key: K, value: V) -> Tuple[List[Tuple[K, V]], Dict]:
        result = self.map(key, value)
        return result, {
            "prompt": self.tracker.last_prompt,
            "response": self.tracker.last_response,
        }

    def validate(self, key: K, value: V, mapped_kv_pairs: List[Tuple[K, V]]) -> bool:
        return True  # Default implementation always returns True

    def correct(
        self, key: K, value: V, mapped_kv_pairs: List[Tuple[K, V]]
    ) -> List[Tuple[K, V]]:
        return mapped_kv_pairs  # Default implementation returns original mapped pairs


class LLMParallelFlatMapper(Operator, ABC):
    @abstractmethod
    def get_mappers(self) -> List[LLMMapper]:
        """Return a list of LLMMapper instances to be executed in parallel."""
        pass

    def map(self, key: K, value: V) -> List[Tuple[K, V]]:
        mappers = self.get_mappers()

        with ThreadPoolExecutor() as executor:
            results = list(executor.map(lambda m: m.execute(key, value), mappers))

        return results

    def execute(self, key: K, value: V) -> Tuple[List[Tuple[K, V]], Dict]:
        results = self.map(key, value)

        # Get the return values, prompts, and responses from the mappers
        return_values = [result[0] for result in results]
        prompts = [result[1]["prompt"] for result in results]
        responses = [result[1]["response"] for result in results]
        return return_values, {"prompts": prompts, "responses": responses}

    def validate(self, key: K, value: V, mapped_kv_pairs: List[Tuple[K, V]]) -> bool:
        return True  # Default implementation always returns True

    def correct(
        self, key: K, value: V, mapped_kv_pairs: List[Tuple[K, V]]
    ) -> List[Tuple[K, V]]:
        return mapped_kv_pairs  # Default implementation returns original mapped pairs


class LLMReducer(Operator, ABC):
    def __init__(self, model: str, **llm_kwargs):
        self.tracker = LLMCallTracker()
        self.model = model
        self.llm_kwargs = llm_kwargs

    @abstractmethod
    def generate_prompt(self, key: K, values: List[V]) -> list:
        """Generate the prompt for the LLM call. Prompt should be a list of messages."""
        pass

    @abstractmethod
    def process_response(self, response: Any, **prompt_kwargs) -> V:
        """Process the LLM response and return the reduced value."""
        pass

    def reduce(self, key: K, values: List[V]) -> V:
        with self.tracker.track_call():
            prompt = self.generate_prompt(key, values)
            response = self.tracker.completion(
                messages=prompt, model=self.model, **self.llm_kwargs
            )
        return self.process_response(response, key=key, values=values)

    def execute(self, key: K, values: List[V]) -> Tuple[V, Dict]:
        result = self.reduce(key, values)
        return result, {
            "prompt": self.tracker.last_prompt,
            "response": self.tracker.last_response,
        }

    def validate(self, key: K, input_values: List[V], output_value: V) -> bool:
        return True  # Default implementation always returns True

    def correct(self, key: K, input_values: List[V], output_value: V) -> Tuple[K, V]:
        return (
            key,
            output_value,
        )  # Default implementation returns original key and output value


class KeyResolver(Operator, ABC):
    compute_embeddings: bool = False  # TODO: implement this

    def precheck(self, x: K, y: K) -> bool:
        return True  # Default implementation always returns True

    @abstractmethod
    def get_label_key(self, keys: Set[K]) -> K:
        # TODO: figure out how to allow embedding-based labels. maybe it's a totally separate class tbh
        pass

    def get_embedding(self, key: K) -> Optional[List[float]]:
        if self.compute_embeddings:
            raise NotImplementedError(
                "Embedding computation is not implemented. Set compute_embeddings to True if you want to use this method."
            )
        return None

    def validate(self, input_key: K, output_key: K) -> bool:
        return True

    def correct(self, input_key: K, output_key: K) -> K:
        return output_key


class PairwiseKeyResolver(KeyResolver, ABC):
    @abstractmethod
    def are_equal(self, x: K, y: K) -> bool:
        pass


class LLMPairwiseKeyResolver(PairwiseKeyResolver):
    def __init__(self, model: str, **llm_kwargs):
        self.tracker = LLMCallTracker()
        self.model = model
        self.llm_kwargs = llm_kwargs

    def generate_prompt(self, x: K, y: K) -> list:
        """Generate the prompt for the LLM call. Prompt should be a list of messages"""
        return [
            {
                "role": "system",
                "content": "You are a key resolver. Your task is to determine if two keys are equal.",
            },
            {
                "role": "user",
                "content": f"Are these two keys equal? Key 1: {x}, Key 2: {y}. Respond with 'Yes' or 'No'.",
            },
        ]

    def process_response(self, response: Any, **prompt_kwargs) -> bool:
        """Process the LLM response and return a boolean indicating whether the keys are equal."""
        return response.choices[0].message.content.strip().lower() == "yes"

    def are_equal(self, x: K, y: K) -> bool:
        with self.tracker.track_call():
            prompt = self.generate_prompt(x, y)
            response = self.tracker.completion(
                messages=prompt, model=self.model, **self.llm_kwargs
            )
            return self.process_response(response, x=x, y=y)

    def get_label_key(self, keys: Set[K]) -> K:
        # For simplicity, we'll just return the first key in the set
        # You might want to implement a more sophisticated method here
        return next(iter(keys))

    def execute(self, x: K, y: K) -> Tuple[bool, Dict]:
        result = self.are_equal(x, y)
        return result, {
            "prompt": self.tracker.last_prompt,
            "response": self.tracker.last_response,
        }

    def validate(self, input_key: K, output_key: K) -> bool:
        # You might want to implement a validation method here
        return True

    def correct(self, input_key: K, output_key: K) -> K:
        # You might want to implement a correction method here
        return output_key


class ListKeyResolver(KeyResolver, ABC):
    @abstractmethod
    def assign_key(self, key: K, label_keys: List[K]) -> K:
        pass


class LLMListKeyResolver(ListKeyResolver):
    def __init__(self, model: str, **llm_kwargs):
        self.tracker = LLMCallTracker()
        self.model = model
        self.llm_kwargs = llm_kwargs

    def generate_prompt(self, key: K, label_keys: List[K]) -> list:
        """Generate the prompt for the LLM call."""
        return [
            {
                "role": "system",
                "content": "You are a key resolver. Your task is to assign a key to a group based on existing label keys.",
            },
            {
                "role": "user",
                "content": f"Given the key '{key}' and the existing label keys {label_keys}, which label key should it be assigned to? If it doesn't match any existing label keys, respond with 'NEW'. Provide your answer as a single word or 'NEW'.",
            },
        ]

    def process_response(self, response: Any, **prompt_kwargs) -> K:
        """Process the LLM response and return the assigned key."""
        content = response.choices[0].message.content.strip()
        return content if content != "NEW" else prompt_kwargs["key"]

    def assign_key(self, key: K, label_keys: List[K]) -> K:
        with self.tracker.track_call():
            prompt = self.generate_prompt(key, label_keys)
            response = self.tracker.completion(
                messages=prompt, model=self.model, **self.llm_kwargs
            )
            return self.process_response(response, key=key, label_keys=label_keys)

    def execute(self, key: K, label_keys: List[K]) -> Tuple[K, Dict]:
        result = self.assign_key(key, label_keys)
        return result, {
            "prompt": self.tracker.last_prompt,
            "response": self.tracker.last_response,
        }

    def validate(self, input_key: K, output_key: K) -> bool:
        # You might want to implement a validation method here
        return True

    def correct(self, input_key: K, output_key: K) -> K:
        # You might want to implement a correction method here
        return output_key


class LLMFilterer(Operator, ABC):
    def __init__(self, model: str, **llm_kwargs):
        self.tracker = LLMCallTracker()
        self.model = model
        self.llm_kwargs = llm_kwargs

    @abstractmethod
    def generate_prompt(self, key: K, value: V) -> list:
        """Generate the prompt for the LLM call. Prompt should be a list of messages"""
        pass

    @abstractmethod
    def process_response(self, response: Any, **prompt_kwargs) -> bool:
        """Process the LLM response and return a boolean indicating whether to keep the item."""
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
        return True  # Default implementation always returns True

    def correct(self, key: K, value: V, output: bool) -> bool:
        return output  # Default implementation returns original output
