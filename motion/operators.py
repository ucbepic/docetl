from abc import ABC, abstractmethod
from typing import List, Any, Tuple, Set, Optional, Callable
from motion.types import ValidatorAction, K, V
from concurrent.futures import ThreadPoolExecutor


class Operator(ABC):
    on_fail: ValidatorAction = ValidatorAction.WARN

    def get_description(self) -> Optional[str]:
        return None  # Default implementation returns None


class Mapper(Operator, ABC):
    @abstractmethod
    def map(self, key: Any, value: Any) -> Tuple[K, V]:
        pass

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


class FlatMapper(Mapper, ABC):
    @abstractmethod
    def map(self, key: Any, value: Any) -> List[Tuple[K, V]]:
        pass

    def validate(self, key: K, value: V, mapped_kv_pairs: List[Tuple[K, V]]) -> bool:
        return True  # Default implementation always returns True

    def correct(
        self, key: K, value: V, mapped_kv_pairs: List[Tuple[K, V]]
    ) -> List[Tuple[K, V]]:
        return mapped_kv_pairs


# Create a type of FlatMapper that employs many map calls in parallel
class ParallelFlatMapper(FlatMapper):
    @abstractmethod
    def get_mappers(self) -> List[Callable[[K, V], Tuple[K, V]]]:
        pass

    def map(self, key: K, value: V) -> List[Tuple[K, V]]:
        mappers = self.get_mappers()
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(lambda f: f(key, value), mappers))
        return results


class Reducer(Operator, ABC):
    @abstractmethod
    def reduce(self, key: K, values: List[V]) -> V:
        pass

    def validate(self, key: K, input_values: List[V], output_value: V) -> bool:
        return True  # Default implementation always returns True

    def correct(self, key: K, input_values: List[V], output_value: V) -> Tuple[K, V]:
        return (
            key,
            output_value,
        )  # Default implementation returns original key and output value


class KeyResolver(Operator, ABC):
    compute_embeddings: bool = False  # TODO: implement this
    _use_are_equal: bool = False

    def __init__(self):
        self._set_implementation()

    def _set_implementation(self):
        if (
            hasattr(self, "are_equal")
            and self.are_equal.__func__ is not KeyResolver.are_equal
        ):
            self._use_are_equal = True
        elif (
            not hasattr(self, "assign_key")
            or self.assign_key.__func__ is KeyResolver.assign_key
        ):
            raise NotImplementedError("Implement either are_equal or assign_key")

    def precheck(self, x: K, y: K) -> bool:
        return True  # Default implementation always returns True

    def are_equal(self, x: K, y: K) -> bool:
        raise NotImplementedError("Implement are_equal method")

    def assign_key(self, key: K, label_keys: List[K]) -> K:
        raise NotImplementedError("Implement assign_key method")

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


class Filterer(Operator, ABC):
    @abstractmethod
    def filter(self, key: K, value: V) -> bool:
        pass

    def validate(self, key: K, value: V, output: bool) -> bool:
        return True  # Default implementation always returns True

    def correct(self, key: K, value: V, output: bool) -> bool:
        return output  # Default implementation returns original output
