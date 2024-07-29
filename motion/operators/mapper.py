from motion.types import RK, RV, K, V
from motion.operators.base_operator import Operator
from abc import abstractmethod, ABC
from typing import List, Tuple, Dict, Any
from uuid import uuid4


class Mapper(Operator, ABC):
    @abstractmethod
    def map(self, key: K, value: V) -> Tuple[RK, RV]:
        pass

    def execute(self, key: K, value: V) -> Tuple[Tuple[RK, RV], Dict[str, Any]]:
        return self.map(key, value), {}


class AddUniqueIDToKey(Mapper):
    def map(self, key: K, value: V) -> Tuple[RK, RV]:
        return (key, uuid4()), value


class RemoveUniqueIDFromKey(Mapper):
    def map(self, key: K, value: V) -> Tuple[RK, RV]:
        return (key[0], value)
