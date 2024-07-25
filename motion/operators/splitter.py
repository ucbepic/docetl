from motion.operators.base_operator import Operator
from typing import List, Tuple, Dict
from abc import ABC, abstractmethod
import tiktoken
from motion.types import RK, RV, K, V


class Splitter(Operator, ABC):
    @abstractmethod
    def split(self, key: K, value: V) -> List[Tuple[RK, RV]]:
        pass

    def execute(self, key: K, value: V) -> Tuple[List[Tuple[RK, RV]], Dict]:
        result = self.split(key, value)
        return result, {}

    def validate(
        self, input_key: K, input_value: V, output_pairs: List[Tuple[RK, RV]]
    ) -> bool:
        return True

    def correct(
        self, input_key: K, input_value: V, output_pairs: List[Tuple[RK, RV]]
    ) -> List[Tuple[RK, RV]]:
        return output_pairs


class ChunkSplitter(Splitter):
    def __init__(self, chunk_size: int, overlap_size: int = 0, model: str = "gpt-4o"):
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.model = model
        self.encoder = tiktoken.encoding_for_model(model)
        super().__init__()

    def split(self, key: K, value: V) -> List[Tuple[RK, RV]]:
        if not isinstance(value, str):
            raise ValueError("Value must be a string for tiktoken tokenization")

        tokens = self.encoder.encode(value)
        chunks = []
        start = 0
        while start < len(tokens):
            end = start + self.chunk_size
            chunk_tokens = tokens[start:end]
            chunk = self.encoder.decode(chunk_tokens)
            chunk_key = f"{key}_{start}_{end}"
            chunks.append((key, (chunk_key, chunk)))
            start = end - self.overlap_size

        return chunks
