import pytest
from typing import List, Tuple, Any
from motion.dataset import Dataset
from motion.operators import (
    LLMMapper,
    LLMReducer,
    LLMPairwiseKeyResolver,
    LLMFilterer,
    LLMFlatMapper,
    LLMParallelFlatMapper,
    Splitter,
)

MODEL = "gpt-4o-mini"

from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()


# Test LLMMapper
class TestLLMMapper(LLMMapper):
    __test__ = False

    def generate_prompt(self, key: str, value: int) -> list:
        return [
            {
                "role": "system",
                "content": "You are a helpful assistant that doubles numbers.",
            },
            {
                "role": "user",
                "content": f"Please double the number: {value}. Return only the number.",
            },
        ]

    def process_response(self, response: Any, **prompt_kwargs) -> Tuple[str, int]:
        doubled_value = int(response.choices[0].message.content.strip())
        return (prompt_kwargs["key"], doubled_value)


def test_llm_mapper():
    data = [("a", 2), ("b", 3)]
    dataset = Dataset(data)
    result = dataset.map(TestLLMMapper(model=MODEL)).execute()
    assert result == [("a", 4), ("b", 6)]


# Test LLMFlatMapper
class TestLLMFlatMapper(LLMFlatMapper):
    __test__ = False

    def generate_prompt(self, key: str, value: int) -> list:
        return [
            {
                "role": "system",
                "content": "You are a helpful assistant that generates two numbers: the original and its successor.",
            },
            {
                "role": "user",
                "content": f"Please provide the original number and its successor for: {value}. Return your answer only as the two numbers, comma-separated.",
            },
        ]

    def process_response(self, response: Any, **prompt_kwargs) -> List[Tuple[str, int]]:
        numbers = [
            int(num.strip()) for num in response.choices[0].message.content.split(",")
        ]
        return [(prompt_kwargs["key"], numbers[0]), (prompt_kwargs["key"], numbers[1])]


def test_llm_flatmapper():
    data = [("a", 2)]
    dataset = Dataset(data)
    result = dataset.flat_map(TestLLMFlatMapper(model=MODEL)).execute()
    assert result == [("a", 2), ("a", 3)]


# Test LLMReducer
class TestLLMReducer(LLMReducer):
    __test__ = False

    def generate_prompt(self, key: str, values: List[int]) -> list:
        return [
            {
                "role": "system",
                "content": "You are a helpful assistant that sums numbers.",
            },
            {
                "role": "user",
                "content": f"Please sum these numbers: {', '.join(map(str, values))}\nOnly return the sum of the numbers.",
            },
        ]

    def process_response(self, response: Any, **prompt_kwargs) -> int:
        return int(response.choices[0].message.content.strip())


def test_llm_reducer():
    data = [("a", 2), ("a", 5)]
    dataset = Dataset(data)
    result = dataset.reduce(TestLLMReducer(model=MODEL)).execute()
    assert result == [("a", 7)]


# Test LLMPairwiseKeyResolver
class TestLLMPairwiseKeyResolver(LLMPairwiseKeyResolver):
    __test__ = False

    def generate_prompt(self, x: int, y: int) -> list:
        return [
            {
                "role": "system",
                "content": "You are a helpful assistant that determines if two numbers are close (within 1 of each other).",
            },
            {
                "role": "user",
                "content": f"Are these two numbers close (within 1 of each other)? {x} and {y}. Answer only Yes or No.",
            },
        ]

    def process_response(self, response: Any, **prompt_kwargs) -> bool:
        return "yes" in response.choices[0].message.content.strip().lower()


def test_llm_pairwise_key_resolver():
    __test__ = False

    resolver = TestLLMPairwiseKeyResolver(model=MODEL)
    data = [(1, "a"), (2, "b"), (5, "c")]
    dataset = Dataset(data)
    result = dataset.resolve_keys(resolver).execute()
    assert set(result) == {(1, "a"), (1, "b"), (5, "c")}


# Test LLMFilterer
class TestLLMFilterer(LLMFilterer):
    __test__ = False

    def generate_prompt(self, key: str, value: int) -> list:
        return [
            {
                "role": "system",
                "content": "You are a helpful assistant that determines if a number is even.",
            },
            {
                "role": "user",
                "content": f"Is this number even: {value}? Answer Yes or No.",
            },
        ]

    def process_response(self, response: Any, **prompt_kwargs) -> bool:
        return "yes" in response.choices[0].message.content.strip().lower()


def test_llm_filterer():
    data = [("a", 1), ("b", 2)]
    dataset = Dataset(data)
    result = dataset.filter(TestLLMFilterer(model=MODEL)).execute()
    assert result == [("b", 2)]


# Test chaining operations
def test_chained_llm_operations():
    data = [("a", 1), ("a", 2), ("c", 4)]
    dataset = Dataset(data)
    result = (
        dataset.filter(TestLLMFilterer(model=MODEL))
        .map(TestLLMMapper(model=MODEL))
        .reduce(TestLLMReducer(model=MODEL))
        .execute()
    )
    assert set(result) == {("a", 4), ("c", 8)}


# Test LLMParallelFlatMapper
class TestLLMParallelFlatMapper(LLMParallelFlatMapper):
    __test__ = False

    def get_mappers(self) -> List[LLMMapper]:
        return [
            TestLLMMapper(model=MODEL),
            TestLLMMapper(model=MODEL),
            TestLLMMapper(model=MODEL),
        ]


def test_llm_parallel_flat_mapper():
    data = [("a", 1)]
    dataset = Dataset(data)
    result = dataset.flat_map(TestLLMParallelFlatMapper()).execute()
    assert result == [("a", 2), ("a", 2), ("a", 2)]


# Test Splitter
class TestSplitter(Splitter):
    __test__ = False

    def split(self, key: str, value: Any) -> List[Tuple[str, Any]]:
        if isinstance(value, str):
            return [(key, char) for char in value]
        return [(key, value)]


def test_splitter():
    data = [("a", "hello"), ("b", 123)]
    dataset = Dataset(data)
    result = dataset.split(TestSplitter()).execute()
    assert result == [
        ("a", "h"),
        ("a", "e"),
        ("a", "l"),
        ("a", "l"),
        ("a", "o"),
        ("b", 123),
    ]


if __name__ == "__main__":
    test_splitter()
