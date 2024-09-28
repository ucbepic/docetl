# ruff: noqa: F811

import pytest
from docetl.operations.map import ParallelMapOperation
from dotenv import load_dotenv
from typing import Dict, Any, List, Tuple
from .conftest import (
    response_lookup as response_lookup,
    parallel_map_config_with_batching as parallel_map_config_with_batching,
    parallel_map_config as parallel_map_config,
    parallel_map_sample_data as parallel_map_sample_data,
    map_sample_data_large as map_sample_data_large,
    default_model as default_model,
    max_threads as max_threads,
    map_config_with_tools as map_config_with_tools,
)

load_dotenv()


class TestParallelMapOperation(ParallelMapOperation):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(
            *args,
            **kwargs,
        )
        self.response_lookup = self.config.get("response_lookup", {}) or kwargs.get(
            "response_lookup", {}
        )

    def _process_map_batch(
        self, batch: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], float]:
        results = []
        total_cost = 0.0
        cost_per_prompt = 0.01  # Assign a mock cost per prompt

        for item in batch:
            item_result = item.copy()
            input_text = item["text"]  # Use item["text"] directly

            # Retrieve the response for the current input text
            response = self.response_lookup.get(input_text, {})

            for prompt_config in self.config["prompts"]:
                if response:
                    # Extract only the keys relevant to this prompt
                    output = {k: response[k] for k in prompt_config["output_keys"]}
                    item_result.update(output)

                total_cost += cost_per_prompt

            results.append(item_result)
        return results, total_cost


@pytest.fixture
def test_parallel_map_operation_instance(
    parallel_map_config_with_batching,
    default_model,
    max_threads,
    response_lookup,
):
    # Ensure 'batch_size' and 'clustering_method' are part of the config
    parallel_map_config_with_batching["batch_size"] = 1
    parallel_map_config_with_batching["clustering_method"] = "sem_cluster"

    return TestParallelMapOperation(
        config=parallel_map_config_with_batching,
        default_model=default_model,
        max_threads=max_threads,
        response_lookup=response_lookup,
    )


def test_parallel_map_operation_clustering_methods(
    test_parallel_map_operation_instance,
    map_sample_data_large,
    response_lookup,
):
    results, cost = test_parallel_map_operation_instance.execute(map_sample_data_large)

    assert len(results) == len(map_sample_data_large)
    assert cost > 0

    for result, item in zip(results, map_sample_data_large):
        expected = response_lookup[item["text"]]
        assert result["sentiment"] == expected["sentiment"]
        assert result["word_count"] == expected["word_count"]


def test_parallel_map_operation_accuracy_preservation(
    test_parallel_map_operation_instance,
    map_sample_data_large,
    response_lookup,
):
    results, cost = test_parallel_map_operation_instance.execute(map_sample_data_large)

    actual_sentiments = [result["sentiment"] for result in results]
    expected_sentiments = [
        response_lookup[item["text"]]["sentiment"] for item in map_sample_data_large
    ]

    actual_word_counts = [result["word_count"] for result in results]
    expected_word_counts = [
        response_lookup[item["text"]]["word_count"] for item in map_sample_data_large
    ]

    assert actual_sentiments == expected_sentiments
    assert actual_word_counts == expected_word_counts
    assert len(results) == len(map_sample_data_large)
    assert cost > 0


def test_parallel_map_operation_with_batching(
    parallel_map_config_with_batching,
    default_model,
    max_threads,
    parallel_map_sample_data,
):
    operation = ParallelMapOperation(
        parallel_map_config_with_batching, default_model, max_threads
    )

    results, cost = operation.execute(parallel_map_sample_data)

    assert len(results) == len(parallel_map_sample_data)
    assert cost > 0

    for result in results:
        assert "sentiment" in result
        assert "word_count" in result
        assert result["sentiment"] in ["positive", "negative", "neutral"]
        assert isinstance(result["word_count"], int)


def test_parallel_map_operation(
    parallel_map_config,
    default_model,
    max_threads,
    parallel_map_sample_data,
):
    operation = ParallelMapOperation(parallel_map_config, default_model, max_threads)
    results, cost = operation.execute(parallel_map_sample_data)

    assert len(results) == len(parallel_map_sample_data)
    assert all("sentiment" in result for result in results)
    assert all("word_count" in result for result in results)
    assert all(
        result["sentiment"] in ["positive", "negative", "neutral"] for result in results
    )
    assert all(isinstance(result["word_count"], int) for result in results)
    assert cost > 0


def test_parallel_map_operation_empty_input(
    parallel_map_config, default_model, max_threads
):
    operation = ParallelMapOperation(parallel_map_config, default_model, max_threads)
    results, cost = operation.execute([])

    assert len(results) == 0
    assert cost == 0


def test_parallel_map_operation_with_empty_input(
    parallel_map_config_with_batching, default_model, max_threads
):
    operation = ParallelMapOperation(
        parallel_map_config_with_batching, default_model, max_threads
    )
    results, cost = operation.execute([])

    assert len(results) == 0
    assert cost == 0


def test_parallel_map_operation_with_large_batch_size(
    parallel_map_config_with_batching,
    default_model,
    max_threads,
    parallel_map_sample_data,
):
    parallel_map_config_with_batching["batch_size"] = (
        5  # Set batch size larger than data
    )
    operation = ParallelMapOperation(
        parallel_map_config_with_batching, default_model, max_threads
    )
    results, cost = operation.execute(parallel_map_sample_data)

    assert len(results) == len(parallel_map_sample_data)
    assert cost > 0


def test_parallel_map_operation_with_different_clustering(
    parallel_map_config_with_batching,
    default_model,
    max_threads,
    parallel_map_sample_data,
):
    # Test with a different clustering method
    parallel_map_config_with_batching["clustering_method"] = (
        "random"  # Change clustering method
    )
    operation = ParallelMapOperation(
        parallel_map_config_with_batching, default_model, max_threads
    )
    results, cost = operation.execute(parallel_map_sample_data)

    assert len(results) == len(parallel_map_sample_data)
    assert cost > 0