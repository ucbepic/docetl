# ruff: noqa: F811

import pytest
from docetl.operations.map import ParallelMapOperation
from dotenv import load_dotenv
from typing import Dict, Any, List, Tuple
from tests.conftest import (
    parallel_map_config as parallel_map_config,
    parallel_map_sample_data as parallel_map_sample_data,
    default_model as default_model,
    max_threads as max_threads,
    api_wrapper as api_wrapper,
)

load_dotenv()


def test_parallel_map_operation(
    parallel_map_config,
    default_model,
    max_threads,
    parallel_map_sample_data,
    api_wrapper,
):
    parallel_map_config["bypass_cache"] = True
    operation = ParallelMapOperation(
        api_wrapper, parallel_map_config, default_model, max_threads
    )
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
    parallel_map_config, default_model, max_threads, api_wrapper
):
    operation = ParallelMapOperation(
        api_wrapper, parallel_map_config, default_model, max_threads
    )
    results, cost = operation.execute([])

    assert len(results) == 0
    assert cost == 0


def test_parallel_map_operation_with_empty_input(
    parallel_map_config, default_model, max_threads, api_wrapper
):
    operation = ParallelMapOperation(
        api_wrapper, parallel_map_config, default_model, max_threads
    )
    results, cost = operation.execute([])

    assert len(results) == 0
    assert cost == 0
