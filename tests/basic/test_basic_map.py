# ruff: noqa: F811

from docetl.operations.map import MapOperation
from tests.conftest import (
    api_wrapper as api_wrapper,
    map_config_with_batching as map_config_with_batching,
    default_model as default_model,
    max_threads as max_threads,
    map_sample_data as map_sample_data,
    map_sample_data_large as map_sample_data_large,
    map_config as map_config,
    test_map_operation_instance as test_map_operation_instance,
    map_config_with_drop_keys as map_config_with_drop_keys,
    map_sample_data_with_extra_keys as map_sample_data_with_extra_keys,
    map_config_with_drop_keys_no_prompt as map_config_with_drop_keys_no_prompt,
)
import pytest
import docetl


def test_map_operation(
    test_map_operation_instance,
    map_sample_data,
):
    results, cost = test_map_operation_instance.execute(map_sample_data)

    assert len(results) == len(map_sample_data)
    assert all("sentiment" in result for result in results)
    valid_sentiments = ["positive", "negative", "neutral"]
    assert all(
        any(vs in result["sentiment"] for vs in valid_sentiments) for result in results
    )


def test_map_operation_empty_input(map_config, default_model, max_threads, api_wrapper):
    operation = MapOperation(api_wrapper, map_config, default_model, max_threads)
    results, cost = operation.execute([])

    assert len(results) == 0
    assert cost == 0


def test_map_operation_with_drop_keys(
    map_config_with_drop_keys,
    default_model,
    max_threads,
    map_sample_data_with_extra_keys,
    api_wrapper,
):
    map_config_with_drop_keys["bypass_cache"] = True
    operation = MapOperation(
        api_wrapper, map_config_with_drop_keys, default_model, max_threads
    )
    results, cost = operation.execute(map_sample_data_with_extra_keys)

    assert len(results) == len(map_sample_data_with_extra_keys)
    assert all("sentiment" in result for result in results)
    assert all("original_sentiment" in result for result in results)
    assert all("to_be_dropped" not in result for result in results)
    assert all(
        result["sentiment"] in ["positive", "negative", "neutral"] for result in results
    )

    assert cost > 0


def test_map_operation_with_drop_keys_no_prompt(
    map_config_with_drop_keys_no_prompt,
    default_model,
    max_threads,
    map_sample_data_with_extra_keys,
    api_wrapper,
):
    operation = MapOperation(
        api_wrapper, map_config_with_drop_keys_no_prompt, default_model, max_threads
    )
    results, cost = operation.execute(map_sample_data_with_extra_keys)

    assert len(results) == len(map_sample_data_with_extra_keys)
    assert all("to_be_dropped" not in result for result in results)
    assert all("text" in result for result in results)
    assert all("original_sentiment" in result for result in results)
    assert cost == 0  # No LLM calls should be made


def test_map_operation_with_batching(
    map_config_with_batching,
    default_model,
    max_threads,
    map_sample_data,
    api_wrapper,
):
    operation = MapOperation(
        api_wrapper, map_config_with_batching, default_model, max_threads
    )
    results, cost = operation.execute(map_sample_data)

    assert len(results) == len(map_sample_data)
    assert all("sentiment" in result for result in results)
    valid_sentiments = ["positive", "negative", "neutral"]
    assert all(
        any(vs in result["sentiment"] for vs in valid_sentiments) for result in results
    )


def test_map_operation_with_empty_input(
    map_config_with_batching, default_model, max_threads, api_wrapper
):
    operation = MapOperation(
        api_wrapper, map_config_with_batching, default_model, max_threads
    )
    results, cost = operation.execute([])

    assert len(results) == 0
    assert cost == 0


def test_map_operation_with_large_max_batch_size(
    map_config_with_batching,
    default_model,
    max_threads,
    map_sample_data,
    api_wrapper,
):
    map_config_with_batching["max_batch_size"] = 5  # Set batch size larger than data
    operation = MapOperation(
        api_wrapper, map_config_with_batching, default_model, max_threads
    )
    results, cost = operation.execute(map_sample_data)

    assert len(results) == len(map_sample_data)


def test_map_operation_with_word_count_tool(
    map_config_with_tools, synthetic_data, api_wrapper
):
    operation = MapOperation(api_wrapper, map_config_with_tools, "gpt-4o-mini", 4)
    results, cost = operation.execute(synthetic_data)

    assert len(results) == len(synthetic_data)
    assert all("word_count" in result for result in results)
    assert [result["word_count"] for result in results] == [5, 6, 5, 1]


@pytest.fixture
def simple_map_config():
    return {
        "name": "simple_sentiment_analysis",
        "type": "map",
        "prompt": "Analyze the sentiment of the following text: '{{ input.text }}'. Classify it as either positive, negative, or neutral.",
        "output": {"schema": {"sentiment": "string"}},
        "model": "gpt-4o-mini",
    }


@pytest.fixture
def simple_sample_data():
    import random
    import string

    def generate_random_text(length):
        return "".join(
            random.choice(
                string.ascii_letters + string.digits + string.punctuation + " "
            )
            for _ in range(length)
        )

    return [
        {"text": generate_random_text(random.randint(20, 100000))},
        {"text": generate_random_text(random.randint(20, 100000))},
        {"text": generate_random_text(random.randint(20, 100000))},
    ]


@pytest.mark.flaky(reruns=2)
def test_map_operation_with_timeout(simple_map_config, simple_sample_data, api_wrapper):
    # Add timeout to the map configuration
    map_config_with_timeout = {
        **simple_map_config,
        "timeout": 1,
        "max_retries_per_timeout": 0,
        "bypass_cache": True,
    }

    operation = MapOperation(api_wrapper, map_config_with_timeout, "gpt-4o-mini", 4)

    # Execute the operation and expect empty results
    results, cost = operation.execute(simple_sample_data)
    assert len(results) == 0


def test_map_operation_with_gleaning(simple_map_config, map_sample_data, api_wrapper):
    # Add gleaning configuration to the map configuration
    map_config_with_gleaning = {
        **simple_map_config,
        "gleaning": {
            "num_rounds": 2,
            "validation_prompt": "Review the sentiment analysis. Is it accurate? If not, suggest improvements.",
        },
        "bypass_cache": True,
    }

    operation = MapOperation(api_wrapper, map_config_with_gleaning, "gpt-4o-mini", 4)

    # Execute the operation
    results, cost = operation.execute(map_sample_data)

    # Assert that we have results for all input items
    assert len(results) == len(map_sample_data)

    # Check that all results have a sentiment
    assert all("sentiment" in result for result in results)

    # Verify that all sentiments are valid
    valid_sentiments = ["positive", "negative", "neutral"]
    assert all(
        any(vs in result["sentiment"] for vs in valid_sentiments) for result in results
    )
