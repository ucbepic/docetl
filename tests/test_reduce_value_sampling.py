import pytest
import random
from docetl.operations.reduce import ReduceOperation
from tests.conftest import api_wrapper


@pytest.fixture
def default_model():
    return "gpt-4o-mini"


@pytest.fixture
def max_threads():
    return 64


@pytest.fixture
def large_sample_data():
    groups = ["A", "B", "C"]
    topics = ["technology", "science", "politics", "economics", "culture"]

    def generate_text():
        return f"This is a sample text about {random.choice(topics)}."

    data = []
    for _ in range(1000):  # Generate 1000 items
        group = random.choice(groups)
        text = generate_text()
        importance = random.randint(1, 10)
        data.append({"group": group, "text": text, "importance": importance})

    return data


def test_random_sampling(api_wrapper, default_model, max_threads, large_sample_data):
    config = {
        "name": "reduce_value_sampling",
        "type": "reduce",
        "reduce_key": "group",
        "value_sampling": {"enabled": True, "method": "random", "sample_size": 50},
        "prompt": "Summarize the following texts: {{ inputs|map(attribute='text')|join(' | ') }}",
        "output": {"schema": {"summary": "string"}},
    }

    operation = ReduceOperation(api_wrapper, config, default_model, max_threads)
    results, cost = operation.execute(large_sample_data)

    assert len(results) == 3, "Should have results for all three groups A, B, and C"
    for result in results:
        assert "summary" in result, "Each result should have a summary"
        assert len(result["summary"]) > 0, "Summary should not be empty"


def test_first_n_sampling(api_wrapper, default_model, max_threads, large_sample_data):
    config = {
        "name": "reduce_value_sampling",
        "type": "reduce",
        "reduce_key": "group",
        "value_sampling": {"enabled": True, "method": "first_n", "sample_size": 100},
        "prompt": "Summarize the following texts: {{ inputs|map(attribute='text')|join(' | ') }}",
        "output": {"schema": {"summary": "string"}},
    }

    operation = ReduceOperation(api_wrapper, config, default_model, max_threads)
    results, cost = operation.execute(large_sample_data)

    assert len(results) == 3, "Should have results for all three groups A, B, and C"
    for result in results:
        assert "summary" in result, "Each result should have a summary"
        assert len(result["summary"]) > 0, "Summary should not be empty"


def test_cluster_sampling(api_wrapper, default_model, max_threads, large_sample_data):
    config = {
        "name": "reduce_value_sampling",
        "type": "reduce",
        "reduce_key": "group",
        "value_sampling": {
            "enabled": True,
            "method": "cluster",
            "sample_size": 50,
            "embedding_model": "text-embedding-3-small",
            "embedding_keys": ["text"],
        },
        "prompt": "Summarize the following texts: {{ inputs|map(attribute='text')|join(' | ') }}",
        "output": {"schema": {"summary": "string"}},
    }

    operation = ReduceOperation(api_wrapper, config, default_model, max_threads)
    results, cost = operation.execute(large_sample_data)

    assert len(results) == 3, "Should have results for all three groups A, B, and C"
    for result in results:
        assert "summary" in result, "Each result should have a summary"
        assert len(result["summary"]) > 0, "Summary should not be empty"


def test_semantic_similarity_sampling(
    api_wrapper, default_model, max_threads, large_sample_data
):
    config = {
        "name": "reduce_value_sampling",
        "type": "reduce",
        "reduce_key": "group",
        "value_sampling": {
            "enabled": True,
            "method": "sem_sim",
            "sample_size": 20,
            "embedding_model": "text-embedding-3-small",
            "embedding_keys": ["text"],
            "query_text": "technology",
        },
        "prompt": "Summarize the following texts: {{ inputs|map(attribute='text')|join(' | ') }}",
        "output": {"schema": {"summary": "string"}},
    }

    operation = ReduceOperation(api_wrapper, config, default_model, max_threads)
    results, cost = operation.execute(large_sample_data)

    assert len(results) == 3, "Should have results for all three groups A, B, and C"
    for result in results:
        assert "summary" in result, "Each result should have a summary"
        assert len(result["summary"]) > 0, "Summary should not be empty"

        # make sure there's no mention of "science", "politics", "economics", "culture"
        assert "science" not in result["summary"]
        assert "politics" not in result["summary"]
        assert "economics" not in result["summary"]
        assert "culture" not in result["summary"]


def test_invalid_sampling_method(
    api_wrapper, default_model, max_threads, large_sample_data
):
    config = {
        "name": "reduce_value_sampling",
        "type": "reduce",
        "reduce_key": "group",
        "value_sampling": {
            "enabled": True,
            "method": "invalid_method",
            "sample_size": 50,
        },
        "prompt": "Summarize the following texts: {{ inputs|map(attribute='text')|join(' | ') }}",
        "output": {"schema": {"summary": "string"}},
    }

    with pytest.raises(ValueError, match="Invalid 'method'"):
        ReduceOperation(api_wrapper, config, default_model, max_threads)
