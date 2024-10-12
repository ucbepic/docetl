# Reduce Operation Tests
import pytest
from docetl.operations.reduce import ReduceOperation
from docetl.operations.resolve import ResolveOperation
from tests.conftest import api_wrapper


@pytest.fixture
def reduce_config():
    return {
        "name": "group_summary",
        "type": "reduce",
        "reduce_key": "group",
        "prompt": "Summarize the following group of values: {{ inputs }} Provide a total and any other relevant statistics.",
        "output": {"schema": {"total": "number", "avg": "number"}},
        "model": "gpt-4o-mini",
    }


@pytest.fixture
def reduce_sample_data():
    return [
        {"group": "A", "value": 10},
        {"group": "B", "value": 20},
        {"group": "A", "value": 15},
        {"group": "C", "value": 30},
        {"group": "B", "value": 25},
    ]


@pytest.fixture
def reduce_sample_data_with_list_key():
    return [
        {"group": "A", "value": 10, "category": "X"},
        {"group": "B", "value": 20, "category": "Y"},
        {"group": "A", "value": 15, "category": "X"},
        {"group": "C", "value": 30, "category": "Z"},
        {"group": "B", "value": 25, "category": "Y"},
    ]


def test_reduce_operation(
    reduce_config, default_model, max_threads, reduce_sample_data, api_wrapper
):
    reduce_config["bypass_cache"] = True
    operation = ReduceOperation(api_wrapper, reduce_config, default_model, max_threads)
    results, cost = operation.execute(reduce_sample_data)

    assert len(results) == 3  # 3 unique groups
    assert all(
        "group" in result and "total" in result and "avg" in result
        for result in results
    )
    assert cost > 0


def test_reduce_operation_with_all_key(
    reduce_config, default_model, max_threads, reduce_sample_data, api_wrapper
):
    reduce_config["reduce_key"] = "_all"
    operation = ReduceOperation(api_wrapper, reduce_config, default_model, max_threads)
    results, cost = operation.execute(reduce_sample_data)

    assert len(results) == 1


def test_reduce_operation_with_list_key(
    reduce_config,
    default_model,
    max_threads,
    reduce_sample_data_with_list_key,
    api_wrapper,
):
    reduce_config["reduce_key"] = ["group", "category"]

    operation = ReduceOperation(api_wrapper, reduce_config, default_model, max_threads)
    results, cost = operation.execute(reduce_sample_data_with_list_key)

    assert len(results) == 3  # 3 unique groups
    assert all(
        "group" in result
        and "category" in result
        and "total" in result
        and "avg" in result
        for result in results
    )


def test_reduce_operation_empty_input(
    reduce_config, default_model, max_threads, api_wrapper
):
    operation = ReduceOperation(api_wrapper, reduce_config, default_model, max_threads)
    results, cost = operation.execute([])

    assert len(results) == 0
    assert cost == 0


# Resolve Operation Tests
@pytest.fixture
def resolve_config():
    return {
        "name": "name_email_resolver",
        "type": "resolve",
        "blocking_keys": ["name", "email"],
        "blocking_threshold": 0.8,
        "comparison_prompt": "Compare the following two entries and determine if they likely refer to the same person: Person 1: {{ input1 }} Person 2: {{ input2 }} Return true if they likely match, false otherwise.",
        "output": {"schema": {"name": "string", "email": "string"}},
        "embedding_model": "text-embedding-3-small",
        "comparison_model": "gpt-4o-mini",
        "resolution_model": "gpt-4o-mini",
        "resolution_prompt": "Given the following list of similar entries, determine one common name and email. {{ inputs }}",
    }


@pytest.fixture
def resolve_sample_data():
    return [
        {"name": "John Doe", "email": "john@example.com"},
        {"name": "John D.", "email": "johnd@example.com"},
        {"name": "J. Smith", "email": "jane@example.com"},
        {"name": "J. Smith", "email": "jsmith@example.com"},
    ]


def test_resolve_operation(
    resolve_config, max_threads, resolve_sample_data, api_wrapper
):
    operation = ResolveOperation(
        api_wrapper, resolve_config, "text-embedding-3-small", max_threads
    )
    results, cost = operation.execute(resolve_sample_data)

    distinct_names = set(result["name"] for result in results)
    assert len(distinct_names) < len(results)


def test_resolve_operation_empty_input(resolve_config, max_threads, api_wrapper):
    operation = ResolveOperation(
        api_wrapper, resolve_config, "text-embedding-3-small", max_threads
    )
    results, cost = operation.execute([])

    assert len(results) == 0
    assert cost == 0
