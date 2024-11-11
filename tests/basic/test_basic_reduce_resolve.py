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


def test_reduce_operation_with_lineage(
    reduce_config, max_threads, reduce_sample_data, api_wrapper
):
    # Add lineage configuration to reduce_config
    reduce_config["output"]["lineage"] = ["name", "email"]

    operation = ReduceOperation(
        api_wrapper, reduce_config, "text-embedding-3-small", max_threads
    )
    results, cost = operation.execute(reduce_sample_data)

    # Check if lineage information is present in the results
    assert all(f"{reduce_config['name']}_lineage" in result for result in results)

    # Check if lineage contains the specified keys
    for result in results:
        lineage = result[f"{reduce_config['name']}_lineage"]
        assert all(isinstance(item, dict) for item in lineage)
        assert all("name" in item and "email" in item for item in lineage)


def test_reduce_with_list_key(api_wrapper, default_model, max_threads):
    """Test reduce operation with a list-type key"""
    
    # Test data with list-type classifications
    input_data = [
        {
            "content": "Document about AI and ML",
            "classifications": ["AI", "ML"]
        },
        {
            "content": "Another ML and AI document",
            "classifications": ["ML", "AI"]  # Same classes but different order
        },
        {
            "content": "Document about AI only",
            "classifications": ["AI"]
        },
        {
            "content": "Document about ML and Data",
            "classifications": ["ML", "Data"]
        }
    ]

    # Configuration for reduce operation
    config = {
        "name": "test_reduce_list",
        "type": "reduce",
        "reduce_key": "classifications",
        "prompt": """Combine the content of documents with the same classifications.
            Input documents: {{ inputs }}
            Please combine the content into a single summary.""",
        "output": {
            "schema": {
                "combined_content": "string",
            }
        }
    }

    # Create and execute reduce operation
    operation = ReduceOperation(api_wrapper, config, default_model, max_threads)
    results, _ = operation.execute(input_data)

    # Verify results
    assert len(results) == 3  # Should have 3 groups: ["AI", "ML"], ["AI"], ["ML", "Data"]
    
    # Find the result with ["AI", "ML"] classifications
    ai_ml_result = next(r for r in results if len(r["classifications"]) == 2 and "AI" in r["classifications"] and "ML" in r["classifications"])
    assert len(ai_ml_result["classifications"]) == 2
    assert set(ai_ml_result["classifications"]) == {"AI", "ML"}
    
    # Find the result with only ["AI"] classification
    ai_result = next((r for r in results if r["classifications"] == ("AI",)), None)
    if ai_result is None:
        raise AssertionError("Could not find result with only ['AI'] classification")
    assert ai_result["classifications"] == ("AI",)
    
    # Find the result with ["ML", "Data"] classifications
    ml_data_result = next((r for r in results if set(r["classifications"]) == {"ML", "Data"}), None)
    if ml_data_result is None:
        raise AssertionError("Could not find result with ['ML', 'Data'] classification")
    assert set(ml_data_result["classifications"]) == {"ML", "Data"}

