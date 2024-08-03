import pytest
from motion.operations.reduce import ReduceOperation
from dotenv import load_dotenv

load_dotenv()


@pytest.fixture
def default_model():
    return "gpt-4o-mini"


@pytest.fixture
def max_threads():
    return 4


@pytest.fixture
def reduce_config():
    return {
        "type": "reduce",
        "reduce_key": "group",
        "prompt": "Summarize the following group of values: {{ values }} Provide a total and any other relevant statistics.",
        "fold_prompt": "Combine the following summaries: {{ output }} with new values: {{ values }}. Provide an updated total and statistics.",
        "merge_prompt": "Merge the following summaries: {% for output in outputs %}{{ output.total }}, {{ output.avg }}, {{ output.count }}{% if not loop.last %} | {% endif %}{% endfor %}. Provide a final total and statistics.",
        "output": {"schema": {"total": "number", "avg": "number", "count": "number"}},
        "model": "gpt-3.5-turbo",  # Using a more widely available model
        "fold_batch_size": 3,
        "merge_batch_size": 2,
        "num_parallel_folds": 3,
    }


@pytest.fixture
def reduce_sample_data():
    return [
        {"group": "A", "value": 10},
        {"group": "B", "value": 20},
        {"group": "A", "value": 15},
        {"group": "C", "value": 30},
        {"group": "B", "value": 25},
        {"group": "A", "value": 20},
        {"group": "C", "value": 35},
        {"group": "B", "value": 30},
        {"group": "A", "value": 12},
        {"group": "C", "value": 40},
        {"group": "B", "value": 22},
        {"group": "A", "value": 18},
        {"group": "C", "value": 33},
        {"group": "B", "value": 28},
        {"group": "A", "value": 14},
    ] * 10


def test_reduce_operation(
    reduce_config, default_model, max_threads, reduce_sample_data
):
    operation = ReduceOperation(reduce_config, default_model, max_threads)
    results, cost = operation.execute(reduce_sample_data)

    assert len(results) == 3  # 3 unique groups
    assert all(
        "group" in result
        and "total" in result
        and "avg" in result
        and "count" in result
        for result in results
    )
    assert cost > 0


def test_reduce_operation_error_handling(reduce_config, default_model, max_threads):
    # Test with invalid num_parallel_folds
    invalid_config = reduce_config.copy()
    invalid_config["num_parallel_folds"] = 0

    with pytest.raises(
        ValueError, match="'num_parallel_folds' must be a positive integer"
    ):
        ReduceOperation(invalid_config, default_model, max_threads)

    # Test with missing fold_prompt when merge_prompt is present
    invalid_config = reduce_config.copy()
    del invalid_config["fold_prompt"]

    with pytest.raises(
        ValueError,
        match="'fold_prompt' and 'num_parallel_folds' are required when 'merge_prompt' is specified",
    ):
        ReduceOperation(invalid_config, default_model, max_threads)


def test_reduce_operation_pass_through(
    reduce_config, default_model, max_threads, reduce_sample_data
):
    reduce_config["pass_through"] = True

    operation = ReduceOperation(reduce_config, default_model, max_threads)
    results, cost = operation.execute(reduce_sample_data)

    assert len(results) == 3  # 3 unique groups
    print(results[0].keys())
    assert all(
        "group" in result
        and "total" in result
        and "avg" in result
        and "count" in result
        and "value" in result
        and "category" in result
        for result in results
    )
    assert cost > 0

    # Check if pass-through fields are present and correct
    for result in results:
        group_data = [
            item for item in reduce_sample_data if item["group"] == result["group"]
        ]
        assert result["value"] in [item["value"] for item in group_data]
        assert result["category"] in [item["category"] for item in group_data]
