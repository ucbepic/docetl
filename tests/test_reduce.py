import pytest
from motion.operations.reduce import ReduceOperation
from dotenv import load_dotenv
import random

load_dotenv()


@pytest.fixture
def default_model():
    return "gpt-4o-mini"


@pytest.fixture
def max_threads():
    return 64


@pytest.fixture
def reduce_config():
    return {
        "type": "reduce",
        "reduce_key": "category",
        "prompt": "Categorize and summarize the following items: {{ inputs }} Provide a brief summary of the category and list the most common themes.",
        "fold_prompt": "Combine the following category summaries: {{ output }} with new items: {{ inputs }}. Provide an updated summary and list of common themes.",
        "merge_prompt": "Merge the following category summaries: {% for output in outputs %}{{ output.summary }}, Themes: {{ output.themes }}{% if not loop.last %} | {% endif %}{% endfor %}. Provide a final summary and list of common themes for each category.",
        "output": {"schema": {"summary": "string", "themes": "list[string]"}},
        "fold_batch_size": 3,
        "merge_batch_size": 2,
    }


@pytest.fixture
def reduce_sample_data():
    categories = ["Technology", "Nature", "Culture"]
    tech_items = [
        "smartphone",
        "laptop",
        "AI",
        "robotics",
        "virtual reality",
        "blockchain",
        "cloud computing",
        "IoT",
        "5G",
        "quantum computing",
    ]
    nature_items = [
        "forest",
        "ocean",
        "mountain",
        "wildlife",
        "ecosystem",
        "climate",
        "biodiversity",
        "conservation",
        "sustainability",
        "renewable energy",
    ]
    culture_items = [
        "art",
        "music",
        "literature",
        "cuisine",
        "festival",
        "language",
        "tradition",
        "fashion",
        "religion",
        "history",
    ]

    def generate_item(category):
        if category == "Technology":
            return {"category": category, "item": random.choice(tech_items)}
        elif category == "Nature":
            return {"category": category, "item": random.choice(nature_items)}
        else:
            return {"category": category, "item": random.choice(culture_items)}

    return [generate_item(random.choice(categories)) for _ in range(300)]


def test_reduce_operation(
    reduce_config, default_model, max_threads, reduce_sample_data
):
    operation = ReduceOperation(reduce_config, default_model, max_threads)
    results, cost = operation.execute(reduce_sample_data)

    assert len(results) == 3, "Should have results for 3 unique categories"
    assert cost > 0, "Cost should be greater than 0"

    for result in results:
        assert "category" in result, "Each result should have a 'category' key"
        assert "summary" in result, "Each result should have a 'summary' key"
        assert "themes" in result, "Each result should have a 'themes' key"

        assert isinstance(result["summary"], str), "'summary' should be a string"
        assert isinstance(result["themes"], list), "'themes' should be a list"

        assert len(result["summary"]) > 0, "'summary' should not be empty"
        assert len(result["themes"]) > 0, "'themes' should not be empty"


def test_reduce_operation_pass_through(
    reduce_config, default_model, max_threads, reduce_sample_data
):
    reduce_config["pass_through"] = True
    operation = ReduceOperation(reduce_config, default_model, max_threads)
    results, cost = operation.execute(reduce_sample_data)

    assert len(results) == 3, "Should have results for 3 unique categories"
    assert cost > 0, "Cost should be greater than 0"

    for result in results:
        assert "category" in result, "Each result should have a 'category' key"
        assert "summary" in result, "Each result should have a 'summary' key"
        assert "themes" in result, "Each result should have a 'themes' key"
        assert (
            "item" in result
        ), "Pass-through field 'item' should be present in the result"

        category_data = [
            item
            for item in reduce_sample_data
            if item["category"] == result["category"]
        ]
        assert result["item"] in [
            item["item"] for item in category_data
        ], "Pass-through 'item' should be from the correct category"


def test_reduce_operation_error_handling(reduce_config, default_model, max_threads):

    # Test with missing fold_prompt when merge_prompt is present
    invalid_config = reduce_config.copy()
    invalid_config["merge_prompt"] = "Some merge prompt"
    if "fold_prompt" in invalid_config:
        del invalid_config["fold_prompt"]

    with pytest.raises(
        ValueError,
        match="'fold_prompt' is required when 'merge_prompt' is specified",
    ):
        ReduceOperation(invalid_config, default_model, max_threads)


def test_reduce_operation_non_commutative(default_model, max_threads):
    # Define a new non-commutative reduce config
    non_commutative_config = {
        "reduce_key": "sequence",
        "commutative": False,
        "prompt": "Combine the sentences in '{{ inputs }}'. Maintain order.",
        "fold_prompt": "Combine sequences: Previous result '{{ output }}', New value '{{ inputs[0] }}'. Maintain order.",
        "fold_batch_size": 1,
        "output": {"schema": {"combined_result": "string"}},
    }

    # Sample data where order matters
    sample_data = [
        {"sequence": "story", "value": "Once upon a time"},
        {"sequence": "story", "value": "there was a brave princess"},
        {"sequence": "story", "value": "who rescued a dragon"},
        {"sequence": "story", "value": "and lived happily ever after."},
    ]

    operation = ReduceOperation(non_commutative_config, default_model, max_threads)
    results, cost = operation.execute(sample_data)

    assert len(results) == 1, "Should have one result for the 'story' sequence"
    assert cost > 0, "Cost should be greater than 0"

    result = results[0]
    assert "combined_result" in result, "Result should have a 'combined_result' key"

    # Check if the order of processing is preserved in the combined result
    combined_result = result["combined_result"].lower()
    assert (
        "once upon a time" in combined_result
    ), "Should contain the beginning of the story"
    assert (
        "happily ever after" in combined_result
    ), "Should contain the end of the story"
    assert combined_result.index("once upon a time") < combined_result.index(
        "happily ever after"
    ), "Order in combined result should match input order"

    # Additional checks for story coherence
    assert "brave princess" in combined_result, "Should mention the brave princess"
    assert "dragon" in combined_result, "Should mention the dragon"
    assert combined_result.index("brave princess") < combined_result.index(
        "dragon"
    ), "Princess should be mentioned before the dragon in the story"
