import pytest
from docetl.operations.map import MapOperation
from tests.conftest import api_wrapper, default_model, max_threads


@pytest.fixture
def map_config_with_validation():
    return {
        "name": "sentiment_analysis_with_validation",
        "type": "map",
        "prompt": "Analyze the sentiment of the following text: '{{ input.text }}'. Classify it as either positive, negative, or neutral.",
        "output": {"schema": {"sentiment": "string", "confidence": "float"}},
        "model": "gpt-4o-mini",
        "validate": [
            "output['sentiment'] in ['positive', 'negative', 'neutral']",
            "0 <= output['confidence'] <= 1",
        ],
        "num_retries_on_validate_failure": 2,
    }


@pytest.fixture
def sample_data():
    return [
        {"text": "I love this product! It's amazing."},
        {"text": "This is the worst experience ever."},
        {"text": "The weather is okay today."},
    ]


def test_map_operation_with_validation(
    map_config_with_validation, sample_data, api_wrapper, default_model, max_threads
):
    map_config_with_validation["bypass_cache"] = True
    operation = MapOperation(
        api_wrapper, map_config_with_validation, default_model, max_threads
    )
    results, cost = operation.execute(sample_data)

    assert len(results) == len(sample_data)
    assert cost > 0

    for result in results:
        assert "sentiment" in result
        assert "confidence" in result
        assert result["sentiment"] in ["positive", "negative", "neutral"]
        assert 0 <= result["confidence"] <= 1
