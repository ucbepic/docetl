import pytest
from docetl.operations.map import MapOperation
from docetl.config_wrapper import ConfigWrapper
from docetl.runner import DSLRunner


@pytest.fixture
def default_model():
    return "gpt-4o-mini"


@pytest.fixture
def max_threads():
    return 4


@pytest.fixture
def response_lookup():
    return {
        "This is a good day.": {"sentiment": "positive", "word_count": 5},
        "This is a bad day.": {"sentiment": "negative", "word_count": 5},
        "This is just a day.": {"sentiment": "neutral", "word_count": 5},
        "Feeling great!": {"sentiment": "positive", "word_count": 2},
        "Everything is terrible.": {"sentiment": "negative", "word_count": 3},
        "Not sure how I feel.": {"sentiment": "neutral", "word_count": 5},
        "Good vibes only.": {"sentiment": "positive", "word_count": 3},
        "Bad news everywhere.": {"sentiment": "negative", "word_count": 4},
        "Neutral stance.": {"sentiment": "neutral", "word_count": 2},
        "Average day overall.": {"sentiment": "neutral", "word_count": 3},
    }


@pytest.fixture
def map_config():
    return {
        "name": "sentiment_analysis",
        "type": "map",
        "prompt": "Analyze the sentiment of the following text: '{{ input.text }}'. Classify it as either positive, negative, or neutral.",
        "output": {"schema": {"sentiment": "string"}},
        "model": "gpt-4o-mini",
    }


@pytest.fixture
def map_config_with_batching():
    return {
        "name": "sentiment_analysis_batching",
        "type": "map",
        "prompt": "Analyze the sentiment of the following text: '{{ input.text }}'. Classify it as either positive, negative, or neutral.",
        "output": {"schema": {"sentiment": "string"}},
        "model": "gpt-4o-mini",
        "batch_size": 2,  # Specify batch size for testing
    }


@pytest.fixture
def api_wrapper():
    return DSLRunner(
        {
            "default_model": "gpt-4o-mini",
            "operations": [],
            "pipeline": {"steps": [], "output": {"path": "/tmp/testingdocetl.json"}},
        },
        max_threads=64,
    )


@pytest.fixture
def test_map_operation_instance(
    map_config_with_batching, default_model, max_threads, api_wrapper
):
    return MapOperation(
        api_wrapper, map_config_with_batching, default_model, max_threads
    )


def test_map_operation_with_batching(
    map_config_with_batching, default_model, max_threads, api_wrapper
):
    operation = MapOperation(
        api_wrapper, map_config_with_batching, default_model, max_threads
    )

    # Sample data for testing
    map_sample_data = [
        {"text": "This is a positive sentence."},
        {"text": "This is a negative sentence."},
        {"text": "This is a neutral sentence."},
    ]

    results, cost = operation.execute(map_sample_data)

    assert len(results) == len(map_sample_data)
    assert cost > 0
    assert all("sentiment" in result for result in results)
    assert all(
        result["sentiment"] in ["positive", "negative", "neutral"] for result in results
    )


@pytest.fixture
def parallel_map_config_with_batching():
    return {
        "name": "sentiment_and_word_count",
        "type": "parallel_map",
        "prompts": [
            {
                "name": "sentiment",
                "prompt": "Analyze the sentiment of the following text: '{{ input.text }}'. Classify it as either positive, negative, or neutral.",
                "output_keys": ["sentiment"],
                "model": "gpt-4o-mini",
            },
            {
                "name": "word_count",
                "prompt": "Count the number of words in the following text: '{{ input.text }}'. Return the count as an integer.",
                "output_keys": ["word_count"],
                "model": "gpt-4o-mini",
            },
        ],
        "output": {"schema": {"sentiment": "string", "word_count": "integer"}},
        "batch_size": 2,
        "clustering_method": "sem_cluster",
    }


@pytest.fixture
def map_sample_data():
    return [
        {"text": "This is a good day."},
        {"text": "This is a bad day."},
        {"text": "This is just a day."},
    ]


@pytest.fixture
def map_sample_data_large():
    return [
        {"text": "This is a good day."},
        {"text": "This is a bad day."},
        {"text": "This is just a day."},
        {"text": "Feeling great!"},
        {"text": "Everything is terrible."},
        {"text": "Not sure how I feel."},
        {"text": "Good vibes only."},
        {"text": "Bad news everywhere."},
        {"text": "Neutral stance."},
        {"text": "Average day overall."},
    ]


# Parallel Map Operation Tests
@pytest.fixture
def parallel_map_config():
    return {
        "name": "sentiment_and_word_count",
        "type": "parallel_map",
        "prompts": [
            {
                "name": "sentiment",
                "prompt": "Analyze the sentiment of the following text: '{{ input.text }}'. Classify it as either positive, negative, or neutral.",
                "output_keys": ["sentiment"],
                "model": "gpt-4o-mini",
            },
            {
                "name": "word_count",
                "prompt": "Count the number of words in the following text: '{{ input.text }}'. Return the count as an integer.",
                "output_keys": ["word_count"],
                "model": "gpt-4o-mini",
            },
        ],
        "output": {"schema": {"sentiment": "string", "word_count": "integer"}},
    }


@pytest.fixture
def parallel_map_sample_data():
    return [
        {"text": "This is a positive sentence."},
        {"text": "This is a negative sentence."},
        {"text": "This is a neutral sentence."},
    ]


@pytest.fixture
def synthetic_data():
    return [
        {"text": "This is a short sentence."},
        {"text": "This sentence has exactly six words."},
        {"text": "Pneumonoultramicroscopicsilicovolcanoconiosis is a long word."},
        {"text": "One"},
    ]


@pytest.fixture
def map_config_with_drop_keys():
    return {
        "name": "sentiment_analysis_with_drop",
        "type": "map",
        "prompt": "Analyze the sentiment of the following text: '{{ input.text }}'. Classify it as either positive, negative, or neutral.",
        "output": {"schema": {"sentiment": "string"}},
        "model": "gpt-4o-mini",
        "drop_keys": ["to_be_dropped"],
    }


@pytest.fixture
def map_config_with_drop_keys_no_prompt():
    return {
        "name": "drop_keys_only",
        "type": "map",
        "drop_keys": ["to_be_dropped"],
        "model": "gpt-4o-mini",
    }


@pytest.fixture
def map_sample_data_with_extra_keys():
    return [
        {
            "text": "This is a positive sentence.",
            "original_sentiment": "positive",
            "to_be_dropped": "extra",
        },
        {
            "text": "This is a negative sentence.",
            "original_sentiment": "negative",
            "to_be_dropped": "extra",
        },
        {
            "text": "This is a neutral sentence.",
            "original_sentiment": "neutral",
            "to_be_dropped": "extra",
        },
    ]


@pytest.fixture
def map_config_with_tools():
    return {
        "type": "map",
        "name": "word_count",
        "prompt": "Count the number of words in the following text: '{{ input.text }}'",
        "output": {"schema": {"word_count": "integer"}},
        "model": "gpt-4o-mini",
        "tools": [
            {
                "required": True,
                "code": """
def count_words(text):
    return {"word_count": len(text.split())}
                """,
                "function": {
                    "name": "count_words",
                    "description": "Count the number of words in a text string.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "text": {
                                "type": "string",
                            }
                        },
                        "required": ["text"],
                    },
                },
            }
        ],
        "validate": ["len(output['text']) > 0"],
        "num_retries_on_validate_failure": 3,
    }
