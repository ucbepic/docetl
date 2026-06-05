import pytest
from docetl.operations.map import MapOperation
from docetl.config_wrapper import ConfigWrapper
from docetl.runner import DSLRunner

import docetl.operations.utils.cascade_runner as cascade_runner


# =============================================================================
# MODEL-CASCADE CACHE ISOLATION
# =============================================================================

class _MemCache:
    """In-memory stand-in for the diskcache ``cache`` (context-manager + get/set)."""

    def __init__(self):
        self.store = {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get(self, k):
        return self.store.get(k)

    def set(self, k, v):
        self.store[k] = v


@pytest.fixture(autouse=True)
def cascade_cache(monkeypatch):
    """Isolate the model-cascade cache for every test.

    The cascade reuses the shared on-disk DocETL cache; without isolation,
    results leak across runs (a cache hit makes zero proxy/oracle calls,
    breaking call-count assertions). Each test gets a fresh in-memory cache.
    Tests that need to inspect it can request this fixture by name.
    """
    mem = _MemCache()
    monkeypatch.setattr(cascade_runner, "cache", mem, raising=False)
    return mem

# =============================================================================
# BASIC TEST CONFIGURATION FIXTURES (SHARED ACROSS MULTIPLE TEST FILES)
# =============================================================================

@pytest.fixture
def default_model():
    return "gpt-4o-mini"


@pytest.fixture
def max_threads():
    return 4


@pytest.fixture
def runner():
    return DSLRunner(
        {
            "default_model": "gpt-4o-mini",
            "operations": [],
            "pipeline": {"steps": [], "output": {"path": "/tmp/testingdocetl.json"}},
        },
        max_threads=64,
    )


# =============================================================================
# SHARED TEST DATA FIXTURES (USED ACROSS MULTIPLE TEST FILES)
# =============================================================================

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


@pytest.fixture
def synthetic_data():
    return [
        {"text": "This is a short sentence."},
        {"text": "This sentence has exactly six words."},
        {"text": "Pneumonoultramicroscopicsilicovolcanoconiosis is a long word."},
        {"text": "One"},
    ]


# =============================================================================
# SHARED OPERATION CONFIGURATION FIXTURES
# =============================================================================

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



