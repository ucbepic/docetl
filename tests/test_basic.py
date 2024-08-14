import pytest
from motion.operations.map import MapOperation, ParallelMapOperation
from motion.operations.filter import FilterOperation
from motion.operations.explode import ExplodeOperation
from motion.operations.equijoin import EquijoinOperation
from motion.operations.split import SplitOperation
from motion.operations.reduce import ReduceOperation
from motion.operations.resolve import ResolveOperation
from dotenv import load_dotenv

load_dotenv()


@pytest.fixture
def default_model():
    return "gpt-4o-mini"


@pytest.fixture
def max_threads():
    return 4


# Map Operation Tests
@pytest.fixture
def map_config():
    return {
        "type": "map",
        "prompt": "Analyze the sentiment of the following text: '{{ input.text }}'. Classify it as either positive, negative, or neutral.",
        "output": {"schema": {"sentiment": "string"}},
        "model": "gpt-4o-mini",
    }


@pytest.fixture
def map_sample_data():
    return [
        {"text": "This is a positive sentence.", "sentiment": "unknown"},
        {"text": "This is a negative sentence.", "sentiment": "unknown"},
        {"text": "This is a neutral sentence.", "sentiment": "unknown"},
    ]


def test_map_operation(map_config, default_model, max_threads, map_sample_data):
    operation = MapOperation(map_config, default_model, max_threads)
    results, cost = operation.execute(map_sample_data)

    assert len(results) == len(map_sample_data)
    assert all("sentiment" in result for result in results)
    assert all(
        result["sentiment"] in ["positive", "negative", "neutral"] for result in results
    )
    assert cost > 0


def test_map_operation_empty_input(map_config, default_model, max_threads):
    operation = MapOperation(map_config, default_model, max_threads)
    results, cost = operation.execute([])

    assert len(results) == 0
    assert cost == 0


# Parallel Map Operation Tests
@pytest.fixture
def parallel_map_config():
    return {
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


def test_parallel_map_operation(
    parallel_map_config, default_model, max_threads, parallel_map_sample_data
):
    operation = ParallelMapOperation(parallel_map_config, default_model, max_threads)
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
    parallel_map_config, default_model, max_threads
):
    operation = ParallelMapOperation(parallel_map_config, default_model, max_threads)
    results, cost = operation.execute([])

    assert len(results) == 0
    assert cost == 0


# Filter Operation Tests
@pytest.fixture
def filter_config():
    return {
        "type": "filter",
        "prompt": "Determine if the following text is longer than 3 words: '{{ input.text }}'. Return true if it is, false otherwise.",
        "output": {"schema": {"keep": "boolean"}},
        "model": "gpt-4o-mini",
    }


@pytest.fixture
def filter_sample_data():
    return [
        {"text": "This is a short sentence.", "word_count": 5},
        {"text": "This is a longer sentence with more words.", "word_count": 8},
        {"text": "Brief.", "word_count": 1},
    ]


def test_filter_operation(
    filter_config, default_model, max_threads, filter_sample_data
):
    operation = FilterOperation(filter_config, default_model, max_threads)
    results, cost = operation.execute(filter_sample_data)

    assert len(results) < len(filter_sample_data)
    assert all(len(result["text"].split()) > 3 for result in results)
    assert cost > 0


def test_filter_operation_empty_input(filter_config, default_model, max_threads):
    operation = FilterOperation(filter_config, default_model, max_threads)
    results, cost = operation.execute([])

    assert len(results) == 0
    assert cost == 0


# Explode Operation Tests
@pytest.fixture
def explode_config():
    return {"type": "explode", "explode_key": "tag"}


@pytest.fixture
def explode_sample_data():
    return [
        {"id": 1, "tag": ["python", "testing", "pytest"]},
        {"id": 2, "tag": ["java", "spring"]},
        {"id": 3, "tag": []},
    ]


def test_explode_operation(
    explode_config, default_model, max_threads, explode_sample_data
):
    operation = ExplodeOperation(explode_config, default_model, max_threads)
    results, cost = operation.execute(explode_sample_data)

    assert len(results) == 5  # 3 + 2 + 0
    assert all("tag" in result for result in results)
    assert cost == 0  # Explode operation doesn't use LLM


def test_explode_operation_empty_input(explode_config, default_model, max_threads):
    operation = ExplodeOperation(explode_config, default_model, max_threads)
    results, cost = operation.execute([])

    assert len(results) == 0
    assert cost == 0


# Equijoin Operation Tests
@pytest.fixture
def equijoin_config():
    return {
        "type": "equijoin",
        "join_key": {"left": {"name": "id"}, "right": {"name": "user_id"}},
        "comparison_prompt": "Compare the following two entries and determine if they are the same id: Left: {{ left.id }} Right: {{ right.user_id }}",
        "embedding_model": "text-embedding-3-small",
        "comparison_model": "gpt-4o-mini",
    }


@pytest.fixture
def left_data():
    return [
        {"id": 1, "name": "John"},
        {"id": 2, "name": "Jane"},
        {"id": 3, "name": "Bob"},
    ]


@pytest.fixture
def right_data():
    return [
        {"user_id": 1, "email": "john@example.com"},
        {"user_id": 2, "email": "jane@example.com"},
        {"user_id": 4, "email": "alice@example.com"},
    ]


def test_equijoin_operation(
    equijoin_config, default_model, max_threads, left_data, right_data
):
    operation = EquijoinOperation(equijoin_config, default_model, max_threads)
    results, cost = operation.execute(left_data, right_data)

    assert len(results) == 2  # Only 2 matches
    assert all("name" in result and "email" in result for result in results)
    assert cost > 0


def test_equijoin_operation_empty_input(equijoin_config, default_model, max_threads):
    operation = EquijoinOperation(equijoin_config, default_model, max_threads)
    results, cost = operation.execute([], [])

    assert len(results) == 0
    assert cost == 0


@pytest.fixture
def split_config():
    return {
        "type": "split",
        "split_key": "content",
        "chunk_size": 4,
        "peripheral_chunks": {
            "previous": {
                "head": {"type": "full", "count": 1},
                "middle": {"type": "summary"},
                "tail": {"type": "full", "count": 1.5},
            },
            "next": {
                "head": {"type": "full", "count": 1},
                "tail": {"type": "summary", "count": 2},
            },
        },
        "summary_prompt": "Summarize the following chunk of content: {{ chunk_content }}\n If the chunk is too short, just repeat the content verbatim.",
    }


@pytest.fixture
def split_sample_data():
    return [
        {
            "id": 1,
            "content": "This is a long piece of content that needs to be split into smaller chunks for processing. It should create multiple chunks to test all aspects of the split operation.",
        },
        {
            "id": 2,
            "content": "Another piece of content that is just long enough to create two chunks.",
        },
    ]


def test_split_operation(split_config, default_model, max_threads, split_sample_data):
    operation = SplitOperation(split_config, default_model, max_threads)
    results, cost = operation.execute(split_sample_data)

    assert len(results) > len(split_sample_data)
    assert all("chunk_id" in result and "content" in result for result in results)
    assert cost > 0  # For summmarizing, we use the summary_prompt

    # Check that chunks are created correctly
    assert len(results) == 12  # 8 chunks for first item, 4 for second

    # Check previous chunks for the middle chunk of the first item
    middle_chunk = results[3][
        "_chunk_intermediates"
    ]  # 4th chunk (index 3) should be the middle chunk
    assert "previous_chunks" in middle_chunk
    assert (
        len(middle_chunk["previous_chunks"]) == 3
    )  # 1 head + 1 middle summary + 1.5 tail
    assert middle_chunk["previous_chunks"][0]["chunk_id"].startswith("chunk_0")
    assert middle_chunk["previous_chunks"][2]["chunk_id"].startswith("chunk_2")

    # Check next chunks for the middle chunk of the first item
    assert "next_chunks" in middle_chunk
    assert len(middle_chunk["next_chunks"]) == 3  # 1 head + 2 tail summaries
    assert middle_chunk["next_chunks"][0]["chunk_id"].startswith("chunk_4")

    # Check the first chunk of the second item
    second_item_first_chunk = results[8]["_chunk_intermediates"]
    assert len(second_item_first_chunk["previous_chunks"]) == 0, "No previous chunks"
    assert len(second_item_first_chunk["next_chunks"]) >= 1, "Some next chunks"


def test_split_operation_without_peripheral_chunks(
    split_config, default_model, max_threads, split_sample_data
):
    # Remove peripheral_chunks from config
    split_config.pop("peripheral_chunks")
    operation = SplitOperation(split_config, default_model, max_threads)
    results, cost = operation.execute(split_sample_data)

    assert len(results) > len(split_sample_data)
    assert all("chunk_id" in result and "content" in result for result in results)
    assert all(
        result["_chunk_intermediates"]["previous_chunks"] == []
        and result["_chunk_intermediates"]["next_chunks"] == []
        for result in results
    )
    assert cost == 0


def test_split_operation_with_partial_config(
    split_config, default_model, max_threads, split_sample_data
):
    # Modify config to only include previous chunks
    split_config["peripheral_chunks"] = {
        "previous": {"head": {"type": "full", "count": 1}}
    }
    operation = SplitOperation(split_config, default_model, max_threads)
    results, cost = operation.execute(split_sample_data)

    assert len(results) > len(split_sample_data)
    assert all(
        "next_chunks" in result["_chunk_intermediates"]
        and result["_chunk_intermediates"]["next_chunks"] == []
        for result in results
    )
    assert cost == 0

    # Check that only head is included in previous chunks
    for result in results:
        chunk_id = result["chunk_id"]
        if chunk_id.startswith("chunk_0"):
            continue
        result = result["_chunk_intermediates"]
        if "previous_chunks" in result:
            assert len(result["previous_chunks"]) == 1
            assert "chunk_id" in result["previous_chunks"][0]
            assert "summary" not in result["previous_chunks"][0]


@pytest.mark.parametrize(
    "invalid_config",
    [
        {"type": "split", "split_key": "content"},  # Missing chunk_size
        {"type": "split", "chunk_size": 10},  # Missing split_key
        {
            "type": "split",
            "split_key": "content",
            "chunk_size": "10",
        },  # Invalid chunk_size type
        {
            "type": "split",
            "split_key": "content",
            "chunk_size": 10,
            "peripheral_chunks": {"previous": {"head": {"type": "invalid"}}},
        },  # Invalid type in peripheral_chunks
        {
            "type": "split",
            "split_key": "content",
            "chunk_size": 10,
            "peripheral_chunks": {"previous": {"head": {"count": "1"}}},
        },  # Invalid count type in peripheral_chunks
    ],
)
def test_split_operation_invalid_config(invalid_config, default_model, max_threads):
    with pytest.raises((ValueError, TypeError)):
        SplitOperation(invalid_config, default_model, max_threads)


def test_split_operation_empty_input(split_config, default_model, max_threads):
    operation = SplitOperation(split_config, default_model, max_threads)
    results, cost = operation.execute([])

    assert len(results) == 0
    assert cost == 0


# Reduce Operation Tests
@pytest.fixture
def reduce_config():
    return {
        "type": "reduce",
        "reduce_key": "group",
        "prompt": "Summarize the following group of values: {{ values }} Provide a total and any other relevant statistics.",
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


def test_reduce_operation(
    reduce_config, default_model, max_threads, reduce_sample_data
):
    operation = ReduceOperation(reduce_config, default_model, max_threads)
    results, cost = operation.execute(reduce_sample_data)

    assert len(results) == 3  # 3 unique groups
    assert all(
        "group" in result and "total" in result and "avg" in result
        for result in results
    )
    assert cost > 0


def test_reduce_operation_empty_input(reduce_config, default_model, max_threads):
    operation = ReduceOperation(reduce_config, default_model, max_threads)
    results, cost = operation.execute([])

    assert len(results) == 0
    assert cost == 0


# Resolve Operation Tests
@pytest.fixture
def resolve_config():
    return {
        "type": "resolve",
        "blocking_keys": ["name", "email"],
        "blocking_threshold": 0.8,
        "comparison_prompt": "Compare the following two entries and determine if they likely refer to the same person: Person 1: {{ input1 }} Person 2: {{ input2 }} Return true if they likely match, false otherwise.",
        "output": {"schema": {"name": "string", "email": "string"}},
        "embedding_model": "text-embedding-3-small",
        "comparison_model": "gpt-4o-mini",
        "resolution_model": "gpt-4o-mini",
        "resolution_prompt": "Given the following list of similar entries, determine one common name and email. {{ matched_entries }}",
    }


@pytest.fixture
def resolve_sample_data():
    return [
        {"name": "John Doe", "email": "john@example.com"},
        {"name": "John D.", "email": "johnd@example.com"},
        {"name": "J. Smith", "email": "jane@example.com"},
        {"name": "J. Smith", "email": "jsmith@example.com"},
    ]


def test_resolve_operation(resolve_config, max_threads, resolve_sample_data):
    operation = ResolveOperation(resolve_config, "text-embedding-3-small", max_threads)
    results, cost = operation.execute(resolve_sample_data)

    distinct_names = set(result["name"] for result in results)
    assert len(distinct_names) < len(results)
    assert cost > 0


def test_resolve_operation_empty_input(resolve_config, max_threads):
    operation = ResolveOperation(resolve_config, "text-embedding-3-small", max_threads)
    results, cost = operation.execute([])

    assert len(results) == 0
    assert cost == 0
