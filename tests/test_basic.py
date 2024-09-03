import pytest
from motion.operations.map import MapOperation, ParallelMapOperation
from motion.operations.filter import FilterOperation
from motion.operations.unnest import UnnestOperation
from motion.operations.equijoin import EquijoinOperation
from motion.operations.split import SplitOperation
from motion.operations.gather import GatherOperation
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


# Unnest Operation Tests
@pytest.fixture
def unnest_config():
    return {"type": "unnest", "unnest_key": "tag", "keep_empty": True}


@pytest.fixture
def unnest_sample_data():
    return [
        {"id": 1, "tag": ["python", "testing", "pytest"]},
        {"id": 2, "tag": ["java", "spring"]},
        {"id": 3, "tag": []},
    ]


@pytest.fixture
def dict_unnest_config():
    return {
        "type": "unnest",
        "unnest_key": "details",
        "expand_fields": ["age", "city", "occupation"],
    }


@pytest.fixture
def dict_unnest_sample_data():
    return [
        {"id": 1, "details": {"age": 30, "city": "New York", "occupation": "Engineer"}},
        {
            "id": 2,
            "details": {"age": 25, "city": "San Francisco", "occupation": "Designer"},
        },
    ]


def test_dict_unnest_operation(
    dict_unnest_config, default_model, max_threads, dict_unnest_sample_data
):
    operation = UnnestOperation(dict_unnest_config, default_model, max_threads)
    results, cost = operation.execute(dict_unnest_sample_data)

    assert len(results) == 2  # due to keep_empty=False
    assert all("age" in result for result in results)
    assert all("city" in result for result in results)
    assert all("details" in result for result in results)
    assert all("occupation" in result for result in results)
    assert results[0]["age"] == 30
    assert results[0]["city"] == "New York"
    assert results[0]["occupation"] == "Engineer"
    assert results[1]["age"] == 25
    assert results[1]["city"] == "San Francisco"
    assert results[1]["occupation"] == "Designer"
    assert cost == 0  # Unnest operation doesn't use LLM


def test_dict_unnest_operation_empty_input(
    dict_unnest_config, default_model, max_threads
):
    operation = UnnestOperation(dict_unnest_config, default_model, max_threads)
    results, cost = operation.execute([])

    assert len(results) == 0
    assert cost == 0


def test_unnest_operation(
    unnest_config, default_model, max_threads, unnest_sample_data
):
    operation = UnnestOperation(unnest_config, default_model, max_threads)
    results, cost = operation.execute(unnest_sample_data)

    assert len(results) == 6  # 3 + 2 + 1
    assert all("tag" in result for result in results)
    assert cost == 0  # Unnest operation doesn't use LLM


def test_unnest_operation_empty_input(unnest_config, default_model, max_threads):
    operation = UnnestOperation(unnest_config, default_model, max_threads)
    results, cost = operation.execute([])

    assert len(results) == 0
    assert cost == 0


# Equijoin Operation Tests
@pytest.fixture
def equijoin_config():
    return {
        "type": "equijoin",
        "blocking_keys": {"left": ["id"], "right": ["user_id"]},
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
        "method": "token_count",
        "method_kwargs": {"token_count": 4},
        "name": "split_doc",
    }


@pytest.fixture
def gather_config():
    return {
        "type": "gather",
        "content_key": "content_chunk",
        "doc_id_key": "split_doc_id",
        "order_key": "split_doc_chunk_num",
        "peripheral_chunks": {
            "previous": {
                "head": {"content_key": "content_chunk", "count": 1},
                "middle": {"content_key": "content_chunk"},
                "tail": {"content_key": "content_chunk", "count": 1},
            },
            "next": {
                "head": {"content_key": "content_chunk", "count": 1},
                "tail": {"content_key": "content_chunk", "count": 1},
            },
        },
    }


@pytest.fixture
def sample_data():
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


def test_split_operation(split_config, default_model, max_threads, sample_data):
    operation = SplitOperation(split_config, default_model, max_threads)
    results, cost = operation.execute(sample_data)

    assert len(results) > len(sample_data)
    assert all(
        "split_doc_chunk_num" in result and "content_chunk" in result
        for result in results
    )
    assert all("split_doc_id" in result for result in results)
    assert cost == 0  # No LLM calls in split operation

    # Check that chunks are created correctly
    assert len(results) == 12

    # Check the structure of the first chunk
    first_chunk = results[0]
    assert first_chunk["split_doc_chunk_num"] == 1
    assert first_chunk["content_chunk"].startswith("This is a long")
    assert "split_doc_id" in first_chunk

    # Check the structure of the last chunk of the first document
    last_chunk_first_doc = results[4]
    assert last_chunk_first_doc["split_doc_chunk_num"] == 5
    assert last_chunk_first_doc["split_doc_id"] == results[0]["split_doc_id"]

    # Check the first chunk of the second document
    first_chunk_second_doc = results[8]
    assert first_chunk_second_doc["split_doc_chunk_num"] == 1
    assert first_chunk_second_doc["split_doc_id"] != results[0]["split_doc_id"]


def test_gather_operation(
    split_config, gather_config, default_model, max_threads, sample_data
):
    # First, split the data
    split_op = SplitOperation(split_config, default_model, max_threads)
    split_results, _ = split_op.execute(sample_data)

    # Now, gather the split results
    gather_op = GatherOperation(gather_config, default_model, max_threads)
    results, cost = gather_op.execute(split_results)

    assert len(results) == len(split_results)
    assert all("content_chunk_rendered" in result for result in results)
    assert cost == 0  # No LLM calls in gather operation

    # Check the structure of a gathered chunk
    middle_chunk = results[2]  # Third chunk of the first document
    formatted_content = middle_chunk["content_chunk_rendered"]

    assert "--- Previous Context ---" in formatted_content
    assert "--- Next Context ---" in formatted_content
    assert "--- Begin Main Chunk ---" in formatted_content
    assert "--- End Main Chunk ---" in formatted_content
    assert "characters skipped" in formatted_content


def test_split_gather_combined(
    split_config, gather_config, default_model, max_threads, sample_data
):
    split_op = SplitOperation(split_config, default_model, max_threads)
    gather_op = GatherOperation(gather_config, default_model, max_threads)

    split_results, split_cost = split_op.execute(sample_data)
    gather_results, gather_cost = gather_op.execute(split_results)

    assert len(gather_results) == len(split_results)
    assert all("content_chunk_rendered" in result for result in gather_results)
    assert split_cost == 0 and gather_cost == 0  # No LLM calls in either operation

    # Check that the gather operation preserved all split operation fields
    for split_result, gather_result in zip(split_results, gather_results):
        for key in split_result:
            assert key in gather_result
            if key != "content_chunk_rendered":
                assert gather_result[key] == split_result[key]


def test_split_gather_empty_input(
    split_config, gather_config, default_model, max_threads
):
    split_op = SplitOperation(split_config, default_model, max_threads)
    gather_op = GatherOperation(gather_config, default_model, max_threads)

    split_results, split_cost = split_op.execute([])
    assert len(split_results) == 0


# Reduce Operation Tests
@pytest.fixture
def reduce_config():
    return {
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


def test_reduce_operation_with_list_key(
    reduce_config, default_model, max_threads, reduce_sample_data_with_list_key
):
    reduce_config["reduce_key"] = ["group", "category"]

    operation = ReduceOperation(reduce_config, default_model, max_threads)
    results, cost = operation.execute(reduce_sample_data_with_list_key)

    assert len(results) == 3  # 3 unique groups
    assert all(
        "group" in result
        and "category" in result
        and "total" in result
        and "avg" in result
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
