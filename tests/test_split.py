import pytest
from motion.operations.split import SplitOperation
from motion.operations.map import MapOperation
from motion.operations.gather import GatherOperation


@pytest.fixture
def default_model():
    return "gpt-4o-mini"


@pytest.fixture
def max_threads():
    return 4


@pytest.fixture
def split_config():
    return {
        "type": "split",
        "split_key": "content",
        "chunk_size": 10,
        "name": "split_doc",
    }


@pytest.fixture
def map_config():
    return {
        "type": "map",
        "prompt": "Summarize the following text:\n\n{{input.content_chunk}}\n\nSummary:",
        "output": {"schema": {"summary": "string"}},
        "model": "gpt-4o-mini",
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
                "head": {"content_key": "summary", "count": 1},
                "middle": {"content_key": "summary"},
                "tail": {"content_key": "content_chunk", "count": 1},
            },
            "next": {
                "head": {"content_key": "content_chunk", "count": 1},
            },
        },
    }


@pytest.fixture
def input_data():
    return [
        {
            "id": "1",
            "content": """
            Machine learning is a subset of artificial intelligence that focuses on the development of algorithms and statistical models that enable computer systems to improve their performance on a specific task through experience. Instead of explicitly programming rules, machine learning allows systems to learn patterns from data and make decisions with minimal human intervention.

            There are several types of machine learning:

            1. Supervised Learning: The algorithm learns from labeled training data, trying to minimize the error between its predictions and the actual labels.

            2. Unsupervised Learning: The algorithm tries to find patterns in unlabeled data without predefined categories or labels.

            3. Semi-supervised Learning: This approach uses both labeled and unlabeled data for training.

            4. Reinforcement Learning: The algorithm learns by interacting with an environment, receiving feedback in the form of rewards or penalties.

            Machine learning has numerous applications across various fields, including:

            - Image and speech recognition
            - Natural language processing
            - Recommendation systems
            - Fraud detection
            - Autonomous vehicles
            - Medical diagnosis
            - Financial market analysis

            As the field continues to evolve, new techniques and applications are constantly emerging, pushing the boundaries of what's possible with artificial intelligence.
            """,
        },
    ]


def test_split_map_gather_operations(
    split_config, map_config, gather_config, input_data, default_model, max_threads
):
    split_op = SplitOperation(split_config, default_model, max_threads)
    map_op = MapOperation(map_config, default_model, max_threads)
    gather_op = GatherOperation(gather_config, default_model, max_threads)

    # Execute split operation
    split_results, split_cost = split_op.execute(input_data)
    assert len(split_results) > len(
        input_data
    ), "Split operation should produce more chunks than input items"
    assert split_cost == 0, "Split operation cost should be zero"

    for result in split_results:
        assert (
            "split_doc_chunk_num" in result
        ), "Each result should have a split_doc_chunk_num"
        assert "content_chunk" in result, "Each result should have content_chunk"
        assert "split_doc_id" in result, "Each result should have split_doc_id"

    # Execute map operation for summarization
    map_results, map_cost = map_op.execute(split_results)
    assert len(map_results) == len(
        split_results
    ), "Map operation should produce same number of results as split operation"
    assert map_cost > 0, "Map operation cost should be greater than zero"

    for result in map_results:
        assert "summary" in result, "Each result should have a summary"
        assert (
            len(result["summary"]) > 10
        ), "Summary should be longer than 10 characters"

    # Execute gather operation
    gather_results, gather_cost = gather_op.execute(map_results)
    assert len(gather_results) == len(
        map_results
    ), "Gather operation should produce same number of results as map operation"
    assert gather_cost == 0, "Gather operation cost should be zero"

    for idx, result in enumerate(gather_results):
        assert (
            "content_chunk_formatted" in result
        ), "Each result should have content_chunk_formatted"
        formatted_content = result["content_chunk_formatted"]

        assert (
            "--- Previous Context ---" in formatted_content
        ), "Formatted content should include previous context"
        assert (
            "--- Next Context ---" in formatted_content
        ), "Formatted content should include next context"
        assert (
            "--- Begin Main Chunk ---" in formatted_content
        ), "Formatted content should include main chunk delimiters"
        assert (
            "--- End Main Chunk ---" in formatted_content
        ), "Formatted content should include main chunk delimiters"
        if idx < 23:
            assert (
                "characters skipped" in formatted_content
            ), "Formatted content should include skipped characters information"

    # Check that the gather operation preserved all split and map operation fields
    for split_result, map_result, gather_result in zip(
        split_results, map_results, gather_results
    ):
        for key in split_result:
            assert (
                key in gather_result
            ), f"Gather result should preserve split result key: {key}"
        for key in map_result:
            assert (
                key in gather_result
            ), f"Gather result should preserve map result key: {key}"


def test_split_map_gather_empty_input(
    split_config, map_config, gather_config, default_model, max_threads
):
    split_op = SplitOperation(split_config, default_model, max_threads)
    map_op = MapOperation(map_config, default_model, max_threads)
    gather_op = GatherOperation(gather_config, default_model, max_threads)

    split_results, split_cost = split_op.execute([])
    assert len(split_results) == 0
    assert split_cost == 0

    map_results, map_cost = map_op.execute(split_results)
    assert len(map_results) == 0
    assert map_cost == 0

    gather_results, gather_cost = gather_op.execute(map_results)
    assert len(gather_results) == 0
    assert gather_cost == 0