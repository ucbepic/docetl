import pytest
from motion.operations.split import SplitOperation


@pytest.fixture
def default_model():
    return "gpt-4o-mini"


@pytest.fixture
def split_config():
    return {
        "split_key": "content",
        "chunk_size": 10,
        "peripheral_chunks": {
            "previous": {
                "head": {"count": 1, "type": "summary"},
                "middle": {"type": "summary"},
            },
        },
        "summary_prompt": "Summarize the following text:\n\n{{chunk_content}}\n\nSummary:",
        "summary_model": "gpt-4o-mini",
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


def test_split_operation_with_summary(split_config, input_data, default_model):
    split_op = SplitOperation(split_config, default_model, 30)

    results, cost = split_op.execute(input_data)

    assert len(results) > len(
        input_data
    ), "Split operation should produce more chunks than input items"
    assert cost > 0, "Operation cost should be greater than zero"

    for result in results:
        assert "chunk_id" in result, "Each result should have a chunk_id"
        assert "chunk_content" in result, "Each result should have chunk_content"
        assert (
            "_chunk_intermediates" in result
        ), "Each result should have _chunk_intermediates"

        intermediates = result["_chunk_intermediates"]
        assert (
            "previous_chunks" in intermediates
        ), "Intermediates should include previous_chunks"
        assert (
            "next_chunks" in intermediates
        ), "Intermediates should include next_chunks"

        for chunk in intermediates["previous_chunks"]:
            assert "summary" in chunk, "Peripheral chunks should include summaries"
            assert (
                len(chunk["summary"]) > 10
            ), "Summary should be longer than 10 characters"
