import pytest
import json
import tempfile
import os

from docetl.runner import DSLRunner
from docetl.operations.split import SplitOperation
from docetl.operations.map import MapOperation
from docetl.operations.gather import GatherOperation
from tests.conftest import api_wrapper


def generate_random_content(length):
    import random

    words = [
        "apple",
        "banana",
        "cherry",
        "date",
        "elderberry",
        "fig",
        "grape",
        "honeydew",
        "kiwi",
        "lemon",
        "mango",
        "nectarine",
        "orange",
        "papaya",
        "quince",
        "raspberry",
        "strawberry",
        "tangerine",
        "ugli fruit",
        "watermelon",
    ]
    return " ".join(random.choices(words, k=length))


@pytest.fixture
def sample_data():
    documents = []
    for i in range(5):
        document = f"# Document {i+1}\n\n"
        document += generate_random_content(100) + "\n\n"
        for j in range(3):
            document += f"## Section {j+1}\n\n"
            document += generate_random_content(60) + "\n\n"
            for k in range(2):
                document += f"### Subsection {k+1}\n\n"
                document += generate_random_content(40) + "\n\n"

        documents.append({"id": i + 1, "content": document})
    return documents


@pytest.fixture
def config_yaml(sample_data):
    with tempfile.NamedTemporaryFile(
        mode="w+", suffix=".yaml", delete=False
    ) as temp_file, tempfile.NamedTemporaryFile(
        mode="w+", suffix=".json", delete=False
    ) as long_documents_file, tempfile.NamedTemporaryFile(
        mode="w+", suffix=".json", delete=False
    ) as output_file:
        config = {
            "datasets": {
                "long_documents": {"type": "file", "path": long_documents_file.name}
            },
            "default_model": "gpt-4o-mini",
            "operations": [
                {
                    "name": "count_words",
                    "type": "map",
                    "optimize": True,
                    "recursively_optimize": False,
                    "output": {"schema": {"count": "integer"}},
                    "prompt": "Count the number of words that start with the letter 'a' in the following text:\n\n{{ input.content }}\n\nReturn only the count as an integer.",
                }
            ],
            "pipeline": {
                "steps": [
                    {
                        "name": "word_analysis",
                        "input": "long_documents",
                        "operations": ["count_words"],
                    }
                ],
                "output": {"type": "file", "path": output_file.name},
            },
        }
        json.dump(config, temp_file)
        temp_file.flush()

        # Create sample data file
        json.dump(sample_data, long_documents_file)
        long_documents_file.flush()

        return temp_file.name, long_documents_file.name, output_file.name


@pytest.mark.flaky(reruns=2, reruns_delay=1)
def test_synth_gather(config_yaml):
    config_path, long_documents_path, output_path = config_yaml

    # Initialize the optimizer
    runner = DSLRunner.from_yaml(config_path)

    # Run the optimization
    optimized_pipeline, cost = runner.optimize(return_pipeline=True)

    # Check if a gather operation was synthesized
    synthesized_gather_found = False
    for step in optimized_pipeline.config["pipeline"]["steps"]:
        for op in step["operations"]:
            synthesized_op = [
                operation
                for operation in optimized_pipeline.config["operations"]
                if operation["name"] == op
            ][0]
            if synthesized_op.get("type") == "gather":
                synthesized_gather_found = True

                # Check if the synthesized operation has the correct properties
                assert synthesized_op["type"] == "gather"
                assert "content_key" in synthesized_op
                assert "doc_id_key" in synthesized_op
                assert "order_key" in synthesized_op
                assert "peripheral_chunks" in synthesized_op
                assert "doc_header_key" in synthesized_op

                break
        if synthesized_gather_found:
            break

    assert (
        synthesized_gather_found
    ), "No synthesized gather operation found in the optimized config"

    # Run the optimized pipeline
    optimized_pipeline.load_run_save()

    # Check if the output file was created
    assert os.path.exists(output_path), "Output file was not created"

    # Load and check the output
    with open(output_path, "r") as f:
        output = json.load(f)

    with open(long_documents_path, "r") as f:
        sample_data = json.load(f)

    assert len(output) == len(
        sample_data
    ), "Output should have the same number of items as input"
    for item in output:
        assert "count" in item, "Each output item should have a 'count' field"
        assert isinstance(item["count"], int), "The 'count' field should be an integer"

    # Clean up temporary files
    os.remove(config_path)
    os.remove(long_documents_path)
    os.remove(output_path)


# # Run the test
# if __name__ == "__main__":
#     sd = sample_data()
#     config = config_yaml(sd)
#     test_synth_gather(config)


def test_split_map_gather(sample_data, api_wrapper):
    default_model = "gpt-4o-mini"
    # Define split operation
    split_config = {
        "name": "split_doc",
        "type": "split",
        "split_key": "content",
        "method": "token_count",
        "method_kwargs": {"num_tokens": 100},
        "name": "split_doc",
    }

    # Define map operation to extract headers
    map_config = {
        "name": "extract_headers",
        "type": "map",
        "optimize": True,
        "prompt": """Analyze the following chunk of a document and extract any headers you see.

        {{ input.content_chunk }}

        Provide your analysis as a list of dictionaries, where each dictionary contains a 'header' (string) and 'level' (integer). For example:

        [
            {"header": "Document 1", "level": 1},
            {"header": "Section 1", "level": 2}
        ]

        Only include headers you find in the text, do not add any that are not present.""",
        "output": {"schema": {"headers": "list[{header: string, level: integer}]"}},
        "model": default_model,
    }

    # Define gather operation
    gather_config = {
        "name": "gather_doc",
        "type": "gather",
        "content_key": "content_chunk",
        "doc_id_key": "split_doc_id",
        "order_key": "split_doc_chunk_num",
        "peripheral_chunks": {
            "previous": {"tail": {"count": 1}},
            "next": {"head": {"count": 1}},
        },
        "doc_header_key": "headers",
    }

    # Initialize operations
    split_op = SplitOperation(api_wrapper, split_config, default_model, max_threads=64)
    map_op = MapOperation(api_wrapper, map_config, default_model, max_threads=64)
    gather_op = GatherOperation(
        api_wrapper, gather_config, default_model, max_threads=64
    )

    # Execute operations
    split_results, split_cost = split_op.execute(sample_data)
    map_results, map_cost = map_op.execute(split_results)
    gather_results, gather_cost = gather_op.execute(map_results)

    # Assertions
    assert len(gather_results) == len(
        split_results
    ), "Number of gathered results should match split results"

    for result in gather_results:
        assert "headers" in result, "Each gathered result should have a 'headers' field"
        assert isinstance(
            result["headers"], list
        ), "The 'headers' field should be a list"

        for header in result["headers"]:
            assert "header" in header, "Each header should have a 'header' field"
            assert "level" in header, "Each header should have a 'level' field"
            assert isinstance(
                header["header"], str
            ), "The 'header' field should be a string"
            assert isinstance(
                header["level"], int
            ), "The 'level' field should be an integer"

        assert (
            "content_chunk_rendered" in result
        ), "Each result should have content_chunk_rendered"
        formatted_content = result["content_chunk_rendered"]

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

    assert split_cost == 0, "Split operation cost should be zero"
    assert map_cost > 0, "Map operation cost should be greater than zero"
    assert gather_cost == 0, "Gather operation cost should be zero"


# Run the tests
# if __name__ == "__main__":
#     sd = sample_data()
#     config = config_yaml(sd)
#     test_synth_gather(config)
#     test_split_map_gather(sd)
