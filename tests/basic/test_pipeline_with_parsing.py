import pytest
import json
import os
import tempfile
from docetl.runner import DSLRunner
from docetl.utils import load_config
import yaml

# Sample configuration for the test
SAMPLE_CONFIG = """
default_model: "gpt-4o-mini"

operations:
  - name: map_operation
    type: map
    prompt: |
      Summarize the following text in one sentence: "{{ input.content }}"
    output:
      schema:
        summary: string
    model: "gpt-4o-mini"

datasets:
  sample_dataset:
    type: file
    source: local
    path: "tests/sample_data.json"
    parsing_tools:
      - input_key: text_file_path
        function: txt_to_string
        output_key: content

pipeline:
  steps:
    - name: summarize_text
      input: sample_dataset
      operations:
        - map_operation

  output:
    type: file
    path: "tests/output.json"
"""

SAMPLE_JSON_DATA = [
    {"id": 1, "text_file_path": "tests/basic/sample_texts/one.txt"},
    {"id": 2, "text_file_path": "tests/basic/sample_texts/two.md"},
]

# Read sample text content from files
with open("tests/basic/sample_texts/one.txt", "r") as f:
    SAMPLE_TEXT_CONTENT_ONE = f.read()

with open("tests/basic/sample_texts/two.md", "r") as f:
    SAMPLE_TEXT_CONTENT_TWO = f.read()


@pytest.fixture
def config_file():
    with tempfile.NamedTemporaryFile(
        mode="w+", suffix=".yaml", delete=False
    ) as temp_file:
        temp_file.write(SAMPLE_CONFIG)
        temp_file.flush()
        yield temp_file.name
    os.unlink(temp_file.name)


def test_pipeline_with_parsing(config_file):
    # Update the config with the correct sample data path
    config = load_config(config_file)

    with tempfile.NamedTemporaryFile(
        mode="w+", suffix=".json", delete=False
    ) as sample_data_file:
        json.dump(SAMPLE_JSON_DATA, sample_data_file)
        sample_data_file.flush()
        config["datasets"]["sample_dataset"]["path"] = sample_data_file.name

        with tempfile.NamedTemporaryFile(
            mode="w+", suffix=".json", delete=False
        ) as output_file:
            config["pipeline"]["output"]["path"] = output_file.name

            # Write the updated config back to the file
            with open(config_file, "w") as f:
                yaml.dump(config, f)

            # Create and run the DSLRunner
            runner = DSLRunner.from_yaml(config_file)
            total_cost = runner.run()

            # Check if the output file was created
            assert os.path.exists(output_file.name), "Output file was not created"

            # Load and check the output
            with open(output_file.name, "r") as f:
                output_data = json.load(f)

            # Verify the output
            assert len(output_data) == len(
                SAMPLE_JSON_DATA
            ), f"Expected {len(SAMPLE_JSON_DATA)} output items"
            for item in output_data:
                assert "summary" in item, "Summary was not generated"
                assert isinstance(item["summary"], str), "Summary is not a string"

            # Check if the cost was calculated and is greater than 0
            assert total_cost > 0, "Total cost was not calculated or is 0"

            print(f"Pipeline executed successfully. Total cost: ${total_cost:.2f}")

            # Assert that each output has at least 50 characters
            for item in output_data:
                assert len(item["summary"]) >= 50, "Summary is not long enough"

        # Clean up the output file
        os.unlink(output_file.name)

    # Clean up the sample data file
    os.remove(sample_data_file.name)
