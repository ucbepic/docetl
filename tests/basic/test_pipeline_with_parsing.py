from typing import Dict, List
import pytest
import json
import os
import tempfile
from docetl.runner import DSLRunner
from docetl.utils import load_config
import yaml
from docetl.api import (
    Pipeline,
    Dataset,
    MapOp,
    PipelineOutput,
    PipelineStep,
)

# Sample configuration for the test
SAMPLE_CONFIG = """
default_model: "gpt-4o-mini"

operations:
  - name: map_operation
    type: map
    bypass_cache: true
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
    parsing:
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
            total_cost = runner.load_run_save()

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

            # Assert that each output has at least 40 characters
            for item in output_data:
                assert len(item["summary"]) >= 40, "Summary is not long enough"

        # Clean up the output file
        os.unlink(output_file.name)

    # Clean up the sample data file
    os.remove(sample_data_file.name)


def custom_exploder(doc: Dict) -> List[Dict]:
    text = doc["text"]
    return [{"text": t} for t in text]


def test_pipeline_with_custom_parsing():
    # Create a temporary input file
    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, suffix=".json"
    ) as tmp_input:
        json.dump(
            [
                {"text": "This is a test sentence.", "label": "test"},
                {"text": "Another test sentence.", "label": "test"},
            ],
            tmp_input,
        )

    # Create a temporary output file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp_output:
        pass

    # Define the pipeline
    pipeline = Pipeline(
        name="test_parsing_pipeline",
        datasets={
            "input_data": Dataset(
                type="file",
                path=tmp_input.name,
                parsing=[
                    {
                        "function": "custom_exploder",
                    }
                ],
            )
        },
        operations=[
            MapOp(
                name="summarize",
                type="map",
                prompt="Summarize the following text: {{ input.text }}",
                output={"schema": {"summary": "string"}},
            )
        ],
        steps=[
            PipelineStep(
                name="summarize_step", input="input_data", operations=["summarize"]
            )
        ],
        output=PipelineOutput(type="file", path=tmp_output.name),
        parsing_tools=[custom_exploder],
        default_model="gpt-4o-mini",
    )

    # Run the pipeline
    cost = pipeline.run(max_threads=4)

    # Verify the output
    with open(tmp_output.name, "r") as f:
        output_data = json.load(f)

    assert len(output_data) == 46, "Expected 10 output items"
    for item in output_data:
        assert "summary" in item, "Summary was not generated"
        assert isinstance(item["summary"], str), "Summary is not a string"
        assert len(item["summary"]) > 0, "Summary is empty"

    # Clean up
    os.unlink(tmp_input.name)
    os.unlink(tmp_output.name)

    print(
        f"Pipeline with custom parsing executed successfully. Total cost: ${cost:.2f}"
    )
