import pytest
import json
import os
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
      Analyze the sentiment of the following text: "{{ input.text }}"
      Classify it as either positive, negative, or neutral.
    output:
      schema:
        sentiment: string
    model: "gpt-4o-mini"

  - name: filter_operation
    type: filter
    prompt: |
      Determine if the following text is longer than 5 words:
      "{{ input.text }}"
    output:
      schema:
        keep: boolean
    model: "gpt-4o-mini"

datasets:
  sample_dataset:
    type: file
    path: "tests/sample_data.json"

pipeline:
  steps:
    - name: sentiment_analysis
      input: sample_dataset
      operations:
        - map_operation
    - name: filter_long_texts
      input: sentiment_analysis
      operations:
        - filter_operation

  output:
    type: file
    path: "tests/output.json"
"""

SAMPLE_DATA = [
    {"text": "This is a very positive sentence.", "id": 1},
    {"text": "A short negative phrase.", "id": 2},
    {"text": "Neutral statement without much edocetl.", "id": 3},
    {"text": "Brief.", "id": 4},
]


@pytest.fixture
def config_file(tmp_path):
    config_path = tmp_path / "test_config.yaml"
    with open(config_path, "w") as f:
        f.write(SAMPLE_CONFIG)
    return config_path


@pytest.fixture
def sample_data_file(tmp_path):
    data_path = tmp_path / "sample_data.json"
    with open(data_path, "w") as f:
        json.dump(SAMPLE_DATA, f)
    return data_path


def test_end_to_end_pipeline(config_file, sample_data_file, tmp_path):
    # Update the config with the correct sample data path
    config = load_config(config_file)
    config["datasets"]["sample_dataset"]["path"] = str(sample_data_file)
    config["pipeline"]["output"]["path"] = str(tmp_path / "output.json")

    # Write the updated config back to the file
    with open(config_file, "w") as f:
        yaml.dump(config, f)

    # Create and run the DSLRunner
    runner = DSLRunner.from_yaml(str(config_file))
    total_cost = runner.load_run_save()

    # Check if the output file was created
    output_path = tmp_path / "output.json"
    assert output_path.exists(), "Output file was not created"

    # Load and check the output
    with open(output_path, "r") as f:
        output_data = json.load(f)

    # Verify the output
    assert len(output_data) > 0, "Output data is empty"
    assert all(
        "sentiment" in item for item in output_data
    ), "Sentiment analysis was not applied to all items"
    assert all(
        len(item["text"].split()) >= 5 for item in output_data
    ), "Filter operation did not remove short texts"

    # Check if the cost was calculated and is greater than 0
    assert total_cost > 0, "Total cost was not calculated or is 0"

    print(f"Pipeline executed successfully. Total cost: ${total_cost:.2f}")
    print(f"Output: {output_data}")
