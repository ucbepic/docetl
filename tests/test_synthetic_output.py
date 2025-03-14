import pytest
import json
import os
import tempfile
from pathlib import Path

from docetl.runner import DSLRunner
from tests.conftest import api_wrapper


@pytest.fixture
def synthetic_fruits_data():
    """Create synthetic fruits data for testing"""
    return [
        {"fruit": "apple"},
        {"fruit": "banana"},
        {"fruit": "orange"},
        {"fruit": "strawberry"},
        {"fruit": "kiwi"}
    ]


@pytest.fixture
def config_yaml(synthetic_fruits_data):
    """Create temporary YAML configuration and data files for testing"""
    with tempfile.NamedTemporaryFile(
        mode="w+", suffix=".yaml", delete=False
    ) as config_file, tempfile.NamedTemporaryFile(
        mode="w+", suffix=".json", delete=False
    ) as fruits_file, tempfile.NamedTemporaryFile(
        mode="w+", suffix=".json", delete=False
    ) as output_file:
        
        # Create the configuration
        config = {
            "datasets": {
                "fruits": {"type": "file", "path": fruits_file.name}
            },
            "default_model": "gpt-4o-mini",
            "operations": [
                {
                    "name": "gen_stories",
                    "type": "map",
                    "bypass_cache": True,
                    "optimize": True,
                    "output": {
                        "n": 3,  # Use a smaller n for faster test execution
                        "schema": {
                            "story": "str"
                        }
                    },
                    "prompt": (
                        "Create a short, imaginative children's story featuring the following fruit as the main character:\n\n"
                        "{{ input.fruit }}\n\n"
                        "The story should:\n"
                        "- Be 3-5 sentences long\n"
                        "- Be appropriate for children ages 4-8\n"
                        "- Have a clear beginning, middle, and end\n"
                        "- Include some dialogue\n"
                        "- Be creative and engaging\n\n"
                        "Your response should only contain the story text, beginning with \"Once upon a time...\""
                    )
                }
            ],
            "pipeline": {
                "steps": [
                    {
                        "name": "gen_stories",
                        "input": "fruits",
                        "operations": ["gen_stories"]
                    }
                ],
                "output": {"type": "file", "path": output_file.name},
            },
        }
        
        # Write the configuration to the config file
        json.dump(config, config_file)
        config_file.flush()
        
        # Write the fruits data to the fruits file
        json.dump(synthetic_fruits_data, fruits_file)
        fruits_file.flush()
        
        return config_file.name, fruits_file.name, output_file.name


@pytest.mark.parametrize("use_runner", [True])
def test_synthetic_output_count(
    synthetic_fruits_data, config_yaml, api_wrapper, use_runner
):
    """
    Test that ensures the synthetic workload generates the expected number of outputs.
    The expected count should be: number_of_inputs * n (from output config)
    """
    config_path, fruits_path, output_path = config_yaml
    
    try:
        # Count the number of inputs
        num_inputs = len(synthetic_fruits_data)
        
        # Create a runner with the YAML file
        runner = DSLRunner.from_yaml(config_path)
        
        # Get the value of n from the output config
        output_n = runner.config["operations"][0]["output"]["n"]
        
        # Run the workload
        total_cost = runner.load_run_save()
        
        # Verify the output file exists
        assert os.path.exists(output_path), f"Output file {output_path} does not exist"
        
        # Load the output file
        with open(output_path, "r") as f:
            output_data = json.load(f)
        
        # Calculate the expected number of outputs
        expected_count = num_inputs * output_n
        actual_count = len(output_data)
        
        # Verify the output count matches the expected count
        assert actual_count == expected_count, (
            f"Expected {expected_count} outputs "
            f"({num_inputs} inputs × {output_n} stories per input), "
            f"but got {actual_count}"
        )
        
        # Verify each output has the expected structure
        for item in output_data:
            assert "story" in item, "Output item missing 'story' field"
            assert isinstance(item["story"], str), "'story' field should be a string"
            assert item["story"].startswith("Once upon a time"), (
                "Story should start with 'Once upon a time'"
            )
    finally:
        # Clean up temporary files
        for path in [config_path, fruits_path, output_path]:
            if os.path.exists(path):
                os.remove(path)


def test_synthetic_output_memory():
    """
    Test with in-memory datasets and outputs.
    
    This is a separate test to show how to use in-memory data instead of files,
    but it's commented out since the current DSLRunner API may not support this directly.
    """
    # This would require modifications to the DSLRunner API to support in-memory datasets
    # and outputs, so we'll leave this test for future implementation.
    pass 