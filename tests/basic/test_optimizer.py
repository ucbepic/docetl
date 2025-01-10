import os
import pytest
import json
import shutil
from docetl.runner import DSLRunner

@pytest.fixture
def test_dir(tmp_path):
    # Create test directories
    data_dir = tmp_path / "tests" / "data"
    data_dir.mkdir(parents=True)
    
    # Create test data file
    data_file = data_dir / "test_data.json"
    test_data = [
        {"text": "My name is John Smith"},
        {"text": "Hello, I'm Alice Johnson"},
        {"text": "Bob Wilson here"}
    ]
    
    with open(data_file, "w") as f:
        json.dump(test_data, f)
    
    yield tmp_path
    
    # Cleanup
    if tmp_path.exists():
        shutil.rmtree(tmp_path)

@pytest.fixture
def test_config(test_dir):
    return {
        "default_model": "gpt-4o-mini",
        "datasets": {
            "test_data": {
                "type": "file",
                "path": str(test_dir / "tests" / "data" / "test_data.json"),
            }
        },
        "operations": [
            {
                "name": "extract_name",
                "type": "map",
                "prompt": "Extract the person's name from the text.",
                "output": {
                    "schema": {
                        "name": "string"
                    }
                },
                "optimize": True
            }
        ],
        "pipeline": {
            "steps": [
                {
                    "name": "name_extraction",
                    "input": "test_data",
                    "operations": ["extract_name"]
                }
            ]
        }
    }

@pytest.fixture
def runner(test_config):
    return DSLRunner(
        config=test_config
    )

def test_optimize_map_operation(runner, test_dir):
    """Test that the optimizer can optimize a simple map operation"""
    
    
    # Run optimization
    optimized_config, total_cost = runner.optimize(return_pipeline=False)
    
    # Check that optimization completed successfully
    assert total_cost >= 0  # Cost should be non-negative
    
    # Check that the optimized config contains operations
    assert "operations" in optimized_config
    assert len(optimized_config["operations"]) > 0
    
    # Check that the pipeline steps are preserved
    assert "pipeline" in optimized_config
    assert "steps" in optimized_config["pipeline"]
    assert len(optimized_config["pipeline"]["steps"]) > 0
    
    # Check that the first step is preserved
    first_step = optimized_config["pipeline"]["steps"][0]
    assert first_step["name"] == "name_extraction"
    assert "operations" in first_step
    assert len(first_step["operations"]) > 0

