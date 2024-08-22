import pytest
import json
import tempfile
import os
from motion.builder import Optimizer


@pytest.fixture
def sample_data():
    return [
        {"id": 1, "text": "Patient reports taking aspirin daily."},
        {"id": 2, "text": "Patient is on lisinopril for blood pressure."},
        {"id": 3, "text": "Patient is prescribed omeprazole for acid reflux."},
        {"id": 4, "text": "Patient takes metformin for diabetes."},
        {"id": 5, "text": "Patient uses albuterol inhaler as needed."},
        {"id": 6, "text": "Patient is prescribed warfarin for blood thinning."},
        {"id": 7, "text": "Patient takes levothyroxine for thyroid issues."},
        {"id": 8, "text": "Patient uses insulin injections to manage diabetes."},
        {"id": 9, "text": "Patient is on amlodipine for blood pressure."},
        {"id": 10, "text": "Patient reports taking ibuprofen for pain relief."},
        {"id": 11, "text": "Patient uses a fluticasone nasal spray for allergies."},
        {"id": 12, "text": "Patient takes metformin for diabetes."},
        {"id": 13, "text": "Patient is prescribed atorvastatin for high cholesterol."},
        {"id": 14, "text": "Patient is taking sertraline for depression."},
        {"id": 15, "text": "Patient uses a budesonide inhaler for asthma control."},
    ]


@pytest.fixture
def config_yaml(sample_data):
    with tempfile.NamedTemporaryFile(
        mode="w+", suffix=".yaml", delete=False
    ) as temp_file:
        config = {
            "datasets": {
                "patient_records": {"type": "file", "path": "patient_records.json"}
            },
            "default_model": "gpt-4o-mini",
            "operations": {
                "extract_medications": {
                    "type": "map",
                    "optimize": False,
                    "output": {"schema": {"medication": "string"}},
                    "prompt": "Extract the first medication from the following text:\n\n{{ input.text }}\n\nReturn the medication.",
                },
                "summarize_medications": {
                    "type": "reduce",
                    "optimize": False,
                    "reduce_key": "medication",
                    "output": {"schema": {"summary": "string"}},
                    "prompt": "Summarize the usage of the medication '{{ reduce_key }}' based on the following contexts:\n\n{% for item in values %}{{ item.text }}\n{% endfor %}\n\nProvide a brief summary of how this medication is typically used.",
                },
            },
            "pipeline": {
                "steps": [
                    {
                        "name": "medication_analysis",
                        "input": "patient_records",
                        "operations": ["extract_medications", "summarize_medications"],
                    }
                ],
                "output": {"type": "file", "path": "output.json"},
            },
        }
        json.dump(config, temp_file)
        temp_file.flush()

        # Create sample data file
        with open("patient_records.json", "w") as f:
            json.dump(sample_data, f)

        return temp_file.name


def test_synth_resolve(config_yaml):
    # Initialize the optimizer
    optimizer = Optimizer(config_yaml)

    # Run the optimization
    optimizer.optimize()

    # Check if a resolve operation was synthesized
    synthesized_resolve_found = False
    for step in optimizer.optimized_config["pipeline"]["steps"]:
        for op in step["operations"]:
            if op.startswith("synthesized_resolve_"):
                synthesized_resolve_found = True
                synthesized_op = optimizer.optimized_config["operations"][op]

                # Check if the synthesized operation has the correct properties
                assert synthesized_op["type"] == "resolve"
                assert "embedding_model" in synthesized_op
                assert "resolution_model" in synthesized_op
                assert "comparison_model" in synthesized_op
                assert "_intermediates" in synthesized_op
                assert "map_prompt" in synthesized_op["_intermediates"]
                assert "reduce_key" in synthesized_op["_intermediates"]
                assert "comparison_prompt" in synthesized_op
                assert "resolution_prompt" in synthesized_op

                break
        if synthesized_resolve_found:
            break

    assert (
        synthesized_resolve_found
    ), "No synthesized resolve operation found in the optimized config"

    # Clean up temporary files
    os.remove(config_yaml)
    os.remove(optimizer.optimized_config_path)
    os.remove("patient_records.json")


# Run the test
if __name__ == "__main__":
    sd = sample_data()
    config = config_yaml(sd)
    test_synth_resolve(config)