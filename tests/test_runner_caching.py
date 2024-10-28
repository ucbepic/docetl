import shutil
import time
import pytest
import json
import tempfile
import os
from docetl.api import Pipeline, Dataset, MapOp, PipelineStep, PipelineOutput


@pytest.fixture
def temp_input_file():
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as tmp:
        json.dump(
            [
                {"text": "This is a test sentence."},
                {"text": "Another test sentence."},
            ],
            tmp,
        )
    yield tmp.name
    os.unlink(tmp.name)


@pytest.fixture
def temp_output_file():
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
        pass
    yield tmp.name
    os.unlink(tmp.name)


@pytest.fixture
def temp_intermediate_dir():
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield tmpdirname


def create_pipeline(input_file, output_file, intermediate_dir, operation_prompt):
    return Pipeline(
        name="test_pipeline",
        datasets={"test_input": Dataset(type="file", path=input_file)},
        operations=[
            MapOp(
                name="test_operation",
                type="map",
                prompt=operation_prompt,
                output={"schema": {"result": "string"}},
            )
        ],
        steps=[
            PipelineStep(
                name="test_step", input="test_input", operations=["test_operation"]
            ),
        ],
        output=PipelineOutput(
            type="file", path=output_file, intermediate_dir=intermediate_dir
        ),
        default_model="gpt-4o-mini",
    )


@pytest.mark.flaky(reruns=3)
def test_pipeline_rerun_on_operation_change(
    temp_input_file, temp_output_file, temp_intermediate_dir
):
    # Initial run
    initial_prompt = "Analyze the sentiment of the following text: '{{ input.text }}'"
    pipeline = create_pipeline(
        temp_input_file, temp_output_file, temp_intermediate_dir, initial_prompt
    )
    initial_cost = pipeline.run()

    # Check that intermediate files were created
    assert os.path.exists(
        os.path.join(temp_intermediate_dir, "test_step", "test_operation.json")
    )

    # Run without modifying the operation
    unmodified_cost = pipeline.run()

    # Check that the pipeline was not rerun (cost should be zero)
    assert unmodified_cost == 0

    # Record the start time
    start_time = time.time()

    # Run again without changes
    _ = pipeline.run()

    # Record the end time
    end_time = time.time()

    # Calculate and store the runtime
    unmodified_runtime = end_time - start_time

    # Modify the operation
    modified_prompt = "Count the words in the following text: '{{ input.text }}'"
    modified_pipeline = create_pipeline(
        temp_input_file, temp_output_file, temp_intermediate_dir, modified_prompt
    )

    # Record the start time
    start_time = time.time()

    _ = modified_pipeline.run()

    # Record the end time
    end_time = time.time()

    # Calculate and store the runtime
    modified_runtime = end_time - start_time

    # Check that the intermediate files were updated
    with open(
        os.path.join(temp_intermediate_dir, "test_step", "test_operation.json"), "r"
    ) as f:
        intermediate_data = json.load(f)
    assert any("word" in str(item).lower() for item in intermediate_data)

    # Check that the runtime is faster when not modifying
    assert unmodified_runtime < modified_runtime


# Test with an incorrect later operation but correct earlier operation
def test_partial_caching(temp_input_file, temp_output_file, temp_intermediate_dir):
    # Create initial pipeline with two operations
    initial_pipeline = Pipeline(
        name="test_pipeline",
        datasets={"test_input": Dataset(type="file", path=temp_input_file)},
        operations=[
            MapOp(
                name="first_operation",
                type="map",
                prompt="Analyze the sentiment of the following text: '{{ input.text }}'",
                output={"schema": {"sentiment": "string"}},
                model="gpt-4o-mini",
            ),
            MapOp(
                name="second_operation_bad",
                type="map",
                prompt="Summarize the following text: '{{ forororororo }}'",
                output={"schema": {"summary": "1000"}},
                model="gpt-4o-mini",
            ),
        ],
        steps=[
            PipelineStep(
                name="first_step",
                input="test_input",
                operations=["first_operation", "second_operation_bad"],
            ),
        ],
        output=PipelineOutput(
            type="file", path=temp_output_file, intermediate_dir=temp_intermediate_dir
        ),
        default_model="gpt-4o-mini",
    )

    # Run the initial pipeline
    # Run the initial pipeline with an expected error
    with pytest.raises(Exception):
        initial_cost = initial_pipeline.run()

    new_pipeline_with_only_one_op = Pipeline(
        name="test_pipeline",
        datasets={"test_input": Dataset(type="file", path=temp_input_file)},
        operations=[
            MapOp(
                name="first_operation",
                type="map",
                prompt="Analyze the sentiment of the following text: '{{ input.text }}'",
                output={"schema": {"sentiment": "string"}},
                model="gpt-4o-mini",
            ),
        ],
        steps=[
            PipelineStep(
                name="first_step",
                input="test_input",
                operations=["first_operation"],
            ),
        ],
        output=PipelineOutput(
            type="file", path=temp_output_file, intermediate_dir=temp_intermediate_dir
        ),
        default_model="gpt-4o-mini",
    )
    rerun_cost = new_pipeline_with_only_one_op.run()

    # Assert that the cost was 0 when rerunning the pipeline
    assert (
        rerun_cost == 0
    ), "Expected zero cost when rerunning the pipeline without changes"
