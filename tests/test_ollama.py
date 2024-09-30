import shutil
import pytest
import json
import tempfile
import os
from docetl.api import (
    Pipeline,
    Dataset,
    MapOp,
    ReduceOp,
    PipelineStep,
    PipelineOutput,
)
from dotenv import load_dotenv

load_dotenv()

# Set the OLLAMA_API_BASE environment variable
os.environ["OLLAMA_API_BASE"] = "http://localhost:11434/"


@pytest.fixture
def temp_input_file():
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as tmp:
        json.dump(
            [
                {"text": "This is a test", "group": "A"},
                {"text": "Another test", "group": "B"},
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


@pytest.fixture
def map_config():
    return MapOp(
        name="sentiment_analysis",
        type="map",
        prompt="Analyze the sentiment of the following text: '{{ input.text }}'. Classify it as either positive, negative, or neutral.",
        output={"schema": {"sentiment": "string"}},
        model="ollama/llama3.1",
    )


@pytest.fixture
def reduce_config():
    return ReduceOp(
        name="group_summary",
        type="reduce",
        reduce_key="group",
        prompt="Summarize the following group of values: {{ inputs }} Provide a total and any other relevant statistics.",
        output={"schema": {"total": "number", "avg": "number"}},
        model="ollama/llama3.1",
    )


@pytest.fixture(autouse=True)
def remove_openai_api_key():
    openai_api_key = os.environ.pop("OPENAI_API_KEY", None)
    yield
    if openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key


def test_ollama_map_reduce_pipeline(
    map_config, reduce_config, temp_input_file, temp_output_file, temp_intermediate_dir
):
    pipeline = Pipeline(
        name="test_ollama_pipeline",
        datasets={"test_input": Dataset(type="file", path=temp_input_file)},
        operations=[map_config, reduce_config],
        steps=[
            PipelineStep(
                name="pipeline",
                input="test_input",
                operations=["sentiment_analysis", "group_summary"],
            ),
        ],
        output=PipelineOutput(
            type="file", path=temp_output_file, intermediate_dir=temp_intermediate_dir
        ),
        default_model="ollama/llama3.1",
    )

    cost = pipeline.run()

    assert isinstance(cost, float)
    assert cost == 0

    # Verify output file exists and contains data
    assert os.path.exists(temp_output_file)
    with open(temp_output_file, "r") as f:
        output_data = json.load(f)
    assert len(output_data) > 0

    # Clean up
    shutil.rmtree(temp_intermediate_dir)
