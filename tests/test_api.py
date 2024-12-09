import pytest
import json
import tempfile
import os
from docetl.api import (
    Pipeline,
    Dataset,
    MapOp,
    ReduceOp,
    ParallelMapOp,
    FilterOp,
    PipelineStep,
    PipelineOutput,
    ResolveOp,
    EquijoinOp,
)
from dotenv import load_dotenv

load_dotenv()


@pytest.fixture
def default_model():
    return "gpt-4o-mini"


@pytest.fixture
def max_threads():
    return 4


@pytest.fixture
def temp_input_file():
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as tmp:
        json.dump(
            [
                {"text": "This is a positive sentence.", "group": "A"},
                {"text": "This is a negative sentence.", "group": "B"},
                {"text": "This is a neutral sentence.", "group": "A"},
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
        model="gpt-4o-mini",
    )


@pytest.fixture
def reduce_config():
    return ReduceOp(
        name="group_summary",
        type="reduce",
        reduce_key="group",
        prompt="Summarize the following group of values: {{ inputs }} Provide a total and any other relevant statistics.",
        output={"schema": {"total": "number", "avg": "number"}},
        model="gpt-4o-mini",
    )


@pytest.fixture
def parallel_map_config():
    return ParallelMapOp(
        name="sentiment_and_word_count",
        type="parallel_map",
        prompts=[
            {
                "name": "sentiment",
                "prompt": "Analyze the sentiment of the following text: '{{ input.text }}'. Classify it as either positive, negative, or neutral.",
                "output_keys": ["sentiment"],
                "model": "gpt-4o-mini",
            },
            {
                "name": "word_count",
                "prompt": "Count the number of words in the following text: '{{ input.text }}'. Return the count as an integer.",
                "output_keys": ["word_count"],
                "model": "gpt-4o-mini",
            },
        ],
        output={"schema": {"sentiment": "string", "word_count": "integer"}},
    )


@pytest.fixture
def filter_config():
    return FilterOp(
        name="positive_sentiment_filter",
        type="filter",
        prompt="Is the sentiment of the following text positive? '{{ input.text }}'. Return true if positive, false otherwise.",
        model="gpt-4o-mini",
        output={"schema": {"filtered": "boolean"}},
    )


@pytest.fixture
def resolve_config():
    return ResolveOp(
        name="name_email_resolver",
        type="resolve",
        blocking_keys=["name", "email"],
        blocking_threshold=0.8,
        comparison_prompt="Compare the following two entries and determine if they likely refer to the same person: Person 1: {{ input1 }} Person 2: {{ input2 }} Return true if they likely match, false otherwise.",
        output={"schema": {"name": "string", "email": "string"}},
        embedding_model="text-embedding-3-small",
        comparison_model="gpt-4o-mini",
        resolution_model="gpt-4o-mini",
        resolution_prompt="Given the following list of similar entries, determine one common name and email. {{ inputs }}",
    )


@pytest.fixture
def reduce_sample_data(temp_input_file):
    data = [
        {"group": "A", "value": 10},
        {"group": "B", "value": 20},
        {"group": "A", "value": 15},
        {"group": "B", "value": 25},
    ]
    with open(temp_input_file, "w") as f:
        json.dump(data, f)
    return temp_input_file


@pytest.fixture
def resolve_sample_data(temp_input_file):
    data = [
        {"name": "John Doe"},
        {"name": "Jane Smith"},
        {"name": "Bob Johnson"},
    ]
    with open(temp_input_file, "w") as f:
        json.dump(data, f)
    return temp_input_file


@pytest.fixture
def left_data(temp_input_file):
    data = [
        {"id": "1", "name": "John Doe"},
        {"id": "2", "name": "Jane Smith"},
        {"id": "3", "name": "Bob Johnson"},
    ]
    with open(temp_input_file, "w") as f:
        json.dump(data, f)
    return temp_input_file


@pytest.fixture
def right_data(temp_input_file):
    data = [
        {"id": "1", "email": "john@example.com", "age": 30},
        {"id": "2", "email": "jane@example.com", "age": 28},
        {"id": "3", "email": "bob@example.com", "age": 35},
    ]
    with open(temp_input_file, "w") as f:
        json.dump(data, f)
    return temp_input_file


def test_pipeline_creation(
    map_config, reduce_config, temp_input_file, temp_output_file, temp_intermediate_dir
):
    pipeline = Pipeline(
        name="test_pipeline",
        datasets={"test_input": Dataset(type="file", path=temp_input_file)},
        operations=[map_config, reduce_config],
        steps=[
            PipelineStep(
                name="map_step", input="test_input", operations=["sentiment_analysis"]
            ),
            PipelineStep(
                name="reduce_step", input="map_step", operations=["group_summary"]
            ),
        ],
        output=PipelineOutput(
            type="file", path=temp_output_file, intermediate_dir=temp_intermediate_dir
        ),
        default_model="gpt-4o-mini",
    )

    assert isinstance(pipeline, Pipeline)
    assert len(pipeline.operations) == 2
    assert len(pipeline.steps) == 2


def test_pipeline_optimization(
    map_config, reduce_config, temp_input_file, temp_output_file, temp_intermediate_dir
):
    pipeline = Pipeline(
        name="test_pipeline",
        datasets={"test_input": Dataset(type="file", path=temp_input_file)},
        operations=[map_config, reduce_config],
        steps=[
            PipelineStep(
                name="map_step", input="test_input", operations=["sentiment_analysis"]
            ),
            PipelineStep(
                name="reduce_step", input="map_step", operations=["group_summary"]
            ),
        ],
        output=PipelineOutput(
            type="file", path=temp_output_file, intermediate_dir=temp_intermediate_dir
        ),
        default_model="gpt-4o-mini",
    )

    optimized_pipeline = pipeline.optimize(
        max_threads=4, model="gpt-4o-mini", timeout=10
    )

    assert isinstance(optimized_pipeline, Pipeline)
    assert len(optimized_pipeline.operations) == len(pipeline.operations)
    assert len(optimized_pipeline.steps) == len(pipeline.steps)


def test_pipeline_execution(
    map_config, temp_input_file, temp_output_file, temp_intermediate_dir
):
    pipeline = Pipeline(
        name="test_pipeline",
        datasets={"test_input": Dataset(type="file", path=temp_input_file)},
        operations=[map_config],
        steps=[
            PipelineStep(
                name="map_step", input="test_input", operations=["sentiment_analysis"]
            ),
        ],
        output=PipelineOutput(
            type="file", path=temp_output_file, intermediate_dir=temp_intermediate_dir
        ),
        default_model="gpt-4o-mini",
    )

    cost = pipeline.run(max_threads=4)

    assert isinstance(cost, float)


def test_parallel_map_pipeline(
    parallel_map_config, temp_input_file, temp_output_file, temp_intermediate_dir
):
    pipeline = Pipeline(
        name="test_pipeline",
        datasets={"test_input": Dataset(type="file", path=temp_input_file)},
        operations=[parallel_map_config],
        steps=[
            PipelineStep(
                name="parallel_map_step",
                input="test_input",
                operations=["sentiment_and_word_count"],
            ),
        ],
        output=PipelineOutput(
            type="file", path=temp_output_file, intermediate_dir=temp_intermediate_dir
        ),
        default_model="gpt-4o-mini",
    )

    cost = pipeline.run(max_threads=4)

    assert isinstance(cost, float)


def test_filter_pipeline(
    filter_config, temp_input_file, temp_output_file, temp_intermediate_dir
):
    pipeline = Pipeline(
        name="test_pipeline",
        datasets={"test_input": Dataset(type="file", path=temp_input_file)},
        operations=[filter_config],
        steps=[
            PipelineStep(
                name="filter_step",
                input="test_input",
                operations=["positive_sentiment_filter"],
            ),
        ],
        output=PipelineOutput(
            type="file", path=temp_output_file, intermediate_dir=temp_intermediate_dir
        ),
        default_model="gpt-4o-mini",
    )

    cost = pipeline.run(max_threads=4)

    assert isinstance(cost, float)


def test_reduce_pipeline(
    reduce_config, reduce_sample_data, temp_output_file, temp_intermediate_dir
):
    pipeline = Pipeline(
        name="test_pipeline",
        datasets={"test_input": Dataset(type="file", path=reduce_sample_data)},
        operations=[reduce_config],
        steps=[
            PipelineStep(
                name="reduce_step", input="test_input", operations=["group_summary"]
            ),
        ],
        output=PipelineOutput(
            type="file", path=temp_output_file, intermediate_dir=temp_intermediate_dir
        ),
        default_model="gpt-4o-mini",
    )

    cost = pipeline.run(max_threads=4)

    assert isinstance(cost, float)


def test_resolve_pipeline(
    resolve_config, resolve_sample_data, temp_output_file, temp_intermediate_dir
):
    pipeline = Pipeline(
        name="test_pipeline",
        datasets={"test_input": Dataset(type="file", path=resolve_sample_data)},
        operations=[resolve_config],
        steps=[
            PipelineStep(
                name="resolve_step",
                input="test_input",
                operations=["name_email_resolver"],
            ),
        ],
        output=PipelineOutput(
            type="file", path=temp_output_file, intermediate_dir=temp_intermediate_dir
        ),
        default_model="gpt-4o-mini",
    )

    cost = pipeline.run(max_threads=4)

    assert isinstance(cost, float)


def test_equijoin_pipeline(
    left_data, right_data, temp_output_file, temp_intermediate_dir
):
    pipeline = Pipeline(
        name="test_pipeline",
        datasets={
            "left": Dataset(type="file", path=left_data),
            "right": Dataset(type="file", path=right_data),
        },
        operations=[
            EquijoinOp(
                name="user_data_join",
                type="equijoin",
                left="left",
                right="right",
                comparison_prompt="Compare the following two entries and determine if they are the same id: Left: {{ left.id }} Right: {{ right.id }}",
                embedding_model="text-embedding-3-small",
                comparison_model="gpt-4o-mini",
            )
        ],
        steps=[
            PipelineStep(
                name="equijoin_step",
                operations=[
                    {
                        "user_data_join": {
                            "left": "left",
                            "right": "right",
                        }
                    }
                ],
            ),
        ],
        output=PipelineOutput(
            type="file", path=temp_output_file, intermediate_dir=temp_intermediate_dir
        ),
        default_model="gpt-4o-mini",
    )

    cost = pipeline.run(max_threads=4)

    assert isinstance(cost, float)
