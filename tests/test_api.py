import inspect

import pytest
import json
import tempfile
import os
import pandas as pd

import docetl
from docetl import _config
from docetl.api import (
    Pipeline,
    Dataset,
    MapOp,
    ReduceOp,
    PipelineStep,
    PipelineOutput,
    ExtractOp,
)
from dotenv import load_dotenv

load_dotenv()


@pytest.fixture(autouse=True)
def reset_default_model():
    """Reset docetl._config.default_model before and after each test."""
    original = _config.default_model
    _config.default_model = None
    yield
    _config.default_model = original


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
def reduce_sample_data():
    data = [
        {"group": "A", "value": 10},
        {"group": "B", "value": 20},
        {"group": "A", "value": 15},
        {"group": "B", "value": 25},
    ]
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as tmp:
        json.dump(data, tmp)
    yield tmp.name
    os.unlink(tmp.name)


@pytest.fixture
def resolve_sample_data():
    data = [
        {"name": "John Doe"},
        {"name": "Jane Smith"},
        {"name": "Bob Johnson"},
    ]
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as tmp:
        json.dump(data, tmp)
    yield tmp.name
    os.unlink(tmp.name)


@pytest.fixture
def left_data():
    data = [
        {"id": "1", "name": "John Doe"},
        {"id": "2", "name": "Jane Smith"},
        {"id": "3", "name": "Bob Johnson"},
    ]
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as tmp:
        json.dump(data, tmp)
    yield tmp.name
    os.unlink(tmp.name)


@pytest.fixture
def right_data():
    data = [
        {"id": "1", "email": "john@example.com", "age": 30},
        {"id": "2", "email": "jane@example.com", "age": 28},
        {"id": "3", "email": "bob@example.com", "age": 35},
    ]
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as tmp:
        json.dump(data, tmp)
    yield tmp.name
    os.unlink(tmp.name)


# ── Frame construction / structure tests ──────────────────────────


def test_frame_creation_from_json(temp_input_file):
    """Frame built from read_json should have correct datasets and chaining."""
    docetl.default_model = "gpt-4o-mini"

    frame = (
        docetl.read_json(temp_input_file)
        .map(
            prompt="Analyze the sentiment of the following text: '{{ input.text }}'. Classify it as either positive, negative, or neutral.",
            output={"schema": {"sentiment": "string"}},
            model="gpt-4o-mini",
        )
        .reduce(
            reduce_key="group",
            prompt="Summarize the following group of values: {{ inputs }} Provide a total and any other relevant statistics.",
            output={"schema": {"total": "number", "avg": "number"}},
            model="gpt-4o-mini",
        )
    )

    assert isinstance(frame, docetl.Frame)
    assert len(frame._operations) == 2
    assert len(frame._steps) == 2
    assert frame._operations[0]["type"] == "map"
    assert frame._operations[1]["type"] == "reduce"


def test_frame_creation_from_list():
    """Frame built from from_list should have correct datasets."""
    docetl.default_model = "gpt-4o-mini"

    data = [
        {"text": "Hello", "group": "A"},
        {"text": "World", "group": "B"},
    ]
    frame = docetl.from_list(data)

    assert isinstance(frame, docetl.Frame)
    assert frame._first_dataset == "data"
    assert frame._datasets["data"]["type"] == "memory"
    assert frame._datasets["data"]["path"] is data


def test_frame_immutability(temp_input_file):
    """Each operation should return a new Frame, leaving the original unchanged."""
    docetl.default_model = "gpt-4o-mini"

    base = docetl.read_json(temp_input_file)
    mapped = base.map(
        prompt="Analyze: {{ input.text }}",
        output={"schema": {"sentiment": "string"}},
    )

    assert len(base._operations) == 0
    assert len(base._steps) == 0
    assert len(mapped._operations) == 1
    assert len(mapped._steps) == 1
    assert base is not mapped


# ── Pipeline optimization (V1) — kept on Pipeline API ────────────


def test_pipeline_optimization(
    temp_input_file, temp_output_file, temp_intermediate_dir
):
    map_config = MapOp(
        name="sentiment_analysis",
        type="map",
        prompt="Analyze the sentiment of the following text: '{{ input.text }}'. Classify it as either positive, negative, or neutral.",
        output={"schema": {"sentiment": "string"}},
        model="gpt-4o-mini",
    )
    reduce_config = ReduceOp(
        name="group_summary",
        type="reduce",
        reduce_key="group",
        prompt="Summarize the following group of values: {{ inputs }} Provide a total and any other relevant statistics.",
        output={"schema": {"total": "number", "avg": "number"}},
        model="gpt-4o-mini",
    )

    temp_input_dataset = Dataset(type="file", path=temp_input_file)

    pipeline = Pipeline(
        name="test_pipeline",
        datasets={"test_input": temp_input_dataset},
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
        optimizer_config={
            "rewrite_agent_model": "gpt-4o",
            "judge_agent_model": "gpt-4o-mini",
        },
    )

    optimized_pipeline = pipeline.optimize(
        method="v1",
        max_threads=64,
    )

    assert isinstance(optimized_pipeline, Pipeline)
    assert len(optimized_pipeline.operations) == len(pipeline.operations) + 1
    assert len(optimized_pipeline.steps) == len(pipeline.steps)


# ── LLM execution tests via Frame API ────────────────────────────


def test_map_execution(temp_input_file):
    """Map operation via Frame API should execute and return results."""
    docetl.default_model = "gpt-4o-mini"

    results = (
        docetl.read_json(temp_input_file)
        .map(
            prompt="Analyze the sentiment of the following text: '{{ input.text }}'. Classify it as either positive, negative, or neutral.",
            output={"schema": {"sentiment": "string"}},
            model="gpt-4o-mini",
        )
        .collect()
    )

    assert isinstance(results, list)
    assert len(results) == 3
    for item in results:
        assert "sentiment" in item


def test_parallel_map_execution(temp_input_file):
    """Parallel map via Frame API should execute and return results."""
    docetl.default_model = "gpt-4o-mini"

    results = (
        docetl.read_json(temp_input_file)
        .parallel_map(
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
        .collect()
    )

    assert isinstance(results, list)
    assert len(results) == 3


def test_filter_execution(temp_input_file):
    """Filter via Frame API should execute and return results."""
    docetl.default_model = "gpt-4o-mini"

    results = (
        docetl.read_json(temp_input_file)
        .filter(
            prompt="Is the sentiment of the following text positive? '{{ input.text }}'. Return true if positive, false otherwise.",
            model="gpt-4o-mini",
            output={"schema": {"filtered": "boolean"}},
        )
        .collect()
    )

    assert isinstance(results, list)
    # Should have filtered out some items
    assert len(results) <= 3


def test_reduce_execution(reduce_sample_data):
    """Reduce via Frame API should execute and return results."""
    docetl.default_model = "gpt-4o-mini"

    results = (
        docetl.read_json(reduce_sample_data)
        .reduce(
            reduce_key="group",
            prompt="Summarize the following group of values: {{ inputs }} Provide a total and any other relevant statistics.",
            output={"schema": {"total": "number", "avg": "number"}},
            model="gpt-4o-mini",
        )
        .collect()
    )

    assert isinstance(results, list)
    assert len(results) == 2  # Two groups: A and B


def test_resolve_execution(resolve_sample_data):
    """Resolve via Frame API should execute and return results."""
    docetl.default_model = "gpt-4o-mini"

    results = (
        docetl.read_json(resolve_sample_data)
        .resolve(
            blocking_keys=["name"],
            blocking_threshold=0.8,
            comparison_prompt="Compare the following two entries and determine if they likely refer to the same person: Person 1: {{ input1 }} Person 2: {{ input2 }} Return true if they likely match, false otherwise.",
            output={"schema": {"name": "string"}},
            embedding_model="text-embedding-3-small",
            comparison_model="gpt-4o-mini",
            resolution_model="gpt-4o-mini",
            resolution_prompt="Given the following list of similar entries, determine one common name. {{ inputs }}",
        )
        .collect()
    )

    assert isinstance(results, list)


def test_equijoin_execution(left_data, right_data):
    """Equijoin via Frame API should execute and return results."""
    docetl.default_model = "gpt-4o-mini"

    left = docetl.read_json(left_data)
    right = docetl.read_json(right_data)

    results = (
        left.equijoin(
            right,
            comparison_prompt="Compare the following two entries and determine if they are the same id: Left: {{ left.id }} Right: {{ right.id }}",
            embedding_model="text-embedding-3-small",
            comparison_model="gpt-4o-mini",
        )
        .collect()
    )

    assert isinstance(results, list)


# ── Code operations via Frame API ─────────────────────────────────


def _code_map_transform(doc: dict) -> dict:
    x = doc.get("x", 0)
    return {"double": x * 2}


def _code_filter_transform(doc: dict) -> bool:
    return bool(doc.get("keep", False))


def _code_reduce_transform(group: list[dict]) -> dict:
    total = sum(item.get("value", 0) for item in group)
    return {"group_total": total}


def _callable_to_code(fn) -> str:
    """Convert a callable to the source-string format expected by code operations."""
    src = inspect.getsource(fn)
    return f"{src}\ntransform = {fn.__name__}"


def test_code_map_via_frame():
    """code_map via Frame API should apply the transform function."""
    docetl.default_model = "gpt-4o-mini"

    data = [{"x": 1}, {"x": 2}, {"x": 3}]
    results = (
        docetl.from_list(data)
        .code_map(code=_callable_to_code(_code_map_transform))
        .collect()
    )

    assert len(results) == 3
    assert results[0]["double"] == 2
    assert results[1]["double"] == 4
    assert results[2]["double"] == 6


def test_code_filter_via_frame():
    """code_filter via Frame API should filter rows by the predicate."""
    docetl.default_model = "gpt-4o-mini"

    data = [
        {"id": 1, "keep": True},
        {"id": 2, "keep": False},
        {"id": 3, "keep": True},
    ]
    results = (
        docetl.from_list(data)
        .code_filter(code=_callable_to_code(_code_filter_transform))
        .collect()
    )

    kept_ids = sorted([d["id"] for d in results])
    assert kept_ids == [1, 3]


def test_code_reduce_via_frame():
    """code_reduce via Frame API should group and reduce."""
    docetl.default_model = "gpt-4o-mini"

    data = [
        {"group": "A", "value": 10},
        {"group": "A", "value": 5},
        {"group": "B", "value": 7},
    ]
    results = (
        docetl.from_list(data)
        .code_reduce(code=_callable_to_code(_code_reduce_transform), reduce_key="group", pass_through=True)
        .collect()
    )

    assert len(results) == 2
    totals = {d["group"]: d["group_total"] for d in results}
    assert totals["A"] == 15
    assert totals["B"] == 7


# ── ExtractOp export test ─────────────────────────────────────────


def test_extractop_is_exported():
    """Ensure ExtractOp is importable and constructible from API schemas."""
    op = ExtractOp(
        name="extract_sections",
        type="extract",
        document_keys=["content"],
        prompt="Extract important parts from {{ input.content }}",
        extraction_method="line_number",
    )
    assert op.type == "extract"


# ── Frame from_dict round-trip ────────────────────────────────────


def test_from_dict_round_trip(temp_input_file, temp_output_file, temp_intermediate_dir):
    """Pipeline.from_dict should produce a Pipeline whose _to_dict matches the input."""
    config = {
        "default_model": "gpt-4o-mini",
        "datasets": {
            "docs": {"type": "file", "path": temp_input_file},
        },
        "operations": [
            {
                "name": "analyze",
                "type": "map",
                "prompt": "Analyze: {{ input.text }}",
                "output": {"schema": {"sentiment": "string"}},
            },
            {
                "name": "summarize",
                "type": "reduce",
                "reduce_key": "group",
                "prompt": "Summarize: {% for item in inputs %}{{ item.text }}{% endfor %}",
                "output": {"schema": {"summary": "string"}},
            },
        ],
        "pipeline": {
            "steps": [
                {"name": "step1", "input": "docs", "operations": ["analyze"]},
                {"name": "step2", "input": "step1", "operations": ["summarize"]},
            ],
            "output": {
                "type": "file",
                "path": temp_output_file,
                "intermediate_dir": temp_intermediate_dir,
            },
        },
    }

    pipeline = Pipeline.from_dict(config, name="test_rt")

    assert pipeline.name == "test_rt"
    assert pipeline.default_model == "gpt-4o-mini"
    assert len(pipeline.operations) == 2
    assert len(pipeline.steps) == 2

    # ops_by_name accessor
    by_name = pipeline.ops_by_name
    assert "analyze" in by_name
    assert "summarize" in by_name
    assert by_name["analyze"].type == "map"
    assert by_name["summarize"].type == "reduce"

    # get_step_for_op
    assert pipeline.get_step_for_op("analyze").name == "step1"
    assert pipeline.get_step_for_op("summarize").name == "step2"

    # Round-trip: from_dict -> _to_dict should preserve operations and steps
    rt = pipeline._to_dict()
    assert len(rt["operations"]) == 2
    assert len(rt["pipeline"]["steps"]) == 2
    rt_op_names = {op["name"] for op in rt["operations"]}
    assert rt_op_names == {"analyze", "summarize"}


def test_from_dict_with_equijoin(temp_output_file, temp_intermediate_dir):
    """from_dict should handle equijoin operations and dict-style step operations."""
    config = {
        "default_model": "gpt-4o-mini",
        "datasets": {
            "left": {"type": "file", "path": "left.json"},
            "right": {"type": "file", "path": "right.json"},
        },
        "operations": [
            {
                "name": "my_join",
                "type": "equijoin",
                "comparison_prompt": "Compare {{ left.id }} with {{ right.id }}",
                "embedding_model": "text-embedding-3-small",
            },
        ],
        "pipeline": {
            "steps": [
                {
                    "name": "join_step",
                    "operations": [{"my_join": {"left": "left", "right": "right"}}],
                },
            ],
            "output": {"type": "file", "path": temp_output_file},
        },
    }

    pipeline = Pipeline.from_dict(config)
    assert len(pipeline.operations) == 1
    assert pipeline.ops_by_name["my_join"].type == "equijoin"
    # The step should have the dict-form operation reference preserved
    step_ops = pipeline.steps[0].operations
    assert isinstance(step_ops[0], dict)
    assert "my_join" in step_ops[0]


# ── Frame terminal action return types ───────────────────────────


def test_collect_returns_rows_and_to_pandas_returns_dataframe():
    """collect() returns list[dict]; to_pandas() returns a DataFrame."""
    docetl.default_model = "gpt-4o-mini"

    data = [{"x": 1}, {"x": 2}, {"x": 3}]
    frame = docetl.from_list(data).code_map(code=_callable_to_code(_code_map_transform))

    rows = frame.collect()
    assert isinstance(rows, list) and isinstance(rows[0], dict)
    assert [r["double"] for r in rows] == [2, 4, 6]

    df = frame.to_pandas()
    assert isinstance(df, pd.DataFrame)
    assert list(df["double"]) == [2, 4, 6]
    assert "_total_cost" in df.attrs


# ── DSLRunner accepts Pipeline ───────────────────────────────────


def test_dsrunner_accepts_pipeline(temp_input_file, temp_output_file, temp_intermediate_dir):
    """DSLRunner should accept a Pipeline object directly."""
    from docetl.runner import DSLRunner

    pipeline = Pipeline(
        name="test_typed",
        datasets={"docs": Dataset(type="file", path=temp_input_file)},
        operations=[
            MapOp(
                name="sentiment",
                type="map",
                prompt="Classify sentiment: {{ input.text }}",
                output={"schema": {"sentiment": "string"}},
            ),
        ],
        steps=[PipelineStep(name="s1", input="docs", operations=["sentiment"])],
        output=PipelineOutput(
            type="file", path=temp_output_file, intermediate_dir=temp_intermediate_dir
        ),
        default_model="gpt-4o-mini",
    )

    runner = DSLRunner(pipeline, max_threads=4)
    assert runner.pipeline is pipeline
    assert runner.default_model == "gpt-4o-mini"
    assert "sentiment" in runner._op_map
