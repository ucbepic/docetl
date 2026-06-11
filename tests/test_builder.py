"""Tests for Frame.from_yaml, Frame.to_python, and yaml_to_python."""

import json
import os
import tempfile

import pytest
import yaml

from docetl import _config
from docetl.frame import Frame, yaml_to_python, from_list, read_json


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def sample_input(tmp_dir):
    path = os.path.join(tmp_dir, "input.json")
    data = [
        {"text": "Hello world", "category": "greeting"},
        {"text": "Goodbye world", "category": "farewell"},
    ]
    with open(path, "w") as f:
        json.dump(data, f)
    return path


@pytest.fixture
def sample_yaml(tmp_dir, sample_input):
    config = {
        "default_model": "gpt-4o-mini",
        "datasets": {
            "docs": {"type": "file", "path": sample_input},
        },
        "operations": [
            {
                "name": "summarize",
                "type": "map",
                "prompt": "Summarize: {{ input.text }}",
                "output": {"schema": {"summary": "string"}},
            },
            {
                "name": "aggregate",
                "type": "reduce",
                "reduce_key": "category",
                "prompt": "Group these: {{ inputs }}",
                "output": {"schema": {"grouped": "string"}},
            },
        ],
        "pipeline": {
            "steps": [
                {"name": "step1", "input": "docs", "operations": ["summarize"]},
                {"name": "step2", "input": "step1", "operations": ["aggregate"]},
            ],
            "output": {
                "type": "file",
                "path": os.path.join(tmp_dir, "output.json"),
            },
        },
    }
    path = os.path.join(tmp_dir, "pipeline.yaml")
    with open(path, "w") as f:
        yaml.safe_dump(config, f)
    return path


@pytest.fixture(autouse=True)
def reset_config():
    old_model = _config.default_model
    old_limits = _config.rate_limits
    yield
    _config.default_model = old_model
    _config.rate_limits = old_limits


class TestFromYaml:
    def test_load_yaml(self, sample_yaml):
        frame = Frame.from_yaml(sample_yaml)

        assert len(frame._datasets) == 1
        assert "docs" in frame._datasets
        assert len(frame._operations) == 2
        assert frame._operations[0]["name"] == "summarize"
        assert frame._operations[1]["name"] == "aggregate"
        assert len(frame._steps) == 2
        assert frame._steps[0]["input"] == "docs"
        assert frame._steps[1]["input"] == "step1"
        assert frame._settings["default_model"] == "gpt-4o-mini"

    def test_roundtrip_yaml(self, sample_yaml):
        frame = Frame.from_yaml(sample_yaml)

        assert frame._operations[0]["prompt"] == "Summarize: {{ input.text }}"
        assert frame._operations[1]["reduce_key"] == "category"

    def test_settings_stay_on_frame(self, sample_yaml):
        """Loading a YAML must not touch process-wide settings; the frame
        carries its own, and they win over the globals in its config."""
        _config.default_model = None
        frame = Frame.from_yaml(sample_yaml)
        assert _config.default_model is None
        assert frame._build_config()["default_model"] == "gpt-4o-mini"

        _config.default_model = "globally-set"
        assert frame._build_config()["default_model"] == "gpt-4o-mini"


class TestToPython:
    def test_basic_codegen(self, sample_yaml):
        code = yaml_to_python(sample_yaml)
        assert "import docetl" in code
        assert "docetl.default_model" in code
        assert "docetl.read_json(" in code
        assert ".map(" in code
        assert ".reduce(" in code
        assert ".collect()" in code

    def test_codegen_no_hardcoded_types(self, sample_yaml):
        code = yaml_to_python(sample_yaml)
        assert "type='file'" not in code
        assert "source='local'" not in code

    def test_codegen_contains_params(self, sample_yaml):
        code = yaml_to_python(sample_yaml)
        assert "'summarize'" in code
        assert "'aggregate'" in code
        assert "reduce_key=" in code
        assert "prompt=" in code

    def test_codegen_is_valid_python(self, sample_yaml):
        code = yaml_to_python(sample_yaml)
        compile(code, "<test>", "exec")

    def test_frame_to_python(self, sample_input):
        _config.default_model = "gpt-4o-mini"
        frame = (
            read_json(sample_input)
            .map("summarize",
                 prompt="Summarize: {{ input.text }}",
                 output={"schema": {"summary": "string"}})
        )
        code = frame.to_python()
        assert "docetl.default_model = 'gpt-4o-mini'" in code
        assert ".map('summarize'" in code
        compile(code, "<test>", "exec")

    def test_multiline_prompt(self, sample_input):
        frame = (
            read_json(sample_input)
            .map("op", prompt="Line 1\nLine 2\nLine 3",
                 output={"schema": {"x": "string"}})
        )
        code = frame.to_python()
        assert '"""' in code
        compile(code, "<test>", "exec")

    def test_multi_op_step_codegen(self, tmp_dir):
        config_path = os.path.join(tmp_dir, "multi.yaml")
        config = {
            "default_model": "gpt-4o-mini",
            "datasets": {"d": {"type": "file", "path": "in.json"}},
            "operations": [
                {"name": "op1", "type": "map", "prompt": "p1"},
                {"name": "op2", "type": "filter", "prompt": "p2",
                 "output": {"schema": {"k": "boolean"}}},
            ],
            "pipeline": {
                "steps": [
                    {"name": "step1", "input": "d", "operations": ["op1", "op2"]},
                ],
                "output": {"type": "file", "path": "out.json"},
            },
        }
        with open(config_path, "w") as f:
            yaml.safe_dump(config, f)

        code = yaml_to_python(config_path)
        assert ".map(" in code
        assert ".filter(" in code
        compile(code, "<test>", "exec")

    def test_memory_dataset_codegen(self):
        frame = from_list([{"x": 1}]).map(prompt="p")
        code = frame.to_python()
        assert "from_list(" in code
        assert "type=" not in code or "type='memory'" not in code
        compile(code, "<test>", "exec")

    def test_csv_reader_codegen(self):
        from docetl.frame import read_csv
        frame = read_csv("data.csv").map(prompt="p")
        code = frame.to_python()
        assert "docetl.read_csv(" in code
        compile(code, "<test>", "exec")

    def test_parquet_reader_codegen(self):
        from docetl.frame import read_parquet
        frame = read_parquet("data.parquet").map(prompt="p")
        code = frame.to_python()
        assert "docetl.read_parquet(" in code
        compile(code, "<test>", "exec")


class TestCodegenEscaping:
    def _round_trip_prompt(self, prompt):
        import ast
        frame = from_list([{"x": 1}]).map(
            "m", prompt=prompt, output={"schema": {"s": "string"}})
        code = frame.to_python()
        tree = ast.parse(code)
        found = [
            node.value.value for node in ast.walk(tree)
            if isinstance(node, ast.keyword) and node.arg == "prompt"
            and isinstance(node.value, ast.Constant)
        ]
        assert found, code
        return found[0]

    def test_prompt_ending_in_quote(self):
        prompt = 'Line one\nEnd with "'
        assert self._round_trip_prompt(prompt) == prompt

    def test_prompt_with_triple_quote(self):
        prompt = 'Contains """ inside\nsecond line'
        assert self._round_trip_prompt(prompt) == prompt

    def test_prompt_with_backslashes(self):
        prompt = "Path C:\\temp\\file\nregex \\d+"
        assert self._round_trip_prompt(prompt) == prompt


class TestEquijoinCodegen:
    def test_chained_right_branch_rendered_inline(self):
        left = from_list([{"k": 1}], name="l").map(
            "lm", prompt="x", output={"schema": {"a": "string"}})
        right = from_list([{"k": 2}], name="r").map(
            "rm", prompt="y", output={"schema": {"b": "string"}})
        joined = left.equijoin(right, comparison_prompt="c")

        code = joined.to_python()
        compile(code, "<test>", "exec")
        assert ".equijoin(docetl.from_list(" in code
        assert ".map('rm'" in code
        # right-branch ops must not leak into the main chain
        assert "\n    .map('rm'" not in code
