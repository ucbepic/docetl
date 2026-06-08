"""Tests for the PySpark-like Frame API."""

import json
import os
import tempfile

import docetl
from docetl import _config
from docetl.frame import Frame, read_json, read_csv, read_parquet, from_list


class TestModuleConfig:
    def setup_method(self):
        _config.default_model = None
        _config.rate_limits = None

    def teardown_method(self):
        _config.default_model = None
        _config.rate_limits = None

    def test_set_default_model(self):
        docetl.default_model = "gpt-4o-mini"
        assert docetl.default_model == "gpt-4o-mini"
        assert _config.default_model == "gpt-4o-mini"

    def test_set_rate_limits(self):
        docetl.rate_limits = {"gpt-4o-mini": 100}
        assert docetl.rate_limits == {"gpt-4o-mini": 100}


class TestReaders:
    def test_read_json(self):
        frame = read_json("data/input.json")
        assert isinstance(frame, Frame)
        assert "input" in frame._datasets
        assert frame._datasets["input"]["type"] == "file"
        assert frame._datasets["input"]["path"] == "data/input.json"

    def test_read_csv(self):
        frame = read_csv("data/input.csv")
        assert isinstance(frame, Frame)
        assert "input" in frame._datasets
        assert frame._datasets["input"]["type"] == "file"

    def test_read_parquet(self):
        frame = read_parquet("data/input.parquet")
        assert isinstance(frame, Frame)
        assert "input" in frame._datasets
        assert frame._datasets["input"]["type"] == "file"

    def test_read_json_with_parsing(self):
        frame = read_json("input.json", parsing=[{"function": "txt_to_string"}])
        assert frame._datasets["input"]["parsing"] == [{"function": "txt_to_string"}]

    def test_from_list(self):
        frame = from_list([{"x": 1}, {"x": 2}])
        assert isinstance(frame, Frame)
        assert "data" in frame._datasets
        assert frame._datasets["data"]["type"] == "memory"

    def test_from_list_custom_name(self):
        frame = from_list([{"x": 1}], name="my_data")
        assert "my_data" in frame._datasets


class TestChaining:
    def test_map_returns_frame(self):
        frame = from_list([{"x": 1}])
        result = frame.map(prompt="p")
        assert isinstance(result, Frame)
        assert result is not frame

    def test_immutable(self):
        frame = from_list([{"x": 1}])
        result = frame.map(prompt="p")
        assert len(frame._operations) == 0
        assert len(result._operations) == 1

    def test_chain_map_filter_reduce(self):
        frame = (
            from_list([{"x": 1}])
            .map(prompt="do thing")
            .filter(prompt="keep good")
            .reduce(reduce_key="x", prompt="combine")
        )
        assert len(frame._operations) == 3
        assert len(frame._steps) == 3
        types = [op["type"] for op in frame._operations]
        assert types == ["map", "filter", "reduce"]

    def test_auto_naming(self):
        frame = (
            from_list([{"x": 1}])
            .map(prompt="p1")
            .map(prompt="p2")
            .filter(prompt="f")
        )
        names = [op["name"] for op in frame._operations]
        assert names == ["map_1", "map_2", "filter_1"]

    def test_explicit_names(self):
        frame = (
            from_list([{"x": 1}])
            .map("summarize", prompt="p")
            .filter("quality_check", prompt="q")
        )
        names = [op["name"] for op in frame._operations]
        assert names == ["summarize", "quality_check"]

    def test_step_wiring(self):
        frame = (
            from_list([{"x": 1}], name="d")
            .map("op1", prompt="p1")
            .filter("op2", prompt="p2")
        )
        assert frame._steps[0]["input"] == "d"
        assert frame._steps[1]["input"] == "step_op1"

    def test_equijoin(self):
        left = from_list([{"id": 1}], name="left")
        right = from_list([{"id": 2}], name="right")
        joined = left.equijoin(right, "join_op",
                               comparison_prompt="Compare {{ left }} {{ right }}")
        assert len(joined._operations) == 1
        assert joined._operations[0]["type"] == "equijoin"
        assert "left" in joined._datasets
        assert "right" in joined._datasets

    def test_structural_ops(self):
        frame = (
            from_list([{"x": 1}])
            .unnest(unnest_key="items")
            .split(split_key="text", method="delimiter")
        )
        types = [op["type"] for op in frame._operations]
        assert types == ["unnest", "split"]

    def test_all_op_types(self):
        frame = from_list([{"x": 1}])

        frame.map(prompt="p")
        frame.parallel_map(prompts=[{"prompt": "p"}])
        frame.filter(prompt="p")
        frame.reduce(reduce_key="x", prompt="p")
        frame.resolve(comparison_prompt="p")
        frame.extract(prompt="p")
        frame.split(split_key="x")
        frame.gather(content_key="x")
        frame.unnest(unnest_key="x")
        frame.cluster(embedding_keys=["x"])
        frame.sample(method="random", samples=10)
        frame.code_map(code="def transform(doc): return [doc]")
        frame.code_reduce(code="def transform(items): return [items[0]]")
        frame.code_filter(code="def transform(doc): return True")

    def test_none_values_excluded(self):
        frame = from_list([{"x": 1}]).map(prompt="p", model=None, limit=None)
        op = frame._operations[0]
        assert "model" not in op
        assert "limit" not in op
        assert op["prompt"] == "p"


class TestBuildRunner:
    def setup_method(self):
        _config.default_model = None

    def teardown_method(self):
        _config.default_model = None

    def _valid_map_kwargs(self):
        return dict(
            prompt="Summarize: {{ input.text }}",
            output={"schema": {"summary": "string"}},
        )

    def test_build_runner_with_global_model(self, tmp_path):
        _config.default_model = "gpt-4o-mini"
        input_path = tmp_path / "input.json"
        input_path.write_text(json.dumps([{"text": "hello"}]))

        frame = read_json(str(input_path)).map(**self._valid_map_kwargs())
        runner = frame._build_runner()
        assert runner.default_model == "gpt-4o-mini"

    def test_build_runner_with_output(self, tmp_path):
        _config.default_model = "gpt-4o-mini"
        input_path = tmp_path / "input.json"
        input_path.write_text(json.dumps([{"text": "hello"}]))

        frame = read_json(str(input_path)).map(**self._valid_map_kwargs())
        out = str(tmp_path / "out.json")
        runner = frame._build_runner(output_path=out)
        assert runner.pipeline.output.path == out

    def test_build_runner_no_output(self, tmp_path):
        _config.default_model = "gpt-4o-mini"
        input_path = tmp_path / "input.json"
        input_path.write_text(json.dumps([{"text": "hello"}]))

        frame = read_json(str(input_path)).map(**self._valid_map_kwargs())
        runner = frame._build_runner()
        assert runner.pipeline.output.path == ""
