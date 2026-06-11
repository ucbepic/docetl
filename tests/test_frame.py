"""Tests for the PySpark-like Frame API."""

import json
import os
import tempfile

import docetl
from docetl import _config
from docetl.frame import Frame, read_json, read_csv, read_parquet, from_list


class TestModuleConfig:
    _ATTRS = (
        "default_model", "agent_model", "max_threads", "bypass_cache",
        "rate_limits", "fallback_models", "fallback_embedding_models",
        "intermediate_dir",
    )

    def setup_method(self):
        self._saved = {a: getattr(_config, a) for a in self._ATTRS}
        for a in self._ATTRS:
            setattr(_config, a, None if a != "bypass_cache" else False)

    def teardown_method(self):
        for a, v in self._saved.items():
            setattr(_config, a, v)

    def test_set_default_model(self):
        docetl.default_model = "gpt-4o-mini"
        assert docetl.default_model == "gpt-4o-mini"
        assert _config.default_model == "gpt-4o-mini"

    def test_set_rate_limits(self):
        docetl.rate_limits = {"gpt-4o-mini": 100}
        assert docetl.rate_limits == {"gpt-4o-mini": 100}

    def test_set_all_config_attrs(self):
        docetl.agent_model = "gpt-4o"
        docetl.max_threads = 32
        docetl.bypass_cache = True
        docetl.fallback_models = ["gpt-4o", "gpt-4o-mini"]
        docetl.fallback_embedding_models = ["text-embedding-3-small"]
        docetl.intermediate_dir = "/tmp/docetl"

        assert docetl.agent_model == "gpt-4o"
        assert docetl.max_threads == 32
        assert docetl.bypass_cache is True
        assert docetl.fallback_models == ["gpt-4o", "gpt-4o-mini"]
        assert docetl.fallback_embedding_models == ["text-embedding-3-small"]
        assert docetl.intermediate_dir == "/tmp/docetl"

    def test_config_flows_to_build_config(self):
        docetl.default_model = "gpt-4o"
        docetl.bypass_cache = True
        docetl.fallback_models = ["gpt-4o-mini"]
        frame = from_list([{"x": 1}]).map(prompt="p", output={"schema": {"y": "str"}})
        cfg = frame._build_config()
        assert cfg["default_model"] == "gpt-4o"
        assert cfg["bypass_cache"] is True
        assert cfg["fallback_models"] == ["gpt-4o-mini"]

    def test_max_threads_flows_to_runner(self):
        docetl.default_model = "gpt-4o-mini"
        docetl.max_threads = 16
        frame = from_list([{"x": 1}]).map(prompt="p", output={"schema": {"y": "str"}})
        runner = frame._build_runner()
        assert runner.max_threads == 16


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


class TestCheckpointSafety:
    """Regressions for checkpoint hashing: previews, lineage, and mutation."""

    def setup_method(self):
        self._saved = (_config.default_model, _config.intermediate_dir)

    def teardown_method(self):
        _config.default_model, _config.intermediate_dir = self._saved

    @staticmethod
    def _doubler(data):
        return from_list(data).code_map(
            "double", code="def transform(doc): return {'y': doc['x'] * 2}"
        )

    def test_show_does_not_poison_collect(self, tmp_path):
        _config.intermediate_dir = str(tmp_path / "ckpt")
        data = [{"x": i} for i in range(20)]
        frame = self._doubler(data)

        assert len(frame.collect()) == 20
        assert len(frame.show(max=3)) == 3
        # show() must neither overwrite nor invalidate the full run's
        # checkpoint: collect() still returns all rows.
        assert len(frame.collect()) == 20

    def test_input_data_change_invalidates_checkpoint(self, tmp_path):
        _config.intermediate_dir = str(tmp_path / "ckpt")
        first = self._doubler([{"x": 1}]).to_list()
        second = self._doubler([{"x": 5}]).to_list()
        assert first[0]["y"] == 2
        assert second[0]["y"] == 10

    def test_upstream_step_change_invalidates_downstream(self, tmp_path):
        _config.intermediate_dir = str(tmp_path / "ckpt")
        data = [{"x": 1}]

        def pipeline(increment):
            return (
                from_list(data)
                .code_map("a", code=f"def transform(doc): return {{'v': doc['x'] + {increment}}}")
                .code_map("b", code="def transform(doc): return {'w': doc['v'] * 10}")
            )

        assert pipeline(1).to_list()[0]["w"] == 20
        # 'b' is unchanged but its input lineage changed — it must re-run.
        assert pipeline(2).to_list()[0]["w"] == 30

    def test_equijoin_config_flows_into_downstream_hash(self, tmp_path):
        _config.default_model = "gpt-4o-mini"
        _config.intermediate_dir = str(tmp_path / "ckpt")

        def hashes(comparison_prompt):
            left = from_list([{"k": 1}], name="l")
            right = from_list([{"k": 2}], name="r")
            joined = left.equijoin(
                right, "j", comparison_prompt=comparison_prompt
            ).code_map("post", code="def transform(doc): return {'z': 1}")
            runner = joined._build_runner()
            return runner.step_op_hashes

        h1 = hashes("prompt A {{ left.k }} {{ right.k }}")
        h2 = hashes("prompt B {{ left.k }} {{ right.k }}")
        assert h1["step_j"]["j"] != h2["step_j"]["j"]
        assert h1["step_post"]["post"] != h2["step_post"]["post"]

    def test_runner_does_not_mutate_frame_ops(self):
        _config.default_model = "model-A"
        frame = from_list([{"x": 1}]).map(
            "m", prompt="p {{ input.x }}", output={"schema": {"s": "string"}}
        )
        frame._build_runner()
        assert "model" not in frame._operations[0]


class TestEquijoinChaining:
    def test_joins_each_sides_last_step(self):
        left = from_list([{"k": 1}], name="l").map(
            "lm", prompt="x", output={"schema": {"a": "string"}})
        right = from_list([{"k": 2}], name="r").map(
            "rm", prompt="y", output={"schema": {"b": "string"}})
        joined = left.equijoin(right, comparison_prompt="c")

        cfg = joined._build_config()
        op_names = [op["name"] for op in cfg["operations"]]
        assert "lm" in op_names and "rm" in op_names
        join_ref = cfg["pipeline"]["steps"][-1]["operations"][0]["equijoin_1"]
        assert join_ref == {"left": "step_lm", "right": "step_rm"}

    def test_namespace_collisions_renamed(self):
        left = from_list([{"k": 1}]).map(prompt="L", output={"schema": {"a": "string"}})
        right = from_list([{"k": 2}]).map(prompt="R", output={"schema": {"a": "string"}})
        joined = left.equijoin(right, comparison_prompt="c")

        cfg = joined._build_config()
        names = [op["name"] for op in cfg["operations"]]
        assert len(names) == len(set(names))
        # Both default-named 'data' datasets survive, with different contents.
        assert len(cfg["datasets"]) == 2
        prompts = {op.get("prompt") for op in cfg["operations"] if op["type"] == "map"}
        assert prompts == {"L", "R"}

    def test_shared_ancestry_deduplicated(self):
        base = from_list([{"k": 1}]).map("shared", prompt="s", output={"schema": {"a": "string"}})
        left = base.filter("lf", prompt="l", output={"schema": {"keep": "boolean"}})
        right = base.filter("rf", prompt="r", output={"schema": {"keep": "boolean"}})
        joined = left.equijoin(right, comparison_prompt="c")

        cfg = joined._build_config()
        assert [op["name"] for op in cfg["operations"]].count("shared") == 1
        step_names = [s["name"] for s in cfg["pipeline"]["steps"]]
        assert step_names.count("step_shared") == 1

    def test_raw_frames_join_datasets(self):
        left = from_list([{"k": 1}], name="l")
        right = from_list([{"k": 2}], name="r")
        joined = left.equijoin(right, "j", comparison_prompt="c")
        join_ref = joined._steps[-1]["operations"][0]["j"]
        assert join_ref == {"left": "l", "right": "r"}


class TestShowSampling:
    def test_sampled_datasets_keep_parsing(self, tmp_path):
        path = tmp_path / "in.json"
        path.write_text(json.dumps([{"t": "alpha"}, {"t": "beta"}]))
        frame = read_json(str(path), parsing=[
            {"function": "txt_to_string", "input_key": "t", "output_key": "c"},
        ])
        sampled = frame._sample_datasets(1)
        ds = sampled["in"]
        assert ds["type"] == "memory"
        assert len(ds["path"]) == 1
        assert ds["parsing"] == frame._datasets["in"]["parsing"]

    def test_bare_show_applies_parsing(self, tmp_path):
        txt = tmp_path / "doc.txt"
        txt.write_text("hello world")
        path = tmp_path / "in.json"
        path.write_text(json.dumps([{"f": str(txt)}]))
        frame = read_json(str(path), parsing=[
            {"function": "txt_to_string", "input_key": "f", "output_key": "content"},
        ])
        data = frame._load_input_data()
        assert data[0]["content"] == "hello world"


class TestMemoization:
    def setup_method(self):
        self._saved = _config.default_model

    def teardown_method(self):
        _config.default_model = self._saved

    @staticmethod
    def _counting_frame(tmp_path, n=4):
        marker = tmp_path / "runs.log"
        code = (
            "def transform(doc):\n"
            f"    open({str(marker)!r}, 'a').write('x')\n"
            "    return {'y': doc['x'] * 2}"
        )
        frame = from_list([{"x": i} for i in range(n)]).code_map("m", code=code)
        return frame, marker

    def test_terminal_actions_execute_once(self, tmp_path):
        frame, marker = self._counting_frame(tmp_path)
        assert frame.count() == 4
        assert len(frame.collect()) == 4
        frame.write_json(str(tmp_path / "out.json"))
        assert len(marker.read_text()) == 4  # one execution of 4 rows
        assert len(json.loads((tmp_path / "out.json").read_text())) == 4

    def test_config_change_invalidates_memo(self, tmp_path):
        frame, marker = self._counting_frame(tmp_path, n=1)
        _config.default_model = "memo-model-a"
        frame.count()
        _config.default_model = "memo-model-b"
        frame.count()
        assert len(marker.read_text()) == 2

    def test_result_mutation_does_not_corrupt_memo(self, tmp_path):
        frame, _ = self._counting_frame(tmp_path, n=1)
        rows = frame.to_list()
        rows[0]["y"] = 999
        assert frame.to_list()[0]["y"] == 0


class TestSchema:
    def test_structural_ops_tracked(self):
        frame = (
            from_list([{"text": "a b c"}])
            .map(prompt="p", output={"schema": {"summary": "string", "tags": "list[string]"}})
            .unnest(unnest_key="tags")
            .split("sp", split_key="summary", method="token_count",
                   method_kwargs={"num_tokens": 10})
            .gather("g", content_key="summary_chunk", doc_id_key="sp_id",
                    order_key="sp_chunk_num")
            .extract("ex", prompt="e {{ input.summary }}", document_keys=["summary"])
        )
        assert frame.schema() == {
            "summary": "string",
            "tags": "string",  # unnested: list[string] -> string
            "summary_chunk": "string",
            "sp_id": "string",
            "sp_chunk_num": "integer",
            "summary_chunk_rendered": "string",
            "summary_extracted_ex": "string",
        }

    def test_filter_key_consumed(self):
        frame = (
            from_list([{"x": 1}])
            .map(prompt="p", output={"schema": {"summary": "string"}})
            .filter(prompt="f", output={"schema": {"keep": "boolean"}})
        )
        assert frame.schema() == {"summary": "string"}

    def test_drop_keys_still_apply(self):
        frame = (
            from_list([{"x": 1}])
            .map(prompt="p", output={"schema": {"a": "string", "b": "string"}},
                 drop_keys=["b"])
        )
        assert frame.schema() == {"a": "string"}

    def test_cluster_adds_output_key(self):
        frame = (
            from_list([{"x": 1}])
            .cluster(embedding_keys=["x"], summary_prompt="s {{ inputs }}",
                     summary_schema={"label": "string"})
        )
        assert frame.schema().get("clusters") == "list"
