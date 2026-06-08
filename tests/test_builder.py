"""Tests for the PipelineBuilder fluent API and YAML conversion."""

import json
import os
import tempfile

import pytest
import yaml

from docetl.builder import PipelineBuilder, yaml_to_python


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


class TestBuilderConstruction:
    def test_minimal_build(self, tmp_dir, sample_input):
        pipe = (
            PipelineBuilder(default_model="gpt-4o-mini")
            .dataset("docs", path=sample_input)
            .output(os.path.join(tmp_dir, "out.json"))
            .map("summarize",
                 prompt="Summarize: {{ input.text }}",
                 output={"schema": {"summary": "string"}})
            .build()
        )

        assert pipe.default_model == "gpt-4o-mini"
        assert len(pipe.operations) == 1
        assert pipe.operations[0].name == "summarize"
        assert len(pipe.steps) == 1
        assert pipe.steps[0].name == "step_summarize"
        assert pipe.steps[0].input == "docs"

    def test_no_name_required(self, tmp_dir, sample_input):
        pipe = (
            PipelineBuilder()
            .dataset("d", path=sample_input)
            .output(os.path.join(tmp_dir, "out.json"))
            .map(prompt="p")
            .build()
        )
        assert pipe.name == "out"

    def test_name_derived_from_output(self, tmp_dir, sample_input):
        pipe = (
            PipelineBuilder()
            .dataset("d", path=sample_input)
            .output(os.path.join(tmp_dir, "my_results.json"))
            .map(prompt="p")
            .build()
        )
        assert pipe.name == "my_results"

    def test_multi_op_chain(self, tmp_dir, sample_input):
        pipe = (
            PipelineBuilder(default_model="gpt-4o-mini")
            .dataset("docs", path=sample_input)
            .output(os.path.join(tmp_dir, "out.json"))
            .map("op1", prompt="Do thing 1: {{ input.text }}",
                 output={"schema": {"result1": "string"}})
            .filter("op2", prompt="Keep good ones: {{ input.result1 }}",
                    output={"schema": {"keep": "boolean"}})
            .reduce("op3", reduce_key="category",
                    prompt="Combine: {{ inputs }}",
                    output={"schema": {"combined": "string"}})
            .build()
        )

        assert len(pipe.operations) == 3
        assert len(pipe.steps) == 3
        assert pipe.steps[0].input == "docs"
        assert pipe.steps[1].input == "step_op1"
        assert pipe.steps[2].input == "step_op2"

    def test_auto_name(self, tmp_dir, sample_input):
        pipe = (
            PipelineBuilder()
            .dataset("d", path=sample_input)
            .output(os.path.join(tmp_dir, "out.json"))
            .map(prompt="p1", output={"schema": {"x": "string"}})
            .map(prompt="p2", output={"schema": {"y": "string"}})
            .build()
        )
        assert pipe.operations[0].name == "map_1"
        assert pipe.operations[1].name == "map_2"

    def test_explicit_step(self, tmp_dir, sample_input):
        pipe = (
            PipelineBuilder()
            .dataset("d", path=sample_input)
            .output(os.path.join(tmp_dir, "out.json"))
            .step("process", input="d")
            .map("op1", prompt="p1")
            .filter("op2", prompt="p2", output={"schema": {"k": "boolean"}})
            .step()
            .reduce("op3", reduce_key="x", prompt="p3",
                    output={"schema": {"r": "string"}})
            .build()
        )
        assert len(pipe.steps) == 2
        assert pipe.steps[0].name == "process"
        assert pipe.steps[0].input == "d"
        assert pipe.steps[0].operations == ["op1", "op2"]
        assert pipe.steps[1].input == "process"

    def test_equijoin(self, tmp_dir):
        left_path = os.path.join(tmp_dir, "left.json")
        right_path = os.path.join(tmp_dir, "right.json")
        for p in (left_path, right_path):
            with open(p, "w") as f:
                json.dump([{"id": "1"}], f)

        pipe = (
            PipelineBuilder()
            .dataset("left", path=left_path)
            .dataset("right", path=right_path)
            .output(os.path.join(tmp_dir, "out.json"))
            .equijoin("join", left="left", right="right",
                      comparison_prompt="Compare: {{ left }} vs {{ right }}")
            .build()
        )
        assert len(pipe.operations) == 1
        assert pipe.operations[0].type == "equijoin"
        step = pipe.steps[0]
        assert isinstance(step.operations[0], dict)
        assert "join" in step.operations[0]

    def test_all_map_params(self, tmp_dir, sample_input):
        pipe = (
            PipelineBuilder()
            .dataset("d", path=sample_input)
            .output(os.path.join(tmp_dir, "out.json"))
            .map("full",
                 prompt="p",
                 output={"schema": {"x": "string"}},
                 model="gpt-4o",
                 optimize=True,
                 tools=[{"function": {"name": "t"}}],
                 batch_size=5,
                 drop_keys=["text"],
                 timeout=120,
                 enable_observability=True,
                 limit=10,
                 litellm_completion_kwargs={"temperature": 0.5})
            .build()
        )
        op = pipe.operations[0]
        assert op.name == "full"
        assert op.model == "gpt-4o"
        assert op.optimize is True
        assert op.batch_size == 5
        assert op.limit == 10

    def test_structural_ops(self, tmp_dir, sample_input):
        pipe = (
            PipelineBuilder()
            .dataset("d", path=sample_input)
            .output(os.path.join(tmp_dir, "out.json"))
            .unnest("u", unnest_key="items", keep_empty=True)
            .split("s", split_key="text", method="delimiter",
                   method_kwargs={"delimiter": "\n"})
            .build()
        )
        assert len(pipe.operations) == 2
        assert pipe.operations[0].type == "unnest"
        assert pipe.operations[1].type == "split"

    def test_memory_dataset(self, tmp_dir):
        pipe = (
            PipelineBuilder()
            .dataset("d", data=[{"x": 1}, {"x": 2}])
            .output(os.path.join(tmp_dir, "out.json"))
            .map(prompt="p")
            .build()
        )
        ds = pipe.datasets["d"]
        assert ds.type == "memory"


class TestFromYaml:
    def test_load_yaml(self, sample_yaml):
        builder = PipelineBuilder.from_yaml(sample_yaml)
        pipe = builder.build()

        assert len(pipe.datasets) == 1
        assert "docs" in pipe.datasets
        assert len(pipe.operations) == 2
        assert pipe.operations[0].name == "summarize"
        assert pipe.operations[1].name == "aggregate"
        assert len(pipe.steps) == 2
        assert pipe.steps[0].input == "docs"
        assert pipe.steps[1].input == "step1"
        assert pipe.default_model == "gpt-4o-mini"

    def test_roundtrip_yaml(self, sample_yaml):
        builder = PipelineBuilder.from_yaml(sample_yaml)
        pipe = builder.build()

        assert pipe.operations[0].prompt == "Summarize: {{ input.text }}"
        assert pipe.operations[1].reduce_key == "category"


class TestToPython:
    def test_basic_codegen(self, sample_yaml):
        code = yaml_to_python(sample_yaml)
        assert "from docetl.builder import PipelineBuilder" in code
        assert "PipelineBuilder(" in code
        assert ".dataset(" in code
        assert ".output(" in code
        assert ".map(" in code
        assert ".reduce(" in code
        assert "pipe.run()" in code

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

    def test_builder_to_python(self, tmp_dir, sample_input):
        builder = (
            PipelineBuilder(default_model="gpt-4o-mini")
            .dataset("docs", path=sample_input)
            .output(os.path.join(tmp_dir, "out.json"))
            .map("summarize",
                 prompt="Summarize: {{ input.text }}",
                 output={"schema": {"summary": "string"}})
        )
        code = builder.to_python()
        assert "default_model='gpt-4o-mini'" in code
        assert ".map('summarize'" in code
        compile(code, "<test>", "exec")

    def test_no_name_in_codegen(self, tmp_dir, sample_input):
        builder = (
            PipelineBuilder(default_model="gpt-4o-mini")
            .dataset("docs", path=sample_input)
            .output(os.path.join(tmp_dir, "out.json"))
            .map("op", prompt="p")
        )
        code = builder.to_python()
        assert "PipelineBuilder(default_model=" in code

    def test_multiline_prompt(self, tmp_dir, sample_input):
        builder = (
            PipelineBuilder()
            .dataset("d", path=sample_input)
            .output(os.path.join(tmp_dir, "out.json"))
            .map("op", prompt="Line 1\nLine 2\nLine 3",
                 output={"schema": {"x": "string"}})
        )
        code = builder.to_python()
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
        assert ".step(" in code
        compile(code, "<test>", "exec")

    def test_memory_dataset_codegen(self, tmp_dir):
        builder = (
            PipelineBuilder()
            .dataset("d", data=[{"x": 1}])
            .output(os.path.join(tmp_dir, "out.json"))
            .map(prompt="p")
        )
        code = builder.to_python()
        assert "data=" in code
        assert "type=" not in code or "type='memory'" not in code
        compile(code, "<test>", "exec")


class TestExtraConfig:
    def test_system_prompt(self, tmp_dir, sample_input):
        pipe = (
            PipelineBuilder(
                default_model="gpt-4o-mini",
                system_prompt={"persona": "analyst"})
            .dataset("d", path=sample_input)
            .output(os.path.join(tmp_dir, "out.json"))
            .map("op", prompt="p")
            .build()
        )
        assert pipe.other_config.get("system_prompt") == {"persona": "analyst"}

    def test_extra_config_in_codegen(self, tmp_dir, sample_input):
        builder = PipelineBuilder(system_prompt={"persona": "analyst"})
        builder.dataset("d", path=sample_input)
        builder.output(os.path.join(tmp_dir, "out.json"))
        builder.map("op", prompt="p")

        code = builder.to_python()
        assert "system_prompt=" in code
        compile(code, "<test>", "exec")
