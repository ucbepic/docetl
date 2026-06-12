"""MOAR integration: invalid directive candidates rejected statically
(before any execution), valid candidates canonicalized by the rewrite
rules in their saved YAMLs."""

import os

import pytest
import yaml

from docetl.moar.MOARSearch import MOARSearch
from docetl.moar.Node import Node
from docetl.plan import InvalidCandidatePlan, validate_config


def root_yaml(tmp_path):
    config = {
        "datasets": {"docs": {"type": "file", "path": str(tmp_path / "docs.json")}},
        "default_model": "gpt-4o-mini",
        "operations": [
            {
                "name": "m",
                "type": "map",
                "prompt": "Summarize {{ input.text }}",
                "output": {"schema": {"summary": "string"}},
            },
            {
                "name": "f",
                "type": "code_filter",
                "code": "def transform(doc):\n    return doc['category'] == 'a'",
            },
        ],
        "pipeline": {
            "steps": [
                {"name": "s1", "input": "docs", "operations": ["m"]},
                {"name": "s2", "input": "s1", "operations": ["f"]},
            ],
            "output": {"type": "file", "path": str(tmp_path / "out.json")},
        },
    }
    path = tmp_path / "root.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return str(path)


def make_search(tmp_path):
    """A minimally-wired MOARSearch: enough state for instantiate_node."""
    search = object.__new__(MOARSearch)
    search.console = __import__("docetl.console", fromlist=["DOCETL_CONSOLE"]).DOCETL_CONSOLE
    search.sample_dataset_path = None
    search.output_dir = str(tmp_path / "candidates")
    search.directive_name_to_obj = {}
    return search


class TestCandidateValidation:
    def test_invalid_candidate_rejected_before_writing(self, tmp_path):
        node = Node(root_yaml(tmp_path))
        search = make_search(tmp_path)

        # Directive output that breaks the pipeline: the map vanished from
        # the ops list, so the step now references an undefined op... but
        # update_pipeline patches steps. Instead: a malformed replacement op
        # (map without prompt/output) — exactly what used to explode at
        # runtime in Node.execute_plan.
        new_ops_list = [
            {"name": "m", "type": "map"},  # invalid: no prompt/output/drop_keys
            node.parsed_yaml["operations"][1],
        ]
        with pytest.raises(InvalidCandidatePlan) as exc_info:
            search.instantiate_node(
                node, new_ops_list, "broken_directive", ["m"], "acc", []
            )
        assert any(issue.where == "m" for issue in exc_info.value.issues)
        # Rejected before any candidate yaml was written.
        candidate_dir = tmp_path / "candidates"
        assert not candidate_dir.exists() or not list(candidate_dir.glob("*.yaml"))

    def test_valid_candidate_written_and_canonicalized(self, tmp_path):
        node = Node(root_yaml(tmp_path))
        search = make_search(tmp_path)

        # A no-op "directive": same ops back. The selection pushdown applies
        # to the root shape (LLM map then disjoint code filter), so the
        # saved candidate must already carry the rewritten step structure.
        new_ops_list = list(node.parsed_yaml["operations"])
        child = search.instantiate_node(
            node, new_ops_list, "noop_directive", ["m"], "acc", []
        )
        assert os.path.exists(child.yaml_file_path)
        saved = yaml.safe_load(open(child.yaml_file_path))
        steps = saved["pipeline"]["steps"]
        assert len(steps) == 1
        assert steps[0]["operations"] == ["f", "m"]
        # Canonicalization is a fixpoint: the saved yaml validates clean and
        # re-rewriting changes nothing.
        from docetl.plan import apply_rewrites_to_config

        reparsed, applied = apply_rewrites_to_config(saved)
        assert reparsed is saved and not applied
        assert not [i for i in validate_config(saved) if i.level == "error"]


class TestRootValidationWarnings:
    def test_validate_config_flags_root_problems(self, tmp_path):
        # MOARSearch.__init__ logs these as warnings; here we check the
        # underlying detection on a root with a dangling op reference.
        config = {
            "datasets": {"docs": {"type": "file", "path": "x.json"}},
            "operations": [],
            "pipeline": {
                "steps": [{"name": "s1", "input": "docs", "operations": ["ghost"]}],
                "output": {"type": "file", "path": "o.json"},
            },
        }
        errors = [i for i in validate_config(config) if i.level == "error"]
        assert errors and "ghost" in errors[0].message
