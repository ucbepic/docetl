"""distill_instructions in MOAR: which plan teaches which."""

import json
import threading
from types import SimpleNamespace

from docetl.moar.MOARSearch import MOARSearch


class _Plan:
    def __init__(self, parsed_yaml, op_dict):
        self.parsed_yaml, self.op_dict = parsed_yaml, op_dict


def _plan(tmp_path, name, model, prompt, rows):
    out = tmp_path / f"{name}.json"
    out.write_text(json.dumps(rows))
    op = {"name": "extract", "type": "map", "model": model, "prompt": prompt,
          "output": {"schema": {"clauses": "list[str]"}}}
    config = {"operations": [op], "pipeline": {"output": {"path": str(out)}}}
    return _Plan(config, {"extract": op})


def _search(accuracy):
    search = MOARSearch.__new__(MOARSearch)
    search.tree_lock = threading.RLock()
    search.pareto_frontier = SimpleNamespace(plans_accuracy=accuracy)
    return search


def test_a_small_model_plan_learns_from_the_best_plan_on_another_model(tmp_path):
    small = _plan(tmp_path, "small", "quail/qwen3-4b-fp8", "find clauses",
                  [{"name": "a.pdf", "clauses": []}, {"name": "b.pdf", "clauses": ["x"]}])
    same_model = _plan(tmp_path, "same", "quail/qwen3-4b-fp8", "same model prompt",
                       [{"name": "a.pdf", "clauses": ["y"]}])
    hosted = _plan(tmp_path, "hosted", "gemini/gemini-4-hydrogen", "teacher prompt",
                   [{"name": "a.pdf", "clauses": ["governing law"]}, {"name": "b.pdf", "clauses": ["x"]}])
    search = _search({small: 0.4, same_model: 0.9, hosted: 0.8})

    context = search._teacher_context(small, ["extract"])

    assert "gemini/gemini-4-hydrogen" in context and "teacher prompt" in context
    assert "Document 'a.pdf'" in context and "Document 'b.pdf'" not in context


def test_the_most_accurate_plan_has_no_teacher(tmp_path):
    best = _plan(tmp_path, "best", "gemini/gemini-4-hydrogen", "p", [{"name": "a.pdf", "clauses": []}])
    other = _plan(tmp_path, "other", "quail/qwen3-4b-fp8", "q", [{"name": "a.pdf", "clauses": ["x"]}])
    assert _search({best: 0.8, other: 0.4})._teacher_context(best, ["extract"]) == ""
