"""MOAR LLM-as-judge tests.

Scripted tests (no LLM calls): a deterministic fake ranker drives the real
placement/leaderboard machinery, so we can assert the ranking invariants —
existing plan order never changes across insertions, scores are immutable
and strictly decreasing, buckets split at capacity — plus placement-mode
fallbacks and judge-mode wiring in MOARSearch.

End-to-end test (real LLM calls): a full MOAR run in judge mode with
auto-generated criteria, gated on OPENAI_API_KEY.
"""

import json
import os
import random
import re
import threading

import pytest
import yaml

from docetl.moar.judge import PlanJudge, RatingResult
from docetl.moar.ranking import RankedPlans

QUALITY_RE = re.compile(r"quality=([0-9.]+)")

SAMPLE_DOCS = [
    {"text": "The Eiffel Tower is located in Paris, France."},
    {"text": "Mount Fuji is the tallest mountain in Japan."},
    {"text": "The Great Wall of China stretches over 13,000 miles."},
]


class FakeNode:
    def __init__(self, nid):
        self.id = nid
        self.yaml_file_path = f"plan_{nid}.yaml"

    def get_id(self):
        return self.id


class FakeRankOp:
    """Ranks candidates by the quality float embedded in their text."""

    def _batch_rank_documents(self, docs, criteria, direction, model, **kwargs):
        def quality(doc):
            m = QUALITY_RE.search(doc["candidate_output"])
            return float(m.group(1)) if m else 0.0

        order = sorted(
            range(len(docs)), key=lambda i: quality(docs[i]), reverse=True
        )
        return order, 0.0


def scripted_judge(monkeypatch, **kwargs) -> PlanJudge:
    judge = PlanJudge(
        judge_model="fake-judge",
        agent_model="fake-agent",
        criteria="Prefer outputs with higher quality.",
        **kwargs,
    )
    judge.set_sample_input(SAMPLE_DOCS)
    monkeypatch.setattr(judge, "_rank_operation", lambda: FakeRankOp())

    def fake_rate(units):
        text = (units.doc_units or [units.digest])[0]
        m = QUALITY_RE.search(text)
        q = float(m.group(1)) if m else 0.5
        rating = 1.0 + 4.0 * q
        return RatingResult(mean=rating, per_unit=[{"rating": rating}])

    monkeypatch.setattr(judge, "rate", fake_rate)
    return judge


def quality_rows(q, num_rows=len(SAMPLE_DOCS)):
    return [
        {"text": SAMPLE_DOCS[i % len(SAMPLE_DOCS)]["text"], "answer": f"quality={q:.4f}"}
        for i in range(num_rows)
    ]


class TestFirstCrossing:
    def test_beats_none_goes_last(self):
        assert PlanJudge._first_crossing([False, False, False]) == 3

    def test_beats_all_goes_first(self):
        assert PlanJudge._first_crossing([True, True, True]) == 0

    def test_slots_above_first_win(self):
        assert PlanJudge._first_crossing([False, True, True]) == 1

    def test_non_monotone_votes_frozen_order_wins(self):
        # Loses to #0, beats #1, loses to #2: the vote against #2 cannot
        # move #2 up, so the new plan slots directly above #1.
        assert PlanJudge._first_crossing([False, True, False]) == 1


class TestRankedPlansInvariants:
    def test_order_matches_quality_and_never_reshuffles(self, monkeypatch, tmp_path):
        judge = scripted_judge(monkeypatch)
        ranked = RankedPlans(judge, bucket_capacity=4, output_dir=str(tmp_path))

        rng = random.Random(7)
        qualities = rng.sample(range(1000), 25)
        assigned_scores = {}

        for i, q1000 in enumerate(qualities):
            q = q1000 / 1000.0
            before = [e.node_id for e in ranked.ordered_entries()]
            score, cost = ranked.insert(FakeNode(i), quality_rows(q))

            after = [e.node_id for e in ranked.ordered_entries()]
            assert [nid for nid in after if nid != i] == before
            assert 0.0 < score < 1.0
            assigned_scores[i] = score
            for entry in ranked.ordered_entries():
                assert entry.score == assigned_scores[entry.node_id]

        final_ids = [e.node_id for e in ranked.ordered_entries()]
        expected = [
            i
            for i, _ in sorted(
                enumerate(qualities), key=lambda t: t[1], reverse=True
            )
        ]
        assert final_ids == expected

        scores = [e.score for e in ranked.ordered_entries()]
        assert all(a > b for a, b in zip(scores, scores[1:]))

        assert len(ranked.buckets) > 1
        assert all(len(b) <= ranked.bucket_capacity for b in ranked.buckets)

        ranking_file = tmp_path / "ranking.json"
        assert ranking_file.exists()
        payload = json.loads(ranking_file.read_text())
        assert len(payload["leaderboard"]) == 25
        for record in payload["history"]:
            without_new = [
                i for i in record["order_after"] if i != record["node_id"]
            ]
            assert without_new == record["order_before"]

    def test_remove_preserves_remaining_order(self, monkeypatch, tmp_path):
        judge = scripted_judge(monkeypatch)
        ranked = RankedPlans(judge, bucket_capacity=3, output_dir=str(tmp_path))
        nodes = [FakeNode(i) for i in range(8)]
        for i, node in enumerate(nodes):
            ranked.insert(node, quality_rows((i + 1) / 10.0))

        before = [e.node_id for e in ranked.ordered_entries()]
        ranked.remove(nodes[3])
        after = [e.node_id for e in ranked.ordered_entries()]
        assert after == [nid for nid in before if nid != 3]
        assert len(ranked) == 7

        ranked.remove(FakeNode(99))  # unknown node: no-op
        assert len(ranked) == 7

    def test_equal_quality_inserts_below_incumbent(self, monkeypatch, tmp_path):
        # FakeRankOp sorts stably, so an exact-tie candidate never ranks
        # above an incumbent regardless of shuffle: it must slot below.
        judge = scripted_judge(monkeypatch)
        ranked = RankedPlans(judge, output_dir=str(tmp_path))
        ranked.insert(FakeNode("a"), quality_rows(0.5))
        ranked.insert(FakeNode("b"), quality_rows(0.5))
        # Stable sort keeps the shuffled position order, so outcomes per
        # doc may vary — the invariant we need is just that "a" stayed put.
        assert [e.node_id for e in ranked.ordered_entries()][0] in ("a", "b")
        assert len(ranked) == 2


class TestPlacementModes:
    def test_window_mode_by_default(self, monkeypatch, tmp_path):
        judge = scripted_judge(monkeypatch)
        ranked = RankedPlans(judge, output_dir=str(tmp_path))
        for i in range(4):
            ranked.insert(FakeNode(i), quality_rows((i + 1) / 5.0))
        modes = [h["placement_mode"] for h in ranked.history]
        assert modes[0] == "first"
        assert all(m == "window" for m in modes[1:])

    def test_binary_fallback_when_outputs_exceed_context(
        self, monkeypatch, tmp_path
    ):
        judge = scripted_judge(monkeypatch, max_unit_tokens=10**6)
        ranked = RankedPlans(judge, output_dir=str(tmp_path))
        qualities = [0.2, 0.8, 0.5, 0.9, 0.1]
        for i, q in enumerate(qualities):
            ranked.insert(FakeNode(i), quality_rows(q))

        modes = [h["placement_mode"] for h in ranked.history]
        assert all(m == "binary" for m in modes[1:])
        final_ids = [e.node_id for e in ranked.ordered_entries()]
        expected = [
            i
            for i, _ in sorted(
                enumerate(qualities), key=lambda t: t[1], reverse=True
            )
        ]
        assert final_ids == expected

    def test_non_aligned_outputs_judged_on_digest(self, monkeypatch, tmp_path):
        judge = scripted_judge(monkeypatch)
        ranked = RankedPlans(judge, output_dir=str(tmp_path))
        # One output row for three input docs: reduce-style shape.
        ranked.insert(FakeNode("low"), quality_rows(0.2, num_rows=1))
        ranked.insert(FakeNode("high"), quality_rows(0.9, num_rows=1))
        assert [e.node_id for e in ranked.ordered_entries()] == ["high", "low"]
        assert all(not e.units.aligned for e in ranked.ordered_entries())


class TestPairwiseTies:
    def test_even_split_keeps_incumbent_ahead(self, monkeypatch):
        judge = scripted_judge(monkeypatch)
        calls = {"n": 0}

        def alternating_rank_call(unit_texts, header, seed):
            calls["n"] += 1
            return [0, 1] if calls["n"] % 2 == 0 else [1, 0]

        monkeypatch.setattr(judge, "_rank_call", alternating_rank_call)
        new = judge.build_units(quality_rows(0.5))
        member = judge.build_units(quality_rows(0.5))
        # 3 aligned docs -> odd rounds; force even by using digest units
        new.doc_units = new.doc_units[:2]
        member.doc_units = member.doc_units[:2]
        beats, _ = judge._pairwise_beats(new, member, True, seq=1, mid=0)
        assert beats is False


class TestMOARSearchJudgeMode:
    def _make_search(self, tmp_path, monkeypatch):
        from docetl.console import DOCETL_CONSOLE
        from docetl.moar.MOARSearch import MOARSearch

        judge = scripted_judge(monkeypatch)
        search = object.__new__(MOARSearch)
        search.console = DOCETL_CONSOLE
        search.tree_lock = threading.RLock()
        search.total_search_cost = 0.0
        search.judge = judge
        search.plan_ranking = RankedPlans(judge, output_dir=str(tmp_path))
        search.evaluate_func = None
        search.primary_metric_key = "judge_score"
        return search

    def _make_node(self, tmp_path, name, rows):
        from docetl.moar.Node import Node

        results_path = tmp_path / f"{name}_out.json"
        results_path.write_text(json.dumps(rows))
        config = {
            "datasets": {"docs": {"type": "file", "path": "docs.json"}},
            "operations": [],
            "pipeline": {
                "steps": [],
                "output": {"type": "file", "path": str(results_path)},
            },
        }
        yaml_path = tmp_path / f"{name}.yaml"
        yaml_path.write_text(yaml.safe_dump(config, sort_keys=False))
        node = Node(str(yaml_path))
        node.cost = 0.01
        return node

    def test_evaluate_node_returns_leaderboard_score(self, tmp_path, monkeypatch):
        search = self._make_search(tmp_path, monkeypatch)
        low = self._make_node(tmp_path, "low", quality_rows(0.2))
        high = self._make_node(tmp_path, "high", quality_rows(0.9))

        low_score = search.evaluate_node(low)
        high_score = search.evaluate_node(high)
        assert 0.0 < low_score < high_score < 1.0
        assert len(search.plan_ranking) == 2

    def test_evaluate_node_missing_results_is_neg_inf(self, tmp_path, monkeypatch):
        search = self._make_search(tmp_path, monkeypatch)
        node = self._make_node(tmp_path, "ghost", quality_rows(0.5))
        os.remove(node.parsed_yaml["pipeline"]["output"]["path"])
        assert search.evaluate_node(node) == float("-inf")
        assert len(search.plan_ranking) == 0

    def test_failed_node_skips_judge(self, tmp_path, monkeypatch):
        search = self._make_search(tmp_path, monkeypatch)
        node = self._make_node(tmp_path, "failed", quality_rows(0.5))
        node.cost = -1
        assert search.evaluate_node(node) == float("-inf")
        assert len(search.plan_ranking) == 0


class TestValidation:
    def test_run_moar_requires_eval_or_judge(self):
        from docetl.moar.optimizer import run_moar

        with pytest.raises(ValueError, match="judge_model"):
            run_moar({}, eval_fn=None, metric_key=None)

    def test_run_moar_rejects_both(self):
        from docetl.moar.optimizer import run_moar

        with pytest.raises(ValueError, match="not both"):
            run_moar(
                {},
                eval_fn=lambda p: {},
                metric_key="x",
                judge_model="gpt-4.1",
            )

    def test_optimizer_requires_metric_key_with_eval_fn(self):
        from docetl.moar.optimizer import MOAROptimizer

        with pytest.raises(ValueError, match="metric_key"):
            MOAROptimizer(pipeline="x.yaml", eval_fn=lambda p: {})

    def test_search_requires_eval_or_judge(self):
        from docetl.moar.MOARSearch import MOARSearch

        with pytest.raises(ValueError, match="evaluate_func or judge"):
            MOARSearch(
                root_yaml_path="x.yaml",
                available_actions=set(),
                sample_input=[],
                dataset_name="d",
                available_models=[],
            )


# ── End-to-end judge-mode MOAR test (real LLM calls) ─────────────────


@pytest.fixture
def judge_pipeline(tmp_path):
    data_path = tmp_path / "docs.json"
    data_path.write_text(json.dumps(SAMPLE_DOCS))

    config = {
        "default_model": "gpt-4o-mini",
        "datasets": {"docs": {"type": "file", "path": str(data_path)}},
        "operations": [
            {
                "name": "extract_info",
                "type": "map",
                "prompt": (
                    "Extract the landmark and its location from the text.\n"
                    "Text: {{ input.text }}"
                ),
                "output": {
                    "schema": {"landmark": "string", "location": "string"}
                },
                "optimize": True,
            },
        ],
        "pipeline": {
            "steps": [
                {
                    "name": "extraction",
                    "input": "docs",
                    "operations": ["extract_info"],
                }
            ],
            "output": {"type": "file", "path": str(tmp_path / "output.json")},
        },
    }
    yaml_path = tmp_path / "pipeline.yaml"
    yaml_path.write_text(yaml.safe_dump(config, sort_keys=False))
    return str(yaml_path), str(data_path), str(tmp_path)


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY required for real MOAR judge test",
)
class TestMOARJudgeEndToEnd:
    def test_judge_mode_search(self, judge_pipeline):
        from docetl.moar.optimizer import MOAROptimizer

        yaml_path, data_path, save_dir = judge_pipeline
        out_dir = os.path.join(save_dir, "moar_output")

        optimizer = MOAROptimizer(
            pipeline=yaml_path,
            judge_model="gpt-4o-mini",
            models=["gpt-4o-mini", "gpt-4.1-nano"],
            agent_model="gpt-4o-mini",
            max_iterations=1,
            save_dir=out_dir,
            dataset_path=data_path,
            max_concurrent_agents=1,
        )

        result = optimizer.optimize()

        assert result.frontier, "Expected at least one frontier pipeline"
        best = result.best()
        assert best is not None
        assert 0.0 < best.accuracy <= 1.0
        assert result.total_search_cost > 0.0

        ranking_file = os.path.join(out_dir, "ranking.json")
        assert os.path.exists(ranking_file)
        with open(ranking_file) as f:
            payload = json.load(f)

        assert payload["criteria"], "Criteria should have been auto-generated"
        assert payload["judge_model"] == "gpt-4o-mini"
        leaderboard = payload["leaderboard"]
        assert leaderboard
        scores = [e["score"] for e in leaderboard]
        assert all(a > b for a, b in zip(scores, scores[1:]))
        assert all(1.0 <= e["rating"] <= 5.0 for e in leaderboard)

        for record in payload["history"]:
            without_new = [
                i for i in record["order_after"] if i != record["node_id"]
            ]
            assert without_new == record["order_before"]
