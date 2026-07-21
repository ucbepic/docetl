"""
LLM-as-judge evaluation for MOAR — no label function required.

Ports the original DocETL paper's agent-guided plan evaluation to the MOAR
search: an agent (the rewrite agent model) synthesizes task-specific
validation criteria from the pipeline, then a judge model (a) rates each
candidate plan's outputs on a 1-5 scale and (b) orders the candidate
against already-evaluated plans via batched ranking calls.

The judge never re-ranks previously evaluated plans; it only answers
"where does this new plan's output slot in?". Ordering state lives in
``docetl.moar.ranking.RankedPlans``.
"""

from __future__ import annotations

import json
import random
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import litellm
from pydantic import BaseModel

from docetl.console import DOCETL_CONSOLE

DEFAULT_CONTEXT_TOKENS = 128_000
MAX_CONTEXT_TOKENS = 120_000
CONTEXT_MARGIN_TOKENS = 4_000


class CriteriaResponseFormat(BaseModel):
    criteria: str


@dataclass
class PlanUnits:
    """A plan's outputs rendered for judging.

    ``doc_units`` holds one rendered output per sample document when the
    output rows align 1:1 with the sample input; otherwise ``None`` and the
    plan is judged on ``digest`` (a truncated rendering of all rows).
    """

    doc_units: Optional[List[str]]
    digest: str

    @property
    def aligned(self) -> bool:
        return self.doc_units is not None


@dataclass
class RatingResult:
    mean: float
    per_unit: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class PlacementResult:
    """Slot for a new plan within a bucket of already-ordered members.

    ``slot`` is the insertion index in the bucket's best-to-worst order:
    0 means above every member, ``len(members)`` means below every member.
    """

    slot: int
    mode: str  # "window" | "binary" | "default"
    votes: Optional[List[bool]] = None
    details: List[Dict[str, Any]] = field(default_factory=list)


def _truncate_to_tokens(text: str, max_tokens: int, model: str) -> str:
    if max_tokens <= 0:
        return ""
    import tiktoken

    try:
        encoder = tiktoken.encoding_for_model(model.split("/")[-1])
    except Exception:
        encoder = tiktoken.encoding_for_model("gpt-4o")
    tokens = encoder.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return encoder.decode(tokens[:max_tokens]) + "\n... [truncated]"


def _strip_internal(row: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in row.items() if not str(k).startswith("_")}


class PlanJudge:
    """Rates and orders plan outputs with LLM calls.

    Criteria generation runs on ``agent_model`` (the MOAR rewrite agent
    model); rating and ranking calls run on ``judge_model``. Batched
    ranking reuses ``RankOperation._batch_rank_documents`` so a whole
    bucket is ordered in one call per sample document; when the bucket's
    outputs don't fit the judge's context window, placement falls back to
    binary insertion with two-candidate calls, and individual outputs are
    token-truncated as a last resort.
    """

    def __init__(
        self,
        judge_model: str,
        agent_model: str,
        criteria: Optional[str] = None,
        num_sample_docs: int = 5,
        max_unit_tokens: int = 1_500,
        max_digest_tokens: int = 4_000,
        max_input_tokens: int = 1_200,
        max_threads: int = 8,
        console=None,
    ):
        self.judge_model = judge_model
        self.agent_model = agent_model
        self.criteria = criteria
        self.num_sample_docs = num_sample_docs
        self.max_unit_tokens = max_unit_tokens
        self.max_digest_tokens = max_digest_tokens
        self.max_input_tokens = max_input_tokens
        self.max_threads = max_threads
        self.console = console if console is not None else DOCETL_CONSOLE

        self.total_cost = 0.0
        self._cost_lock = threading.Lock()
        self._rank_op = None
        self._sample_input: List[Dict[str, Any]] = []
        self._input_excerpts: List[str] = []
        self._digest_input_context = ""

    # ── setup ──────────────────────────────────────────────────────

    def set_sample_input(self, sample_input: List[Dict[str, Any]]) -> None:
        """Fix the sample documents all plans are judged against."""
        self._sample_input = list(sample_input or [])[: self.num_sample_docs]
        self._input_excerpts = [
            _truncate_to_tokens(
                json.dumps(_strip_internal(doc), indent=2, default=str),
                self.max_input_tokens,
                self.judge_model,
            )
            for doc in self._sample_input
        ]
        joined = "\n\n".join(
            f"--- Input document {i + 1} ---\n{excerpt}"
            for i, excerpt in enumerate(self._input_excerpts)
        )
        self._digest_input_context = _truncate_to_tokens(
            joined, 3 * self.max_input_tokens, self.judge_model
        )

    def ensure_criteria(self, pipeline_config: Dict[str, Any]) -> float:
        """Synthesize validation criteria from the pipeline if not provided.

        Runs on the rewrite agent model. Returns the LLM cost incurred
        (0.0 when criteria were supplied by the user).
        """
        if self.criteria:
            return 0.0

        op_descriptions = []
        for op in pipeline_config.get("operations", []):
            desc = f"Operation: {op.get('name')} (type: {op.get('type')})"
            if op.get("prompt"):
                prompt_text = _truncate_to_tokens(
                    str(op["prompt"]), 500, self.agent_model
                )
                desc += f"\nPrompt: {prompt_text}"
            schema = (op.get("output") or {}).get("schema")
            if schema:
                desc += f"\nOutput schema: {json.dumps(schema, default=str)}"
            op_descriptions.append(desc)

        sample_context = (
            self._digest_input_context or "(no sample documents available)"
        )
        user_message = f"""Your task is to write validation criteria for judging the output quality of a document processing pipeline.

The pipeline consists of these operations:

{chr(10).join(op_descriptions)}

Here are sample input documents the pipeline runs on:

{sample_context}

Write a concise validation prompt that a judge can use to assess how well a pipeline's final output accomplishes the task for a given input. The criteria should:
- Describe what a high-quality output looks like for this specific task (completeness, correctness, faithfulness to the input, format).
- Call out likely failure modes worth checking (missed information, hallucinated content, vague or generic answers).
- Be self-contained: the judge will see only these criteria, an input document, and candidate outputs.

Return only the validation criteria text."""

        response = litellm.completion(
            model=self.agent_model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at designing evaluation rubrics for document processing tasks. Your output must follow the structured output format.",
                },
                {"role": "user", "content": user_message},
            ],
            response_format=CriteriaResponseFormat,
        )
        cost = response._hidden_params.get("response_cost", 0) or 0.0
        self._add_cost(cost)
        self.criteria = json.loads(response.choices[0].message.content)["criteria"]
        self.console.log(
            f"[bold blue]Judge criteria generated by {self.agent_model}:[/bold blue]\n"
            f"[dim]{self.criteria}[/dim]"
        )
        return cost

    # ── unit construction ──────────────────────────────────────────

    def build_units(self, output_rows: List[Dict[str, Any]]) -> PlanUnits:
        """Render a plan's output rows into judgeable units."""
        rows = [_strip_internal(r) for r in output_rows if isinstance(r, dict)]

        produced_keys: List[str] = []
        if rows and self._sample_input:
            input_keys = set(self._sample_input[0].keys())
            produced_keys = [k for k in rows[0] if k not in input_keys]
        if not produced_keys and rows:
            produced_keys = list(rows[0].keys())

        def render(row: Dict[str, Any], budget: int) -> str:
            projected = {k: row[k] for k in produced_keys if k in row} or row
            return _truncate_to_tokens(
                json.dumps(projected, indent=2, default=str),
                budget,
                self.judge_model,
            )

        doc_units = None
        if self._sample_input and len(rows) == len(self._sample_input):
            doc_units = [render(r, self.max_unit_tokens) for r in rows]

        digest_rows = rows[: self.num_sample_docs]
        per_row_budget = max(
            200, self.max_digest_tokens // max(1, len(digest_rows))
        )
        digest = "\n\n".join(
            f"--- Output row {i + 1} ---\n{render(r, per_row_budget)}"
            for i, r in enumerate(digest_rows)
        ) or "(empty output)"
        digest = _truncate_to_tokens(
            digest, self.max_digest_tokens, self.judge_model
        )
        return PlanUnits(doc_units=doc_units, digest=digest)

    # ── rating (1-5) ───────────────────────────────────────────────

    def rate(self, units: PlanUnits) -> RatingResult:
        """Rate a plan's outputs 1-5 against the criteria, per unit."""
        from concurrent.futures import ThreadPoolExecutor

        tasks: List[Tuple[str, str]] = []
        if units.aligned:
            for i, unit in enumerate(units.doc_units):
                input_context = (
                    f"Input document:\n{self._input_excerpts[i]}"
                    if i < len(self._input_excerpts)
                    else "Input document: (unavailable)"
                )
                tasks.append((input_context, unit))
        else:
            tasks.append(
                (
                    f"Input documents:\n{self._digest_input_context}",
                    units.digest,
                )
            )

        def rate_one(task: Tuple[str, str]) -> Dict[str, Any]:
            input_context, output_text = task
            prompt = f"""Validation criteria:
{self.criteria}

{input_context}

Candidate output:
{output_text}

Rate how well the candidate output satisfies the validation criteria for this input, on a 1-5 scale:
1 = Very poor: fails the criteria almost entirely
2 = Poor: meets a small fraction of the criteria
3 = Fair: meets some criteria with clear gaps
4 = Good: meets most criteria with minor issues
5 = Excellent: fully satisfies the criteria

Provide an integer rating from 1 to 5 and a brief reason."""
            response = self._api().call_llm(
                self.judge_model,
                "moar_judge_rate",
                [{"role": "user", "content": prompt}],
                {"rating": "int", "reason": "str"},
                op_config=self._judge_op_config("moar_judge_rate"),
            )
            self._add_cost(response.total_cost)
            parsed = self._api().parse_llm_response(
                response.response, {"rating": "int", "reason": "str"}
            )[0]
            rating = min(5, max(1, int(parsed.get("rating", 3))))
            return {"rating": rating, "reason": parsed.get("reason", "")}

        per_unit: List[Dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            futures = [executor.submit(rate_one, t) for t in tasks]
            for f in futures:
                try:
                    per_unit.append(f.result())
                except Exception as e:
                    self.console.log(
                        f"[yellow]Judge rating call failed: {e}[/yellow]"
                    )

        if not per_unit:
            return RatingResult(mean=3.0, per_unit=[])
        mean = sum(u["rating"] for u in per_unit) / len(per_unit)
        return RatingResult(mean=mean, per_unit=per_unit)

    # ── placement within a bucket ──────────────────────────────────

    def place(
        self,
        new_units: PlanUnits,
        member_units: List[PlanUnits],
        seq: int,
    ) -> PlacementResult:
        """Slot a new plan among a bucket's members (best-to-worst order).

        Uses one batched ranking call per sample document covering the
        whole bucket when everything fits in the judge's context window;
        otherwise binary insertion with two-candidate calls. Existing
        members' relative order is never used as an output — only the new
        plan's position relative to them is read off.
        """
        if not member_units:
            return PlacementResult(slot=0, mode="default")

        per_doc = new_units.aligned and all(m.aligned for m in member_units)

        if self._window_fits(new_units, member_units, per_doc):
            votes, details = self._window_votes(new_units, member_units, per_doc, seq)
            if votes is not None:
                return PlacementResult(
                    slot=self._first_crossing(votes),
                    mode="window",
                    votes=votes,
                    details=details,
                )

        lo, hi = 0, len(member_units)
        details: List[Dict[str, Any]] = []
        while lo < hi:
            mid = (lo + hi) // 2
            beats, pair_details = self._pairwise_beats(
                new_units, member_units[mid], per_doc, seq, mid
            )
            details.extend(pair_details)
            if beats:
                hi = mid
            else:
                lo = mid + 1
        return PlacementResult(slot=lo, mode="binary", details=details)

    @staticmethod
    def _first_crossing(votes: List[bool]) -> int:
        """Insertion slot from best-to-worst votes (True = new plan wins).

        The new plan goes directly above the first member it beats; a
        non-monotone vote further down cannot move that member up, so it
        is ignored — existing order is frozen.
        """
        for i, beats in enumerate(votes):
            if beats:
                return i
        return len(votes)

    def _window_votes(
        self,
        new_units: PlanUnits,
        member_units: List[PlanUnits],
        per_doc: bool,
        seq: int,
    ) -> Tuple[Optional[List[bool]], List[Dict[str, Any]]]:
        """Majority per-member votes from batched full-window rankings."""
        from concurrent.futures import ThreadPoolExecutor

        if per_doc:
            rounds = [
                (d, self._doc_texts(new_units, member_units, d), self._doc_header(d))
                for d in range(len(new_units.doc_units))
            ]
        else:
            texts = [new_units.digest] + [m.digest for m in member_units]
            header = f"Input documents:\n{self._digest_input_context}"
            rounds = [
                (r, texts, header)
                for r in range(min(3, max(1, self.num_sample_docs)))
            ]

        def run_round(round_spec) -> Optional[Dict[str, Any]]:
            round_idx, texts, header = round_spec
            order = self._rank_call(texts, header, seed=(seq, round_idx))
            if order is None:
                return None
            new_pos = order.index(0)
            return {
                "round": round_idx,
                "order": order,
                "new_before": [
                    new_pos < order.index(j + 1)
                    for j in range(len(member_units))
                ],
            }

        results: List[Dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            futures = [executor.submit(run_round, r) for r in rounds]
            for f in futures:
                try:
                    result = f.result()
                    if result is not None:
                        results.append(result)
                except Exception as e:
                    self.console.log(
                        f"[yellow]Judge ranking round failed: {e}[/yellow]"
                    )

        if not results:
            return None, []

        votes = []
        for j in range(len(member_units)):
            wins = sum(1 for r in results if r["new_before"][j])
            votes.append(wins * 2 > len(results))
        return votes, results

    def _pairwise_beats(
        self,
        new_units: PlanUnits,
        member: PlanUnits,
        per_doc: bool,
        seq: int,
        mid: int,
    ) -> Tuple[bool, List[Dict[str, Any]]]:
        """Majority verdict of new-vs-member over per-doc two-candidate calls.

        A tie keeps the incumbent ahead.
        """
        from concurrent.futures import ThreadPoolExecutor

        if per_doc:
            rounds = [
                (
                    d,
                    [new_units.doc_units[d], member.doc_units[d]],
                    self._doc_header(d),
                )
                for d in range(len(new_units.doc_units))
            ]
        else:
            header = f"Input documents:\n{self._digest_input_context}"
            rounds = [
                (r, [new_units.digest, member.digest], header)
                for r in range(min(3, max(1, self.num_sample_docs)))
            ]

        def run_round(round_spec) -> Optional[bool]:
            round_idx, texts, header = round_spec
            order = self._rank_call(texts, header, seed=(seq, mid, round_idx))
            if order is None:
                return None
            return order.index(0) < order.index(1)

        outcomes: List[bool] = []
        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            futures = [executor.submit(run_round, r) for r in rounds]
            for f in futures:
                try:
                    result = f.result()
                    if result is not None:
                        outcomes.append(result)
                except Exception as e:
                    self.console.log(
                        f"[yellow]Judge comparison round failed: {e}[/yellow]"
                    )

        wins = sum(outcomes)
        beats = wins * 2 > len(outcomes) if outcomes else False
        details = [
            {"vs_member": mid, "wins": wins, "rounds": len(outcomes)}
        ]
        return beats, details

    # ── LLM plumbing ───────────────────────────────────────────────

    def _rank_call(
        self,
        unit_texts: List[str],
        context_header: str,
        seed: Tuple,
    ) -> Optional[List[int]]:
        """One batched ranking call; returns original indices best-first.

        Candidates are shuffled per call (seeded, deterministic) to break
        position bias; the returned order is mapped back to input indices.
        """
        rng = random.Random(hash(seed))
        perm = list(range(len(unit_texts)))
        rng.shuffle(perm)
        docs = [{"candidate_output": unit_texts[i]} for i in perm]

        criteria_text = f"""{self.criteria}

{context_header}

Order the candidate outputs by how well they satisfy the criteria above for the given input, best first."""

        try:
            ranking, cost = self._rank_operation()._batch_rank_documents(
                docs,
                criteria_text,
                "desc",
                self.judge_model,
                batch_label="MOAR judge",
            )
        except Exception as e:
            self.console.log(f"[yellow]Judge rank call failed: {e}[/yellow]")
            return None
        self._add_cost(cost)
        if ranking is None:
            return None
        return [perm[i] for i in ranking]

    def _doc_texts(
        self, new_units: PlanUnits, member_units: List[PlanUnits], d: int
    ) -> List[str]:
        return [new_units.doc_units[d]] + [m.doc_units[d] for m in member_units]

    def _doc_header(self, d: int) -> str:
        excerpt = (
            self._input_excerpts[d]
            if d < len(self._input_excerpts)
            else "(unavailable)"
        )
        return f"Input document:\n{excerpt}"

    def _window_fits(
        self,
        new_units: PlanUnits,
        member_units: List[PlanUnits],
        per_doc: bool,
    ) -> bool:
        unit_budget = self.max_unit_tokens if per_doc else self.max_digest_tokens
        input_budget = self.max_input_tokens if per_doc else 3 * self.max_input_tokens
        estimated = (
            (len(member_units) + 1) * unit_budget
            + input_budget
            + CONTEXT_MARGIN_TOKENS
        )
        return estimated <= self._context_budget()

    def _context_budget(self) -> int:
        try:
            from litellm import model_cost

            ctx = model_cost.get(self.judge_model, {}).get("max_input_tokens")
            if not ctx:
                ctx = model_cost.get(
                    self.judge_model.split("/")[-1], {}
                ).get("max_input_tokens")
        except Exception:
            ctx = None
        return min(ctx or DEFAULT_CONTEXT_TOKENS, MAX_CONTEXT_TOKENS)

    def _rank_operation(self):
        if self._rank_op is None:
            from docetl.operations.rank import RankOperation
            from docetl.runner import DSLRunner

            runner = DSLRunner(
                {
                    "default_model": self.judge_model,
                    "operations": [],
                    "pipeline": {"steps": [], "output": {"path": "/tmp/moar_judge.json"}},
                },
                max_threads=self.max_threads,
            )
            self._rank_op = RankOperation(
                runner,
                self._judge_op_config("moar_judge_rank"),
                default_model=self.judge_model,
                max_threads=self.max_threads,
            )
        return self._rank_op

    def _judge_op_config(self, name: str) -> Dict[str, Any]:
        return {
            "name": name,
            "type": "order",
            "prompt": self.criteria or "Judge candidate outputs.",
            "direction": "desc",
            "input_keys": ["candidate_output"],
            "batch_size": 16,
        }

    def _api(self):
        return self._rank_operation().runner.api

    def _add_cost(self, cost: float) -> None:
        with self._cost_lock:
            self.total_cost += cost or 0.0
