"""Bridge an AI-SQL query's semantic work to the DocETL plan IR.

The design premise (docs/design/ai-sql.md) is that AI functions compile
to ordinary DocETL operators, so the existing plan IR and MOAR optimize
them with no special-casing. This module makes that concrete: each
``SemanticStage`` of a compiled query becomes a standard pipeline config
that ``docetl.plan.lift`` accepts and MOAR can tune.

The relational stages are DuckDB's; only the semantic stages are LLM work
MOAR would optimize. Stages stay separate because a relational stage may
sit between two semantic ones (e.g. an ``ai_score`` comparison filter
before a SELECT-list map) — collapsing them would drop that filter.
Equijoin stages are two-input and excluded here; tuning a join query goes
through the whole-query path (run_sql + an eval function).
"""

from __future__ import annotations

from docetl.aisql.compile import CompiledQuery, SemanticStage


def to_pipeline_config(
    operations: list[dict], dataset_name: str = "aisql_input"
) -> dict:
    """A standalone DocETL pipeline config for a list of operations, with
    each op chained as its own step. Lifts via ``docetl.plan.lift`` and is
    the shape MOAR consumes."""
    steps = []
    prev = dataset_name
    for op in operations:
        step_name = f"step_{op['name']}"
        steps.append({"name": step_name, "input": prev, "operations": [op["name"]]})
        prev = step_name
    return {
        "datasets": {dataset_name: {"type": "memory", "path": []}},
        "operations": list(operations),
        "pipeline": {
            "steps": steps,
            "output": {"type": "file", "path": ""},
        },
    }


def semantic_pipelines(compiled: CompiledQuery) -> list[dict]:
    """One pipeline config per semantic stage of *compiled* — the LLM work
    available for plan-IR rewrites and MOAR optimization."""
    return [
        to_pipeline_config(stage.operations)
        for stage in compiled.stages
        if isinstance(stage, SemanticStage) and stage.operations
    ]
