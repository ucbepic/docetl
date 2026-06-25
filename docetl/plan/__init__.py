"""Typed relational logical-plan IR over DocETL pipeline configs.

The config dict stays the canonical serialized plan (checkpoint hashing
and Frame memoization already key on it); this package lifts it into a
typed DAG and lowers it back deterministically::

    config ──lift──▶ LogicalPlan ──apply_rules──▶ LogicalPlan ──lower──▶ config'

``lower(lift(c))`` is deep- and yaml-dump-equal to ``c``, and when no
rewrite fires the original config object is returned untouched, so
checkpoint hashes can never churn on a no-op.
"""

from docetl import _config
from docetl.operations.base import Cardinality
from docetl.plan.analysis import (
    InvalidCandidatePlan,
    output_schema,
    propagate_schemas,
    validate,
    validate_config,
)
from docetl.plan.ir import (
    JoinNode,
    LogicalPlan,
    OpaqueNode,
    PlanIssue,
    PlanNode,
    ScanNode,
    StepGroup,
)
from docetl.plan.lift import lift
from docetl.plan.lower import lower
from docetl.plan.rewrite import (
    AppliedRewrite,
    RewriteRule,
    all_rules,
    apply_rewrites_to_config,
    apply_rules,
    default_rules,
    resolve_rules,
)


def configured_rules(config: dict):
    """The rewrite rules a config's ``plan_rewrites`` setting selects,
    falling back to the ``docetl.plan_rewrites`` module global."""
    spec = config.get("plan_rewrites", _config.plan_rewrites)
    if not spec:
        return []
    return resolve_rules(spec)


__all__ = [
    "AppliedRewrite",
    "Cardinality",
    "InvalidCandidatePlan",
    "JoinNode",
    "LogicalPlan",
    "OpaqueNode",
    "PlanIssue",
    "PlanNode",
    "RewriteRule",
    "ScanNode",
    "StepGroup",
    "all_rules",
    "apply_rewrites_to_config",
    "apply_rules",
    "configured_rules",
    "default_rules",
    "lift",
    "lower",
    "output_schema",
    "propagate_schemas",
    "resolve_rules",
    "validate",
    "validate_config",
]
