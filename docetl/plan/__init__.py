"""Typed relational logical-plan IR over DocETL pipeline configs.

The config dict stays the canonical serialized plan (checkpoint hashing
and Frame memoization already key on it); this package lifts it into a
typed DAG and lowers it back deterministically::

    config ──lift──▶ LogicalPlan ──apply_rules──▶ LogicalPlan ──lower──▶ config'

``lower(lift(c))`` is deep- and yaml-dump-equal to ``c``, and when no
rewrite fires the original config object is returned untouched, so
checkpoint hashes can never churn on a no-op.

Engine-readiness notes
----------------------
The IR is modeled in spirit on DataFusion's LogicalPlan so a future
Rust executor can consume the same plans:

- ``ScanNode`` ↔ ``TableScan``; selection/projection/aggregate-shaped
  ops (per their ``cardinality`` trait) ↔ the native Filter/Projection/
  Aggregate nodes with the LLM work expressed as Extension nodes
  (``UserDefinedLogicalNode``).
- The operator traits (cardinality, fields_read/written, row-locality,
  order preservation) become node properties driving custom
  ``OptimizerRule`` impls — DataFusion's built-in pushdowns treat
  extension nodes conservatively, so these rules are the spec for the
  custom ones a ``docetl-engine`` crate would register.
- The deterministic config round-trip is the interchange contract; no
  Substrait serialization is attempted here.
"""

from docetl.operations.base import Cardinality
from docetl.plan.explain import format_plan
from docetl.plan.lift import lift
from docetl.plan.lower import lower
from docetl.plan.nodes import JoinNode, OpaqueNode, PlanNode, ScanNode
from docetl.plan.plan import LogicalPlan, PlanIssue, StepGroup
from docetl.plan.prepare import configured_rules, prepare_config
from docetl.plan.rewrite import (
    AppliedRewrite,
    RewriteRule,
    all_rules,
    apply_rewrites_to_config,
    apply_rules,
    default_rules,
    resolve_rules,
)
from docetl.plan.schema import output_schema, propagate_schemas
from docetl.plan.validate import InvalidCandidatePlan, validate, validate_config

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
    "format_plan",
    "lift",
    "lower",
    "output_schema",
    "prepare_config",
    "propagate_schemas",
    "resolve_rules",
    "validate",
    "validate_config",
]
