"""Single-lift validate-and-rewrite entry point.

``prepare_config`` is what callers that need both static validation and
the configured rewrites should use (MOAR candidate instantiation): it
lifts once, validates, applies exactly the rules the config's own
``plan_rewrites`` setting asks for (same gate semantics as
``DSLRunner._set_config``, including the ``docetl.plan_rewrites`` module
default), and lowers only if something fired.
"""

from __future__ import annotations

from typing import Any

from docetl import _config
from docetl.plan.lift import lift
from docetl.plan.lower import lower
from docetl.plan.plan import PlanIssue
from docetl.plan.rewrite import AppliedRewrite, apply_rules, could_fire, resolve_rules
from docetl.plan.validate import validate


def configured_rules(config: dict[str, Any]):
    """The rewrite rules a config's ``plan_rewrites`` setting selects,
    falling back to the ``docetl.plan_rewrites`` module global."""
    spec = config.get("plan_rewrites", _config.plan_rewrites)
    if not spec:
        return []
    return resolve_rules(spec)


def prepare_config(
    config: dict[str, Any],
) -> tuple[dict[str, Any], list[PlanIssue], list[AppliedRewrite]]:
    """Validate *config* and apply its configured rewrites in one lift.

    Returns ``(config', issues, applied)``. On error-level issues the
    original config is returned unrewritten (callers decide whether to
    reject); otherwise rewrites run and ``config'`` is the lowered
    result — or the original object when nothing fired.
    """
    rules = configured_rules(config)
    plan = lift(config)
    issues = validate(plan)
    if any(issue.level == "error" for issue in issues):
        return config, issues, []
    if not rules or not could_fire(config, rules):
        return config, issues, []
    applied = apply_rules(plan, rules=rules)
    if not applied:
        return config, issues, []
    return lower(plan), issues, applied
