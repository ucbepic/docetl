"""Build the execution DAG from a typed Pipeline and compute operation hashes."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from typing import TYPE_CHECKING

from docetl.containers import OpContainer, StepBoundary

if TYPE_CHECKING:
    from docetl.runner import DSLRunner


def build_operation_graph(runner: DSLRunner) -> None:
    runner.op_container_map = {}
    runner.last_op_container = None
    runner._op_map = {op["name"]: op for op in runner._raw_ops_list}

    for step in runner.pipeline.steps:
        step_dict = {k: v for k, v in step.dict().items() if v is not None}
        _validate_step(step_dict)

        if step.input:
            _add_scan_operation(runner, step_dict)
        elif step.operations and isinstance(step.operations[0], dict):
            _add_equijoin_operation(runner, step_dict)
        else:
            _add_empty_scan_operation(runner, step_dict)

        _add_step_operations(runner, step_dict)
        _add_step_boundary(runner, step_dict)


def _validate_step(step: dict) -> None:
    assert "name" in step, f"Step {step} does not have a name"
    assert "operations" in step, f"Step {step} does not have `operations`"


def _add_scan_operation(runner: DSLRunner, step: dict) -> None:
    scan_op_container = OpContainer(
        f"{step['name']}/scan_{step['input']}",
        runner,
        {
            "type": "scan",
            "dataset_name": step["input"],
            "name": f"scan_{step['input']}",
        },
    )
    runner.op_container_map[f"{step['name']}/scan_{step['input']}"] = (
        scan_op_container
    )
    if runner.last_op_container:
        scan_op_container.add_child(runner.last_op_container)
    runner.last_op_container = scan_op_container


def _add_empty_scan_operation(runner: DSLRunner, step: dict) -> None:
    dataset_name = "__empty__"
    scan_op_container = OpContainer(
        f"{step['name']}/scan_{dataset_name}",
        runner,
        {
            "type": "scan",
            "dataset_name": dataset_name,
            "name": f"scan_{dataset_name}",
        },
    )
    runner.op_container_map[f"{step['name']}/scan_{dataset_name}"] = scan_op_container
    if runner.last_op_container:
        scan_op_container.add_child(runner.last_op_container)
    runner.last_op_container = scan_op_container


def _add_equijoin_operation(runner: DSLRunner, step: dict) -> None:
    equijoin_operation_name = list(step["operations"][0].keys())[0]
    left_dataset_name = list(step["operations"][0].values())[0]["left"]
    right_dataset_name = list(step["operations"][0].values())[0]["right"]

    left_scan_op_container = OpContainer(
        f"{step['name']}/scan_{left_dataset_name}",
        runner,
        {
            "type": "scan",
            "dataset_name": left_dataset_name,
            "name": f"scan_{left_dataset_name}",
        },
    )
    if runner.last_op_container:
        left_scan_op_container.add_child(runner.last_op_container)
    right_scan_op_container = OpContainer(
        f"{step['name']}/scan_{right_dataset_name}",
        runner,
        {
            "type": "scan",
            "dataset_name": right_dataset_name,
            "name": f"scan_{right_dataset_name}",
        },
    )
    if runner.last_op_container:
        right_scan_op_container.add_child(runner.last_op_container)
    equijoin_op_container = OpContainer(
        f"{step['name']}/{equijoin_operation_name}",
        runner,
        runner.find_operation(equijoin_operation_name),
        left_name=left_dataset_name,
        right_name=right_dataset_name,
    )

    equijoin_op_container.add_child(left_scan_op_container)
    equijoin_op_container.add_child(right_scan_op_container)

    runner.last_op_container = equijoin_op_container
    runner.op_container_map[f"{step['name']}/{equijoin_operation_name}"] = (
        equijoin_op_container
    )
    runner.op_container_map[f"{step['name']}/scan_{left_dataset_name}"] = (
        left_scan_op_container
    )
    runner.op_container_map[f"{step['name']}/scan_{right_dataset_name}"] = (
        right_scan_op_container
    )


def _add_step_operations(runner: DSLRunner, step: dict) -> None:
    is_equijoin = (
        step.get("input") is None
        and step["operations"]
        and isinstance(step["operations"][0], dict)
    )
    op_start_idx = 1 if is_equijoin else 0

    for operation_name in step["operations"][op_start_idx:]:
        if not isinstance(operation_name, str):
            raise ValueError(
                f"Operation {operation_name} in step {step['name']} should be a string. "
                "If you intend for it to be an equijoin, don't specify an input in the step."
            )

        op_container = OpContainer(
            f"{step['name']}/{operation_name}",
            runner,
            runner.find_operation(operation_name),
        )
        op_container.add_child(runner.last_op_container)
        runner.last_op_container = op_container
        runner.op_container_map[f"{step['name']}/{operation_name}"] = op_container


def _add_step_boundary(runner: DSLRunner, step: dict) -> None:
    step_boundary = StepBoundary(
        f"{step['name']}/boundary",
        runner,
        {"type": "step_boundary", "name": f"{step['name']}/boundary"},
    )
    step_boundary.add_child(runner.last_op_container)
    runner.op_container_map[f"{step['name']}/boundary"] = step_boundary
    runner.last_op_container = step_boundary


def compute_operation_hashes(runner: DSLRunner) -> None:
    runner.step_op_hashes = defaultdict(dict)

    for step in runner.pipeline.steps:
        for idx, entry in enumerate(step.operations):
            op_name = entry if isinstance(entry, str) else list(entry.keys())[0]

            all_ops_until_and_including_current = (
                [runner._op_map[prev] for prev in step.operations[:idx]
                 if isinstance(prev, str)]
                + [runner._op_map[op_name]]
                + [runner.pipeline.other_config.get("system_prompt", {})]
            )

            for op_cfg in all_ops_until_and_including_current:
                if isinstance(op_cfg, dict) and "model" not in op_cfg:
                    op_cfg["model"] = runner.default_model

            all_ops_str = json.dumps(all_ops_until_and_including_current)
            runner.step_op_hashes[step.name][op_name] = hashlib.sha256(
                all_ops_str.encode()
            ).hexdigest()
