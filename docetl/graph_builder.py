"""Build the execution DAG from a typed Pipeline and compute operation hashes."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

from docetl.containers import OpContainer, StepBoundary
from docetl.utils import op_ref_name

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
            _add_scan_operation(runner, step_dict, step.input)
        elif step.operations and isinstance(step.operations[0], dict):
            _add_equijoin_operation(runner, step_dict)
        else:
            _add_scan_operation(runner, step_dict, "__empty__")

        _add_step_operations(runner, step_dict)
        _add_step_boundary(runner, step_dict)


def _validate_step(step: dict) -> None:
    assert "name" in step, f"Step {step} does not have a name"
    assert "operations" in step, f"Step {step} does not have `operations`"


def _add_scan_operation(runner: DSLRunner, step: dict, dataset_name: str) -> None:
    runner.last_op_container = _make_scan_container(runner, step["name"], dataset_name)


def _make_scan_container(
    runner: DSLRunner, step_name: str, dataset_name: str
) -> OpContainer:
    key = f"{step_name}/scan_{dataset_name}"
    container = OpContainer(
        key,
        runner,
        {"type": "scan", "dataset_name": dataset_name, "name": f"scan_{dataset_name}"},
    )
    runner.op_container_map[key] = container
    if runner.last_op_container:
        container.add_child(runner.last_op_container)
    return container


def _add_equijoin_operation(runner: DSLRunner, step: dict) -> None:
    equijoin_op_name = op_ref_name(step["operations"][0])
    join_cfg = list(step["operations"][0].values())[0]
    left_name, right_name = join_cfg["left"], join_cfg["right"]

    left_scan = _make_scan_container(runner, step["name"], left_name)
    right_scan = _make_scan_container(runner, step["name"], right_name)

    equijoin = OpContainer(
        f"{step['name']}/{equijoin_op_name}",
        runner,
        runner.find_operation(equijoin_op_name),
        left_name=left_name,
        right_name=right_name,
    )
    equijoin.add_child(left_scan)
    equijoin.add_child(right_scan)

    runner.last_op_container = equijoin
    runner.op_container_map[f"{step['name']}/{equijoin_op_name}"] = equijoin


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
    """Compute checkpoint-invalidation hashes for every operation.

    An operation's hash covers everything that determines its output: its
    own config (with the effective model resolved), every upstream op in
    the same step — including equijoin entries — and the full lineage of
    the step's input: the input dataset's config (for memory datasets,
    the data itself) or, when the input is a previous step, that step's
    final hash. Hashes are built from derived views; the caller's config
    dicts are never mutated.

    Fills ``runner.step_op_hashes`` in place so references held elsewhere
    (e.g. the CheckpointStore) stay valid across recomputation.
    """
    runner.step_op_hashes.clear()
    if not runner.intermediate_dir:
        return  # hashes are only consumed by the checkpoint store

    datasets = runner.config.get("datasets", {})
    system_prompt = runner.pipeline.other_config.get("system_prompt", {})
    step_final_hash: dict[str, str] = {}

    def effective(op_cfg: dict) -> dict:
        if "model" in op_cfg:
            return op_cfg
        return {**op_cfg, "model": runner.default_model}

    def input_token(name: str | None) -> dict | None:
        if name is None:
            return None
        if name in step_final_hash:
            return {"step": name, "hash": step_final_hash[name]}
        return {"dataset": name, "config": datasets.get(name)}

    for step in runner.pipeline.steps:
        # Fold incrementally so each chain element (including memory-dataset
        # contents in the input token) is serialized exactly once per step.
        digest = hashlib.sha256()
        for element in ({"system_prompt": system_prompt}, input_token(step.input)):
            digest.update(json.dumps(element, sort_keys=True, default=str).encode())

        for entry in step.operations:
            op_name = op_ref_name(entry)
            if isinstance(entry, str):
                element = effective(runner._op_map[op_name])
            else:
                join_cfg = entry[op_name]
                element = {
                    "equijoin": effective(runner._op_map[op_name]),
                    "left": input_token(join_cfg.get("left")),
                    "right": input_token(join_cfg.get("right")),
                }
            digest.update(json.dumps(element, sort_keys=True, default=str).encode())
            runner.step_op_hashes[step.name][op_name] = digest.copy().hexdigest()

        step_final_hash[step.name] = digest.hexdigest()
