from __future__ import annotations

import math
import os
import random
from datetime import datetime
from typing import Any

import yaml
from dotenv import load_dotenv

from docetl.reasoning_optimizer.directives import Directive
from docetl.runner import DSLRunner
from docetl.utils import extract_output_from_json


class Node:
    """MCTS node holding a pipeline config variant and its evaluation state."""

    _id_counter = 0

    @classmethod
    def get_next_id(cls) -> int:
        return cls._id_counter

    @classmethod
    def increment_id_counter(cls) -> int:
        new_id = cls._id_counter
        cls._id_counter += 1
        return new_id

    def __init__(
        self,
        yaml_file_path: str,
        parent: Node | None = None,
        c: float = 1.414,
        message_history=[],
        id: int | None = None,
        is_multi_instance: bool = False,
        console=None,
    ):
        from docetl.console import DOCETL_CONSOLE

        self.console = console if console is not None else DOCETL_CONSOLE
        self.yaml_file_path = yaml_file_path
        self.parsed_yaml = self._load_yaml()
        try:
            self.result_path: str | None = (
                self.parsed_yaml.get("pipeline", {}).get("output", {}).get("path")
            )
        except Exception:
            self.result_path = None
        self.on_frontier = False

        self._pipeline = None
        self.op_dict = {op["name"]: op for op in self.parsed_yaml.get("operations", [])}
        self.used_actions = {name: set() for name in self.op_dict}
        self.visits = 0
        self.value = 0
        self.parent = parent
        self.children = []
        self.c = c
        self.cost = -1.0
        self.scaled_cost = -1.0
        self.sample_result = []
        self.latest_action = None
        self.optimization_goal = None
        self.message_history = message_history
        self.memo = []
        self.is_multi_instance = is_multi_instance

        if id:
            self.id = id
        else:
            self.id = Node._id_counter
            Node._id_counter += 1

    @property
    def pipeline(self):
        if self._pipeline is None:
            from docetl.api import Pipeline
            self._pipeline = Pipeline.from_dict(self.parsed_yaml)
        return self._pipeline

    @property
    def op_to_step(self) -> dict[str, str]:
        mapping = {}
        for step in self.pipeline.steps:
            for op in step.operations:
                name = op if isinstance(op, str) else list(op.keys())[0]
                mapping[name] = step.name
        return mapping

    def execute_plan(self, max_threads: int | None = None) -> float:
        self.console.log(f"[dim]EXECUTING PLAN:[/dim] {self.yaml_file_path}")

        cwd = os.getcwd()
        env_file = os.path.join(cwd, ".env")
        if os.path.exists(env_file):
            load_dotenv(env_file)

        try:
            runner = DSLRunner(
                self.parsed_yaml,
                max_threads=max_threads,
                base_name=str(self.yaml_file_path).rsplit(".", 1)[0],
                yaml_file_suffix=os.path.basename(str(self.yaml_file_path)).split(".")[0],
            )
            runner.print_query_plan()
            runner.load()

            if runner.last_op_container:
                result_data, _, _ = runner.last_op_container.next()
                runner.save(result_data)

            self.cost = runner.total_cost
            runner.reset_env()

            try:
                self.sample_result = extract_output_from_json(self.yaml_file_path)[:1]
            except Exception as e:
                self.console.log(
                    f"[yellow]Error extracting output from JSON for {self.yaml_file_path}: {e}[/yellow]"
                )
                self.sample_result = []

            return self.cost

        except Exception as e:
            self.cost = -1  # Indicate failure
            self.value = -float("inf")

            # Log -inf occurrence for debugging
            self._log_inf_occurrence("execution_failure", str(e), self.yaml_file_path)

            raise Exception(f"Failed to execute plan {self.yaml_file_path}: {str(e)}")

    def _load_yaml(self) -> dict[str, Any]:
        try:
            with open(self.yaml_file_path, "r", encoding="utf-8") as file:
                return yaml.safe_load(file)
        except Exception as e:
            self.console.log(
                f"[yellow]Error loading YAML file {self.yaml_file_path}: {e}[/yellow]"
            )
            return {}

    def best_child(self) -> Node:
        def ucb(child: Node) -> float:
            if child.cost == -1 or child.visits == 0:
                return float("-inf")
            exploitation = child.value / child.visits
            exploration = self.c * math.sqrt(math.log(self.visits) / child.visits)
            return exploitation + exploration

        for child in self.children:
            self.console.log(
                f"[dim]Child {child.yaml_file_path}: visits = {child.visits}, value = {child.value}[/dim]"
            )

        ucb_values = [(child, ucb(child)) for child in self.children]
        max_ucb = max(ucb_values, key=lambda x: x[1])[1]
        tied_children = [child for child, ucb_val in ucb_values if ucb_val == max_ucb]
        return random.choice(tied_children)

    def add_child(self, child: Node):

        self.children.append(child)
        child.parent = self

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def mark_action_used(self, op_name, action: Directive):
        self.used_actions[op_name].add(action)

    def is_root(self) -> bool:
        return self.parent is None

    def update_value(self, value: float):
        if (
            value is None
            or (isinstance(value, float) and (value != value))
            or value == float("-inf")
        ):
            self.console.log(
                f"[yellow]⚠️ Skipping backpropagation of -inf / NaN value to node {self.get_id()}[/yellow]"
            )
            # Log -inf occurrence for debugging
            self._log_inf_occurrence(
                "backpropagation_skipped",
                f"Skipped backpropagation of value: {value}",
                self.yaml_file_path,
            )
            return
        self.value = self.value + value

    def update_visit(self):
        self.visits += 1

    def get_ucb(self) -> float:
        if self.visits == 0:
            return float("inf")
        if self.parent is None:
            return self.value / self.visits

        exploitation = self.value / self.visits
        exploration = self.c * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration

    def get_id(self) -> int:
        return self.id

    def set_id_to_counter(self):
        old_id = self.id
        new_id = self.increment_id_counter()
        self._rename_files_for_new_id(old_id, new_id)
        self.id = new_id
        return self.id

    def _log_inf_occurrence(
        self, failure_type: str, error_message: str, yaml_path: str
    ):
        try:
            log_dir = os.path.join(os.path.dirname(yaml_path), "inf_logs")
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, "inf_occurrences.txt")

            parent_id = getattr(self.parent, "id", "root") if self.parent else "root"
            log_entry = (
                f"\n{'='*80}\n"
                f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"Node ID: {self.id} | Parent ID: {parent_id}\n"
                f"Latest Action: {self.latest_action}\n"
                f"Failure Type: {failure_type}\n"
                f"YAML Path: {yaml_path}\n"
                f"Error Message: {error_message}\n"
                f"{'='*80}\n"
            )
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(log_entry)
        except Exception as log_error:
            self.console.log(
                f"[yellow]Warning: Failed to log -inf occurrence: {log_error}[/yellow]"
            )

    def _rename_files_for_new_id(self, old_id, new_id):
        try:
            if os.path.exists(self.yaml_file_path):
                old_yaml_path = self.yaml_file_path
                new_yaml_path = old_yaml_path.replace(
                    f"_{old_id}.yaml", f"_{new_id}.yaml"
                )
                os.rename(old_yaml_path, new_yaml_path)
                self.yaml_file_path = new_yaml_path
        except Exception as e:
            self.console.log(
                f"[yellow]Warning: Could not rename YAML file from {old_id} to {new_id}: {e}[/yellow]"
            )

        try:
            if self.result_path and os.path.exists(self.result_path):
                old_result_path = self.result_path
                new_result_path = old_result_path.replace(
                    f"_{old_id}.json", f"_{new_id}.json"
                )
                os.rename(old_result_path, new_result_path)
                self.result_path = new_result_path

                if hasattr(self, "parsed_yaml") and self.parsed_yaml:
                    self.parsed_yaml["pipeline"]["output"]["path"] = new_result_path
                    with open(self.yaml_file_path, "w") as f:
                        yaml.dump(
                            self.parsed_yaml,
                            f,
                            default_flow_style=False,
                            allow_unicode=True,
                            sort_keys=False,
                        )
        except Exception as e:
            self.console.log(
                f"[yellow]Warning: Could not rename result file from {old_id} to {new_id}: {e}[/yellow]"
            )

    def add_memo_entry(self, directive_name: str, target_operator: str):
        self.memo.append((directive_name, target_operator))

    def get_optimization_path(self) -> str:
        if not self.memo:
            return "ROOT"

        path_parts = ["ROOT"]
        for directive, target_op in self.memo:
            path_parts.append(f"{directive}({target_op})")

        return " → ".join(path_parts)

    def get_exploration_tree_summary(
        self, root: Node, node_accuracies: dict["Node", float] | None = None
    ) -> str:
        successful, failed = self._collect_exploration_paths(root, node_accuracies)

        summary_parts = [f"CURRENT POSITION: {self.get_optimization_path()}"]

        if successful:
            summary_parts.append(
                f"\nSUCCESSFUL EXPLORATIONS ({len(successful)} total):"
            )
            for i, path in enumerate(sorted(successful, key=self._path_sort_key)):
                summary_parts.append(f"  {i+1}. {path}")

        return "\n".join(summary_parts)

    def _collect_exploration_paths(
        self, root: Node, node_accuracies: dict["Node", float] | None,
    ) -> tuple[list[str], list[str]]:
        successful = []
        failed = []

        def traverse(node, current_path="ROOT"):
            if node != root:
                if hasattr(node, "cost") and node.cost != -1:
                    label = f"cost: ${node.cost:.2f}"
                    if node_accuracies and node in node_accuracies:
                        label += f", accuracy: {node_accuracies[node]:.3f}"
                    successful.append(f"{current_path} ({label})")
                else:
                    failed.append(f"{current_path} (failed)")

            for child in node.children:
                if child.memo:
                    d, t = child.memo[-1]
                    child_path = f"{current_path} → {d}({t})"
                else:
                    action = child.latest_action.name if child.latest_action else "unknown"
                    child_path = f"{current_path} → {action}"
                traverse(child, child_path)

        traverse(root)
        return successful, failed

    @staticmethod
    def _path_sort_key(path: str) -> tuple[float, float]:
        if "cost: $" not in path:
            return (0, float("inf"))
        try:
            cost_part = path.split("cost: $")[1]
            if ", accuracy:" in cost_part:
                cost_str = cost_part.split(", accuracy:")[0]
                acc_str = cost_part.split(", accuracy:")[1].split(")")[0]
                return (-float(acc_str), float(cost_str))
            return (0, float(cost_part.split(")")[0]))
        except (ValueError, IndexError):
            return (0, float("inf"))

    def get_memo_for_llm(
        self, root_node: Node, node_accuracies: dict["Node", float] | None = None
    ) -> str:
        return self.get_exploration_tree_summary(root_node, node_accuracies)

    def delete(self, selected_node_final_id=None):
        if self.parent and self in self.parent.children:
            self.parent.children.remove(self)

        if self.is_multi_instance and selected_node_final_id is not None:
            self._backup_multi_instance_files(selected_node_final_id)
        else:
            self._delete_files_permanently()

        self.parent = None
        self.children = []
        self.parsed_yaml = {}
        self._pipeline = None
        self.message_history = []
        self.memo = []
        self.sample_result = []

    def _backup_multi_instance_files(self, selected_node_final_id):
        try:
            current_id_str = str(self.id)
            if "-" in current_id_str:
                instantiation_num = current_id_str.split("-")[1]
                new_backup_id = f"{selected_node_final_id}-{instantiation_num}"
            else:
                new_backup_id = f"{selected_node_final_id}-backup"

            if os.path.exists(self.yaml_file_path):
                yaml_dir = os.path.dirname(self.yaml_file_path)
                backup_dir = os.path.join(yaml_dir, "backup_plans")
                os.makedirs(backup_dir, exist_ok=True)

                yaml_filename = os.path.basename(self.yaml_file_path)
                new_yaml_filename = yaml_filename.replace(
                    f"_{current_id_str}.yaml", f"_{new_backup_id}.yaml"
                )
                backup_yaml_path = os.path.join(backup_dir, new_yaml_filename)

                if os.path.exists(self.yaml_file_path):
                    os.rename(self.yaml_file_path, backup_yaml_path)
                    self.console.log(
                        f"[dim]Moved YAML to backup:[/dim] {self.yaml_file_path} → {backup_yaml_path}"
                    )

                if self.result_path and os.path.exists(self.result_path):
                    result_filename = os.path.basename(self.result_path)
                    new_result_filename = result_filename.replace(
                        f"_{current_id_str}.json", f"_{new_backup_id}.json"
                    )
                    backup_result_path = os.path.join(backup_dir, new_result_filename)

                    os.rename(self.result_path, backup_result_path)
                    self.console.log(
                        f"[dim]Moved result to backup:[/dim] {self.result_path} → {backup_result_path}"
                    )

        except Exception as e:
            self.console.log(
                f"[yellow]Warning: Could not backup files for multi-instance node {self.id}: {e}[/yellow]"
            )
            self._delete_files_permanently()

    def _delete_files_permanently(self):
        try:
            if os.path.exists(self.yaml_file_path) and str(
                self.yaml_file_path
            ).endswith((".yaml", ".yml")):
                if any(char.isdigit() for char in os.path.basename(str(self.yaml_file_path))):
                    os.remove(self.yaml_file_path)
                    self.console.log(
                        f"[dim]Deleted generated YAML file:[/dim] {self.yaml_file_path}"
                    )
        except Exception as e:
            self.console.log(
                f"[yellow]Warning: Could not delete YAML file {self.yaml_file_path}: {e}[/yellow]"
            )

        try:
            if self.result_path and os.path.exists(self.result_path):
                if any(char.isdigit() for char in os.path.basename(self.result_path)):
                    os.remove(self.result_path)
                    self.console.log(
                        f"[dim]Deleted generated result file:[/dim] {self.result_path}"
                    )
        except Exception as e:
            self.console.log(
                f"[yellow]Warning: Could not delete result file {self.result_path}: {e}[/yellow]"
            )
