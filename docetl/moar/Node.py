from __future__ import annotations

import math
import os
import random
from datetime import datetime
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv

from docetl.reasoning_optimizer.directives import Directive
from docetl.runner import DSLRunner
from docetl.utils import extract_output_from_json


class Node:
    """
    A Node class for Monte Carlo Tree Search that represents a state in the search tree.

    Each node holds:
    - YAML file path and parsed content
    - Visit count and value for UCB calculation
    - Parent and children relationships
    - Methods for tree traversal and expansion
    """

    # A class-level counter for unique IDs
    _id_counter = 0

    @classmethod
    def get_next_id(cls) -> int:
        """
        Get the next available ID from the counter without incrementing it.

        Returns:
            int: The next ID that would be assigned
        """
        return cls._id_counter

    @classmethod
    def increment_id_counter(cls) -> int:
        """
        Increment the ID counter and return the new ID.

        Returns:
            int: The newly assigned ID
        """
        new_id = cls._id_counter
        cls._id_counter += 1
        return new_id

    def __init__(
        self,
        yaml_file_path: str,
        parent: Optional[Node] = None,
        c: float = 1.414,
        message_history=[],
        id: Optional[int] = None,
        is_multi_instance: bool = False,
        console=None,
    ):
        """
        Initialize a Node with YAML file information.

        Args:
            yaml_file_path: Path to the YAML configuration file
            parent: Parent node in the search tree
            c: Exploration constant for UCB calculation (default: sqrt(2))
            is_multi_instance: Whether this node is a multi-instance candidate (default: False)
            console: Console instance for logging (default: None, uses DOCETL_CONSOLE)
        """
        from docetl.console import DOCETL_CONSOLE

        self.console = console if console is not None else DOCETL_CONSOLE
        self.yaml_file_path = yaml_file_path
        self.parsed_yaml = self._load_yaml()
        # Where the JSON results will be written (if defined in the YAML). This is
        # useful later for evaluation without having to guess filenames.
        try:
            self.result_path: str | None = (
                self.parsed_yaml.get("pipeline", {}).get("output", {}).get("path")
            )
        except Exception:
            self.result_path = None
        self.on_frontier = False
        self.used_actions = {}

        self.op_dict = {}  # Dict: op_name -> op
        for op in self.parsed_yaml["operations"]:
            op_name = op["name"]
            self.op_dict[op_name] = op
            self.used_actions[op_name] = set()
        self.visits = 0
        self.value = 0
        self.parent = parent
        self.children = []
        self.c = c  # Exploration constant for UCB
        self.cost = -1.0
        self.scaled_cost = -1.0  # Scaled cost in [0,1] range for reward calculations
        self.sample_result = []
        self.latest_action = None  # Latest action that led to this node
        self.optimization_goal = (
            None  # The optimization goal used for this node (e.g., 'acc', 'cost')
        )

        # Message history from root to this node (accumulated LLM conversations)
        self.message_history = message_history

        # Memo list to track (directive, target_operator) pairs from root to this node
        self.memo = []

        # Flag to indicate if this is a multi-instance candidate
        self.is_multi_instance = is_multi_instance

        # Assign a unique ID to this node
        if id:
            self.id = id
        else:
            self.id = Node._id_counter
            Node._id_counter += 1

    def execute_plan(self, max_threads: Optional[int] = None) -> tuple[float, list]:
        """
        This method execute the query plan by running the YAML file with docetl.

        Args:
            max_threads (Optional[int]): Maximum number of threads to use for running operations.

        Returns:
            tuple[float, list]: A tuple containing (total_cost, result_data)

        Raises:
            Exception: If the pipeline execution fails.
        """

        self.console.log(f"[dim]EXECUTING PLAN:[/dim] {self.yaml_file_path}")

        # Get the current working directory (where the user called the command)
        cwd = os.getcwd()

        # Load .env file from the current working directory if it exists
        env_file = os.path.join(cwd, ".env")
        if os.path.exists(env_file):
            load_dotenv(env_file)

        try:
            runner = DSLRunner.from_yaml(self.yaml_file_path, max_threads=max_threads)

            # Print the query plan
            runner.print_query_plan()

            # Load datasets and execute the pipeline
            runner.load()

            # Execute the pipeline and get the result data
            if runner.last_op_container:
                result_data, _, _ = runner.last_op_container.next()
                runner.save(result_data)
            else:
                result_data = []

            # Get the total cost
            total_cost = runner.total_cost

            # Reset the environment
            runner.reset_env()

            self.cost = total_cost

            try:
                self.sample_result = extract_output_from_json(self.yaml_file_path)[:1]
            except Exception as e:
                self.console.log(
                    f"[yellow]Error extracting output from JSON for {self.yaml_file_path}: {e}[/yellow]"
                )
                self.sample_result = []

            return total_cost

        except Exception as e:
            self.cost = -1  # Indicate failure
            self.value = -float("inf")

            # Log -inf occurrence for debugging
            self._log_inf_occurrence("execution_failure", str(e), self.yaml_file_path)

            raise Exception(f"Failed to execute plan {self.yaml_file_path}: {str(e)}")

    def _load_yaml(self) -> Dict[str, Any]:
        """
        Load and parse the YAML file.

        Returns:
            Parsed YAML content as a dictionary
        """

        try:
            with open(self.yaml_file_path, "r", encoding="utf-8") as file:
                return yaml.safe_load(file)
        except Exception as e:
            self.console.log(
                f"[yellow]Error loading YAML file {self.yaml_file_path}: {e}[/yellow]"
            )
            return {}

    def best_child(self) -> Node:
        """
        Return the child with the highest UCB (Upper Confidence Bound) value.
        If there are ties, randomly select among the tied children.

        UCB formula: value/visits + c * sqrt(ln(parent_visits) / visits)

        Returns:
            Child node with highest UCB, or None if no children exist
        """

        def ucb(child: Node) -> float:
            if child.cost == -1 or child.visits == 0:
                return float("-inf")
            exploitation = child.value / child.visits
            exploration = self.c * math.sqrt(math.log(self.visits) / child.visits)
            return exploitation + exploration

        # Print visits and value for each child
        for child in self.children:
            self.console.log(
                f"[dim]Child {child.yaml_file_path}: visits = {child.visits}, value = {child.value}[/dim]"
            )

        # Calculate UCB values for all children
        ucb_values = [(child, ucb(child)) for child in self.children]

        # Find the maximum UCB value
        max_ucb = max(ucb_values, key=lambda x: x[1])[1]

        # Find all children with the maximum UCB value (ties)
        tied_children = [child for child, ucb_val in ucb_values if ucb_val == max_ucb]

        # Randomly select among tied children
        return random.choice(tied_children)

    def add_child(self, child: Node):
        """
        Add a new child node during tree expansion.

        Args:
            yaml_file_path: Path to the YAML file for the new child node

        Returns:
            The newly created child node
        """

        self.children.append(child)
        child.parent = self

    def is_leaf(self) -> bool:
        """
        Check if this node is a leaf (has no children).

        Returns:
            True if the node has no children, False otherwise
        """
        return len(self.children) == 0

    def mark_action_used(self, op_name, action: Directive):
        """
        Mark a rewrite action as used.

        Args:
            action: The action identifier to mark as used
        """
        self.used_actions[op_name].add(action)

    def is_root(self) -> bool:
        """
        Check if this node is the root (has no parent).

        Returns:
            True if the node has no parent, False otherwise
        """
        return self.parent is None

    def update_value(self, value: float):
        """
        Update the node's value (typically after a simulation).

        Args:
            value: The value to add to the current node value
        """
        # Guard against NaN and -inf values to prevent corruption of node.value
        # Don't backpropagate -inf (failed evaluations) or NaN to parent nodes
        if (
            value is None
            or (isinstance(value, float) and (value != value))
            or value == float("-inf")
        ):  # NaN or -inf check
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
        """
        Update the node's visit by 1 (typically after a simulation).
        """
        self.visits += 1

    def get_ucb(self) -> float:
        """
        Calculate the UCB value for this node.

        Returns:
            UCB value for this node
        """
        if self.visits == 0:
            return float("inf")
        if self.parent is None:
            return self.value / self.visits

        exploitation = self.value / self.visits
        exploration = self.c * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration

    def get_id(self) -> int:
        """
        Return the unique identifier for this node.
        Returns:
            int: The unique ID of the node
        """
        return self.id

    def set_id_to_counter(self):
        """
        Change this node's ID to the next available counter ID.
        This is used after selecting the best multi-instance candidate.
        Also renames the associated files to match the new ID.

        Returns:
            int: The new ID assigned to this node
        """
        old_id = self.id
        new_id = self.increment_id_counter()

        # Rename files to match the new ID
        self._rename_files_for_new_id(old_id, new_id)

        self.id = new_id
        return self.id

    def _log_inf_occurrence(
        self, failure_type: str, error_message: str, yaml_path: str
    ):
        """
        Log -inf occurrences to a dedicated log file for debugging.

        Args:
            failure_type: Type of failure (e.g., "execution_failure", "evaluation_failure")
            error_message: The error message that caused the failure
            yaml_path: Path to the YAML file that failed
        """
        try:
            # Create log directory if it doesn't exist
            log_dir = os.path.join(os.path.dirname(yaml_path), "inf_logs")
            os.makedirs(log_dir, exist_ok=True)

            # Create log file path
            log_file = os.path.join(log_dir, "inf_occurrences.txt")

            # Get current timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Get node information
            node_id = getattr(self, "id", "unknown")
            latest_action = getattr(self, "latest_action", "unknown")
            parent_id = getattr(self.parent, "id", "root") if self.parent else "root"

            # Format log entry
            log_entry = f"""
                {'='*80}
                Timestamp: {timestamp}
                Node ID: {node_id}
                Parent ID: {parent_id}
                Latest Action: {latest_action}
                Failure Type: {failure_type}
                YAML Path: {yaml_path}
                Error Message: {error_message}
                {'='*80}
                """

            # Append to log file
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(log_entry)

        except Exception as log_error:
            self.console.log(
                f"[yellow]Warning: Failed to log -inf occurrence: {log_error}[/yellow]"
            )

    def _rename_files_for_new_id(self, old_id, new_id):
        """
        Rename the YAML and output files to match the new node ID.

        Args:
            old_id: The old node ID (e.g., "7-2")
            new_id: The new node ID (e.g., 8)
        """
        try:
            # Rename YAML file
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
            # Rename output JSON file
            if self.result_path and os.path.exists(self.result_path):
                old_result_path = self.result_path
                new_result_path = old_result_path.replace(
                    f"_{old_id}.json", f"_{new_id}.json"
                )
                os.rename(old_result_path, new_result_path)
                self.result_path = new_result_path

                # Also update the YAML to point to the new output file
                if hasattr(self, "parsed_yaml") and self.parsed_yaml:
                    self.parsed_yaml["pipeline"]["output"]["path"] = new_result_path
                    # Rewrite the YAML file with the updated output path
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
        """
        Add a (directive, target_operator) pair to the memo list.

        Args:
            directive_name: Name of the directive that was applied
            target_operator: Name of the target operator the directive was applied to
        """
        self.memo.append((directive_name, target_operator))

    def get_optimization_path(self) -> str:
        """
        Get a formatted string showing the optimization path from root to this node.

        Returns:
            Formatted path string like "ROOT → chaining(extract_clause) → gleaning(extract_entity)"
        """
        if not self.memo:
            return "ROOT"

        path_parts = ["ROOT"]
        for directive, target_op in self.memo:
            path_parts.append(f"{directive}({target_op})")

        return " → ".join(path_parts)

    def get_exploration_tree_summary(
        self, root: Node, node_accuracies: Optional[Dict["Node", float]] = None
    ) -> str:
        """
        Generate a comprehensive but concise summary of the entire exploration tree.
        This gives the LLM agent complete context about what has been tried.

        Args:
            root: Root node of the tree
            node_accuracies: Optional dictionary mapping nodes to their accuracy values

        Returns:
            Formatted tree summary optimized for LLM consumption
        """

        # Collect all exploration paths and their outcomes
        successful_paths = []
        failed_paths = []

        def traverse_tree(node, current_path="ROOT"):
            # Add this node's path if it's not the root
            if node != root:
                if hasattr(node, "cost") and node.cost != -1:
                    # Include accuracy if available
                    if node_accuracies and node in node_accuracies:
                        accuracy = node_accuracies[node]
                        successful_paths.append(
                            f"{current_path} (cost: ${node.cost:.2f}, accuracy: {accuracy:.3f})"
                        )
                    else:
                        successful_paths.append(
                            f"{current_path} (cost: ${node.cost:.2f})"
                        )
                else:
                    failed_paths.append(f"{current_path} (failed)")

            # Traverse children
            for child in node.children:
                if child.memo:
                    # Get the most recent directive-operator pair for this child
                    latest_directive, latest_target = child.memo[-1]
                    child_path = f"{current_path} → {latest_directive}({latest_target})"
                else:
                    child_path = f"{current_path} → {child.latest_action.name if child.latest_action else 'unknown'}"
                traverse_tree(child, child_path)

        traverse_tree(root)

        # Group paths by directive patterns for better insights
        directive_patterns = {}
        for path in successful_paths + failed_paths:
            # Extract directive sequence
            directives = []
            parts = path.split(" → ")
            for part in parts[1:]:  # Skip ROOT
                if "(" in part:
                    directive = part.split("(")[0]
                    directives.append(directive)

            if directives:
                pattern_key = " → ".join(directives)
                if pattern_key not in directive_patterns:
                    directive_patterns[pattern_key] = {"successful": [], "failed": []}

                if "(failed)" in path:
                    directive_patterns[pattern_key]["failed"].append(path)
                else:
                    directive_patterns[pattern_key]["successful"].append(path)

        # Build summary
        summary_parts = []

        # Current position
        current_path = self.get_optimization_path()
        summary_parts.append(f"CURRENT POSITION: {current_path}")

        # Successful explorations
        if successful_paths:
            summary_parts.append(
                f"\nSUCCESSFUL EXPLORATIONS ({len(successful_paths)} total):"
            )

            # Show best performers first - sort by accuracy (highest first), then by cost (lowest first)
            def extract_sort_key(path):
                if "cost: $" not in path:
                    return (
                        0,
                        float("inf"),
                    )  # lowest accuracy, highest cost for failed cases
                try:
                    cost_part = path.split("cost: $")[1]
                    if ", accuracy:" in cost_part:
                        cost_str = cost_part.split(", accuracy:")[0]
                        accuracy_str = cost_part.split(", accuracy:")[1].split(")")[0]
                        return (
                            -float(accuracy_str),
                            float(cost_str),
                        )  # negative accuracy for descending order
                    else:
                        cost_str = cost_part.split(")")[0]
                        return (
                            0,
                            float(cost_str),
                        )  # no accuracy info, sort by cost only
                except (ValueError, IndexError):
                    return (0, float("inf"))

            sorted_successful = sorted(successful_paths, key=extract_sort_key)
            for i, path in enumerate(sorted_successful):
                summary_parts.append(f"  {i+1}. {path}")

        return "\n".join(summary_parts)

    def get_memo_for_llm(
        self, root_node: Node, node_accuracies: Optional[Dict["Node", float]] = None
    ) -> str:
        """
        Get a comprehensive exploration summary formatted for LLM prompts.

        Args:
            root_node: Root node of the tree
            node_accuracies: Optional dictionary mapping nodes to their accuracy values

        Returns:
            Complete exploration context to guide decision making
        """
        return self.get_exploration_tree_summary(root_node, node_accuracies)

    def delete(self, selected_node_final_id=None):
        """
        Delete this node and clean up its resources.
        For multi-instance candidates, moves files to backup_plans folder instead of deleting.

        Args:
            selected_node_final_id: The final ID of the selected node (for backup naming)

        This method:
        1. Removes the node from its parent's children list
        2. Moves multi-instance files to backup or deletes regular files
        3. Clears references to prevent memory leaks

        The global node ID counter is maintained (not decremented) to ensure
        unique IDs across the lifetime of the program.
        """
        # Remove from parent's children list
        if self.parent and self in self.parent.children:
            self.parent.children.remove(self)

        # Check if this is a multi-instance candidate using the explicit flag
        if self.is_multi_instance and selected_node_final_id is not None:
            # Move files to backup_plans folder with renamed IDs
            self._backup_multi_instance_files(selected_node_final_id)
        else:
            # Regular deletion for non-multi-instance nodes
            self._delete_files_permanently()

        # Clear references to help with garbage collection
        self.parent = None
        self.children = []
        self.parsed_yaml = {}
        self.message_history = []
        self.memo = []
        self.sample_result = []

    def _backup_multi_instance_files(self, selected_node_final_id):
        """
        Move multi-instance files to backup_plans folder with new naming scheme.

        Args:
            selected_node_final_id: The final ID of the selected node
        """
        try:
            # Extract the instantiation number from current ID (e.g., "7-2" -> "2")
            current_id_str = str(self.id)
            if "-" in current_id_str:
                instantiation_num = current_id_str.split("-")[1]
                new_backup_id = f"{selected_node_final_id}-{instantiation_num}"
            else:
                new_backup_id = f"{selected_node_final_id}-backup"

            # Create backup_plans directory
            if os.path.exists(self.yaml_file_path):
                yaml_dir = os.path.dirname(self.yaml_file_path)
                backup_dir = os.path.join(yaml_dir, "backup_plans")
                os.makedirs(backup_dir, exist_ok=True)

                # Move YAML file
                yaml_filename = os.path.basename(self.yaml_file_path)
                # Replace old ID with new backup ID in filename
                new_yaml_filename = yaml_filename.replace(
                    f"_{current_id_str}.yaml", f"_{new_backup_id}.yaml"
                )
                backup_yaml_path = os.path.join(backup_dir, new_yaml_filename)

                if os.path.exists(self.yaml_file_path):
                    os.rename(self.yaml_file_path, backup_yaml_path)
                    self.console.log(
                        f"[dim]Moved YAML to backup:[/dim] {self.yaml_file_path} → {backup_yaml_path}"
                    )

                # Move result JSON file
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
            # Fall back to regular deletion if backup fails
            self._delete_files_permanently()

    def _delete_files_permanently(self):
        """
        Permanently delete files (for non-multi-instance nodes).
        """
        try:
            if os.path.exists(self.yaml_file_path) and str(
                self.yaml_file_path
            ).endswith((".yaml", ".yml")):
                # Only delete if it looks like a generated file (contains numbers)
                if any(
                    char.isdigit()
                    for char in os.path.basename(str(self.yaml_file_path))
                ):
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
                # Only delete if it looks like a generated file (contains numbers)
                if any(char.isdigit() for char in os.path.basename(self.result_path)):
                    os.remove(self.result_path)
                    self.console.log(
                        f"[dim]Deleted generated result file:[/dim] {self.result_path}"
                    )
        except Exception as e:
            self.console.log(
                f"[yellow]Warning: Could not delete result file {self.result_path}: {e}[/yellow]"
            )
