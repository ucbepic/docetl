import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional

import litellm
import yaml

from docetl.console import DOCETL_CONSOLE
from docetl.reasoning_optimizer.directives import (
    DIRECTIVE_GROUPS,
    MULTI_INSTANCE_DIRECTIVES,
    Directive,
)
from docetl.reasoning_optimizer.directives.change_model_cost import (
    ChangeModelCostDirective,
    create_model_specific_directives,
)
from docetl.reasoning_optimizer.op_descriptions import *  # noqa: F403, F405
from docetl.utils import extract_output_from_json

from .Node import Node
from .ParetoFrontier import ParetoFrontier
from .search_utils import *  # noqa: F403, F405


class MOARSearch:
    """
    This class implements The search algorithm in the MOAR optimizer. It contains the following phases:
    1. Selection: Choose the best child using UCB with utility function
    2. Expansion: Add new children to the graph
    3. Simulation: Execute a pipeline on the sample dataset to get its empirical cost and accuracy
    4. Backpropagation: Update node values up the tree to inform the selection of the next iteration
    """

    def __init__(
        self,
        root_yaml_path: str,
        available_actions: set[Directive],
        sample_input,
        dataset_stats: str,
        dataset_name: str,
        available_models: List[str],
        evaluate_func: Callable,
        exploration_constant: float = 1.414,
        max_iterations: int = 20,
        model="gpt-5",
        output_dir: Optional[str] = None,
        build_first_layer: Optional[bool] = True,
        custom_metric_key: Optional[str] = None,
        sample_dataset_path: Optional[str] = None,
    ):
        """
        Initialize the MOARSearch algorithm.

        Args:
            root_yaml_path: Path to the initial YAML configuration file
            available_actions: List of available actions for expansion
            sample_input: sample input data
            dataset_stats: Statistics about the dataset
            dataset_name: Name of the dataset
            available_models: List of available model names
            exploration_constant: UCB exploration constant (default: sqrt(2))
            max_iterations: Maximum number of MOARSearch iterations
            model: Model to use for agent LLM calls
            output_dir: Directory to save new pipeline files (None means same dir as original)
            build_first_layer: Whether to build first layer nodes (default: True)
            evaluate_func: Evaluation function (results_file_path: str) -> dict
            custom_metric_key: Key to extract from evaluation results dict for accuracy metric
        """

        self.console = DOCETL_CONSOLE
        self.root = Node(root_yaml_path, c=exploration_constant, console=self.console)
        self.tree_lock = threading.RLock()
        self.max_concurrent_agents = 3

        # Start with base available actions
        self.available_actions = set(available_actions)
        self.available_models = available_models

        # Update change model directives with available_models list
        from docetl.reasoning_optimizer.directives.change_model import (
            ChangeModelDirective,
        )
        from docetl.reasoning_optimizer.directives.change_model_acc import (
            ChangeModelAccDirective,
        )
        from docetl.reasoning_optimizer.directives.change_model_cost import (
            ChangeModelCostDirective,
        )

        for action in self.available_actions:
            if isinstance(
                action,
                (
                    ChangeModelDirective,
                    ChangeModelAccDirective,
                    ChangeModelCostDirective,
                ),
            ):
                action.allowed_model_list = self.available_models

        # Initialize action tracking for all actions (base + model-specific)
        self.action_rewards = {action: 0.0 for action in self.available_actions}
        self.action_cost_changes = {action: 0.0 for action in self.available_actions}
        self.action_accuracy_changes = {
            action: 0.0 for action in self.available_actions
        }
        self.action_counts = {action: 0.0 for action in self.available_actions}
        self.exploration_constant = exploration_constant
        self.max_iterations = max_iterations
        self.iteration_count = 0
        self.start_time = None
        self.model = model
        self.sample_input = sample_input
        self.dataset_stats = dataset_stats
        self.dataset_name = dataset_name
        self.output_dir = output_dir

        # Set up evaluation function and dataset metrics
        self.evaluate_func = evaluate_func

        # Use predefined metric key for known datasets if custom_metric_key not provided
        if custom_metric_key is not None:
            self.primary_metric_key = custom_metric_key
        else:
            self.dataset_metrics = {
                "cuad": "avg_f1",
                "blackvault": "avg_distinct_locations",
                "game_reviews": "weighted_score",
                "medec": "combined_score",
                "sustainability": "economic_activity_accuracy",
                "biodex": "avg_rp_at_5",
            }
            self.primary_metric_key = self.dataset_metrics.get(dataset_name)
        # Set up log file path
        if self.output_dir:
            self.log_path = os.path.join(self.output_dir, "moar_tree_log.txt")
        else:
            self.log_path = "moar_tree_log.txt"

        # Initialize log file (clear it)
        with open(self.log_path, "w", encoding="utf-8") as f:
            f.write("Search Graph Visits and Values Log\n")
            f.write(f"Root YAML: {root_yaml_path}\n")
            f.write(f"Max iterations: {max_iterations}\n")
            f.write(f"{'='*50}\n")

        # Initialize Pareto frontier
        self.pareto_frontier = ParetoFrontier(
            self.action_rewards,
            self.action_cost_changes,
            self.action_accuracy_changes,
            dataset_name,
            self.evaluate_func,
            console=self.console,
        )

        # Create comprehensive directive mapping
        self.directive_name_to_obj = {
            action.name: action for action in self.available_actions
        }

        # Track iterations without new Pareto optimal plans for early stopping
        self.iterations_without_improvement = 0

        # Track total cost for all completion calls during search
        self.total_search_cost = 0.0

        self.model_stats = {}
        self.frontier_models = []
        self.sample_dataset_path = (
            sample_dataset_path  # Path to sample dataset for optimization
        )
        self.console.log(
            "[bold blue]Building first layer nodes of the search graph[/bold blue]"
        )
        # Initialization: build the first layer nodes of the search graph
        for model in self.available_models:
            new_yaml_file = deepcopy(self.root.parsed_yaml)
            new_yaml_file["default_model"] = model
            # Update dataset path to use sample dataset if provided
            if self.sample_dataset_path and "datasets" in new_yaml_file:
                datasets = new_yaml_file["datasets"]
                if isinstance(datasets, dict) and datasets:
                    # Update the first dataset's path
                    first_dataset_key = next(iter(datasets.keys()))
                    if isinstance(datasets[first_dataset_key], dict):
                        datasets[first_dataset_key]["path"] = self.sample_dataset_path
            fix_models(new_yaml_file)
            if self.output_dir:
                original_filename = os.path.basename(
                    str(self.root.yaml_file_path)
                ).removesuffix(".yaml")
                base_path = os.path.join(self.output_dir, original_filename)
                os.makedirs(self.output_dir, exist_ok=True)
            else:
                # Use same directory as original pipeline
                base_path = str(self.root.yaml_file_path).removesuffix(".yaml")

            # Sanitize model name for file paths (replace / with _)
            sanitized_model = model.replace("/", "_")
            new_yaml_path = f"{base_path}_{sanitized_model}.yaml"
            new_yaml_file["pipeline"]["output"][
                "path"
            ] = f"{base_path}_{sanitized_model}.json"

            with open(new_yaml_path, "w") as file:
                yaml.dump(
                    new_yaml_file,
                    file,
                    default_flow_style=False,
                    allow_unicode=True,
                    sort_keys=False,
                )

            new_node = Node(
                yaml_file_path=new_yaml_path, parent=self.root, console=self.console
            )
            cost, accuracy = self.simulate(new_node)
            if cost == -1:
                new_node.delete()
                continue
            self.root.add_child(new_node)
            self.model_stats[model] = {
                "cost": cost,
                "accuracy": accuracy,
            }
            self.pareto_frontier.add_plan_f1(new_node, accuracy)

        # Remove non-frontier children
        children_to_remove = []
        for child in self.root.children:
            if child not in self.pareto_frontier.frontier_plans:
                children_to_remove.append(child)

        for child in children_to_remove:
            self.pareto_frontier.delete_plan(child)
            child.delete()

        # Process remaining frontier children (only those on the frontier)
        for child in self.root.children:
            if child not in self.pareto_frontier.frontier_plans:
                continue
            child_model = child.parsed_yaml["default_model"]
            self.frontier_models.append(child_model)
            child.value = 0
            child.visits = 1

            model_specific_directive = create_model_specific_directives(
                child_model, self.available_models
            )
            self.available_actions.add(model_specific_directive)
            self.directive_name_to_obj[model_specific_directive.name] = (
                model_specific_directive
            )

            action_name = "change to " + child_model
            action = self.directive_name_to_obj.get(action_name)
            child.latest_action = action
            child.sample_result = extract_output_from_json(child.yaml_file_path)
            # Add memo entry for first layer nodes
            if child.op_dict:
                first_op_name = list(child.op_dict.keys())[0]
                child.add_memo_entry(action_name, first_op_name)
            # Mark all change model directives as used for first layer nodes
            # since they are already model-specific plans
            for op_name in child.op_dict.keys():
                for directive in self.available_actions:
                    if isinstance(directive, ChangeModelCostDirective):
                        child.mark_action_used(op_name, directive)

        self.root.visits = len(self.root.children)

        self.log_tree_to_file(0)

        return

    def evaluate_node(self, node: Node) -> float:
        """
        Evaluate a node's accuracy by running the evaluation function on its output.

        Args:
            node: Node object representing the plan

        Returns:
            The accuracy score for the node
        """
        if node.cost == -1:  # Handle error case
            return float("-inf")

        result_file_path = node.parsed_yaml["pipeline"]["output"]["path"]

        try:
            results = self.evaluate_func(result_file_path)

            # Extract the appropriate metric based on dataset or custom metric key
            if hasattr(self, "primary_metric_key") and self.primary_metric_key:
                # Use custom metric key or predefined metric key
                if self.primary_metric_key in results:
                    true_accuracy = results[self.primary_metric_key]
                else:
                    # Fallback to first numerical value found if metric missing
                    true_accuracy = next(
                        (v for v in results.values() if isinstance(v, (int, float))),
                        0.5,
                    )
            else:
                # Fallback to first numerical value found if dataset unknown or metric missing
                true_accuracy = next(
                    (v for v in results.values() if isinstance(v, (int, float))), 0.5
                )

        except Exception as e:
            self.console.log(
                f"[yellow]⚠️ Evaluation failed for node {node.get_id()}: {e}[/yellow]"
            )
            node._log_inf_occurrence("evaluation_failure", str(e), node.yaml_file_path)
            # Return -inf when evaluation fails, same as unexecutable plans
            return float("-inf")

        # Guard against NaN values
        if true_accuracy is None or (
            isinstance(true_accuracy, float) and (true_accuracy != true_accuracy)
        ):  # NaN check
            self.console.log(
                f"[yellow]⚠️ Evaluation returned NaN for node {node.get_id()}, setting to -inf[/yellow]"
            )
            # Log -inf occurrence for debugging
            node._log_inf_occurrence(
                "nan_evaluation",
                f"Evaluation returned NaN or None: {true_accuracy}",
                node.yaml_file_path,
            )
            return float("-inf")

        return true_accuracy

    def search(self):
        """
        Perform MCTS search to find the optimal query plan with concurrent agents.

        Returns:
            Tuple of (best_node, search_statistics)
        """
        self.start_time = time.time()
        self.iteration_count = 0

        self.console.log(
            f"[bold blue]Starting concurrent MCTS search with {self.max_iterations} iterations and {self.max_concurrent_agents} agents...[/bold blue]"
        )

        with ThreadPoolExecutor(max_workers=self.max_concurrent_agents) as executor:
            while self.should_continue():
                # Submit up to max_concurrent_agents tasks
                futures = []
                with self.tree_lock:
                    remaining_iterations = self.max_iterations - self.iteration_count
                    if remaining_iterations <= 0:
                        break
                    agents_to_submit = min(
                        self.max_concurrent_agents,
                        remaining_iterations,
                    )

                for _ in range(agents_to_submit):
                    future = executor.submit(self.search_iteration)
                    futures.append(future)

                # Wait for at least one agent to complete
                for future in as_completed(futures):
                    try:
                        success = future.result()
                        if success:
                            with self.tree_lock:
                                # Double-check we haven't exceeded max_iterations
                                if self.iteration_count < self.max_iterations:
                                    self.iteration_count += 1
                                else:
                                    # Already reached max, don't increment further
                                    break
                    except Exception as e:
                        self.console.log(
                            f"[bold red]Agent failed with error: {e}[/bold red]"
                        )

                    # Check if we should continue after each completion
                    if not self.should_continue():
                        break

                # Cancel remaining futures if stopping
                if not self.should_continue():
                    for future in futures:
                        future.cancel()

        # Return all frontier plans
        frontier_plans = [
            summary["node"]
            for summary in self.pareto_frontier.get_all_plans_summary()
            if summary["is_frontier"]
        ]

        return frontier_plans

    def search_iteration(self):
        """Perform one complete MCTS iteration with thread safety."""
        thread_id = threading.current_thread().ident

        # 1. Selection: Find the best leaf node (requires tree lock)
        dual_expand = False
        with self.tree_lock:
            self.console.log(f"[dim][Thread {thread_id}][/dim] [cyan]SELECTION[/cyan]")
            leaf = self.select(self.root)
            self.console.log(
                f"[dim][Thread {thread_id}][/dim] [cyan]SELECTED NODE:[/cyan] {leaf.get_id()}"
            )
            if self.is_first_layer_node(leaf) and leaf.visits == 1:
                dual_expand = True
                self.increment_visits_up_tree(leaf)
            self.increment_visits_up_tree(leaf)

        # 2. Expansion: Always attempt to expand the leaf, catch errors (requires tree lock)
        with self.tree_lock:
            self.console.log(
                f"[dim][Thread {thread_id}][/dim] [magenta]EXPANSION[/magenta]"
            )
            acc_children = []
            cost_children = []
            if dual_expand:
                self.console.log(
                    f"[dim][Thread {thread_id}][/dim] [magenta]DUAL EXPANDING node {leaf.get_id()}[/magenta]"
                )
                try:
                    acc_children = self.expand(leaf, "acc")
                    has_leaf_acc = 1
                except RuntimeError as e:
                    self.console.log(f"[yellow]{e}[/yellow]")
                    has_leaf_acc = 0
                try:
                    cost_children = self.expand(leaf, "cost")
                    has_leaf_cost = 1
                except RuntimeError as e:
                    self.console.log(f"[yellow]{e}[/yellow]")
                    has_leaf_cost = 0
            else:
                optimize_goal = self.get_optimize_goal(leaf)
                if optimize_goal == "acc":
                    acc_children = self.expand(leaf, optimize_goal="acc")
                    has_leaf_acc = 1
                    has_leaf_cost = 0
                else:
                    cost_children = self.expand(leaf, optimize_goal="cost")
                    has_leaf_cost = 1
                    has_leaf_acc = 0

        # 3. Simulation: Run simulations from the leaf
        is_frontier_updated = False
        if has_leaf_acc:
            is_frontier_updated = self._simulate_children(acc_children, "acc")

        if has_leaf_cost:
            cost_frontier_updated = self._simulate_children(cost_children, "cost")
            is_frontier_updated = is_frontier_updated or cost_frontier_updated

        # Update counter for early stopping (requires tree lock)
        with self.tree_lock:
            if is_frontier_updated:
                self.iterations_without_improvement = 0
            else:
                self.iterations_without_improvement += 1

            self.log_tree_to_file(self.iteration_count + 1)

        success = has_leaf_acc or has_leaf_cost
        self.console.log(
            f"[dim][Thread {thread_id}][/dim] [green]MCTS iteration completed:[/green] success={success}, has_leaf_acc={has_leaf_acc}, has_leaf_cost={has_leaf_cost}"
        )
        return success

    def _simulate_children(self, children: List[Node], goal_type: str) -> bool:
        """
        Simulate a list of children and handle multi-instance vs single instance logic.

        Args:
            children: List of child nodes to simulate
            goal_type: Type of goal ("acc" or "cost") for logging purposes

        Returns:
            True if frontier was updated, False otherwise
        """
        is_frontier_updated = False

        if len(children) > 1:
            self.console.log(
                f"[bold blue]Handling {len(children)} multi-instance candidates[/bold blue]"
            )
            candidate_results = []

            # Simulate all candidates
            for num, candidate in enumerate(children):
                cost, accuracy = self.simulate(candidate)
                with self.tree_lock:
                    cost_to_add = max(cost, 0)
                    self.console.log(
                        f"[green]💰 Adding multi-instance {goal_type} candidate cost:[/green] ${cost_to_add:.4f} (total before: ${self.total_search_cost:.4f})"
                    )
                    self.total_search_cost += cost_to_add
                self.console.log(
                    f"[dim]Multi-instance candidate {num} - Cost:[/dim] ${cost:.2f}, [dim]Accuracy:[/dim] {accuracy:.4f}"
                )
                if cost != -1 and accuracy != float("-inf"):
                    candidate_results.append((candidate, accuracy, cost))
                else:
                    self.console.log(
                        f"[yellow]Multi-instance candidate {num} failed during simulation[/yellow]"
                    )

            if candidate_results:
                # Select the best candidate based on accuracy
                best_candidate, best_accuracy, best_cost = max(
                    candidate_results, key=lambda x: x[1]
                )
                self.console.log(
                    f"[bold green]Selected best multi-instance candidate {best_candidate.get_id()} with accuracy {best_accuracy:.4f} and cost ${best_cost:.2f}[/bold green]"
                )

                # Change the best candidate's ID back to a proper counter ID
                old_id = best_candidate.get_id()
                new_id = best_candidate.set_id_to_counter()
                self.console.log(
                    f"[dim]Updated best candidate ID from {old_id} to {new_id}[/dim]"
                )

                # Delete ALL non-selected candidates (both failed and successful ones that weren't chosen)
                for candidate in children:
                    if candidate != best_candidate:
                        candidate.delete(selected_node_final_id=new_id)

                # Process only the best candidate (requires tree lock for backprop)
                affected_nodes, is_frontier_updated = self.add_to_frontier(
                    best_candidate, best_accuracy
                )
                with self.tree_lock:
                    self.backpropagate(affected_nodes, best_candidate)
            else:
                self.console.log(
                    "[yellow]No successful candidates found, deleting all multi-instance candidates[/yellow]"
                )
                for candidate in children:
                    candidate.delete(selected_node_final_id=None)

        else:
            # Single instantiations
            for child in children:
                cost, accuracy = self.simulate(child)
                with self.tree_lock:
                    cost_to_add = max(cost, 0)
                    self.console.log(
                        f"[green]💰 Adding leaf {goal_type} simulation:[/green] ${cost_to_add:.4f} (total before: ${self.total_search_cost:.4f})"
                    )
                    self.total_search_cost += cost_to_add
                affected_nodes, temp_updated = self.add_to_frontier(child, accuracy)
                if temp_updated:
                    is_frontier_updated = True
                # Check if any node was added to the frontier (value = 1)
                with self.tree_lock:
                    self.backpropagate(affected_nodes, child)

        return is_frontier_updated

    def select(self, node: Node) -> Node:
        """
        Select the best child using the Node's best_child method.

        Args:
            node: Starting node for selection

        Returns:
            Selected leaf node
        """
        current = node

        while self.is_fully_explored(current):
            self.console.log(
                f"[dim]Node {current.get_id()} is fully explored, selecting best child[/dim]"
            )
            current = current.best_child()

        return current

    def is_fully_explored(self, node: Node) -> bool:
        """Check if a node has been fully explored based on visit count."""
        return is_fully_explored(node)

    def expansion_prompt_acc(
        self, node, action_options, input_query
    ) -> tuple[str, str]:
        return create_expansion_prompt_acc(
            node,
            action_options,
            input_query,
            self.available_actions,
            self.action_cost_changes,
            self.action_accuracy_changes,
            self.action_counts,
            self.sample_input,
            self.root,
            node.yaml_file_path,
            self.dataset_name,
            self.pareto_frontier.plans_accuracy,
            self.model_stats,
            self.available_models,
        )

    def expansion_prompt_cost(
        self, node, action_options, input_query
    ) -> tuple[str, str]:
        return create_expansion_prompt_cost(
            node,
            action_options,
            input_query,
            self.available_actions,
            self.action_cost_changes,
            self.action_accuracy_changes,
            self.action_counts,
            self.sample_input,
            self.root,
            node.yaml_file_path,
            self.dataset_name,
            self.pareto_frontier.plans_accuracy,
            self.model_stats,
            self.available_models,
        )

    def is_first_layer_node(self, node: Node) -> bool:
        """
        Check if a node is a first layer node (direct child of root with model change action).

        Args:
            node: Node to check

        Returns:
            True if this is a first layer node, False otherwise
        """
        # First layer nodes are direct children of root
        if node.parent != self.root:
            return False

        return True

    def get_optimize_goal(self, node: Node) -> str:
        """
        Determine the optimization goal based on the node's accuracy relative to other plans.

        Args:
            node: Node to determine optimization goal for

        Returns:
            "cost" for nodes in the top 50% accuracy plans, "acc" otherwise
        """
        if not self.pareto_frontier.plans_accuracy:
            # If no plans exist yet, optimize for accuracy
            return "acc"

        # Get all accuracy values from tracked plans
        all_accuracies = list(self.pareto_frontier.plans_accuracy.values())

        # Filter out invalid accuracies (negative infinity indicates failed plans)
        valid_accuracies = [acc for acc in all_accuracies if acc != float("-inf")]

        if not valid_accuracies:
            # If no valid accuracies exist, optimize for accuracy
            return "acc"

        # Calculate the 50th percentile threshold
        valid_accuracies.sort()
        n = len(valid_accuracies)
        percentile_50_index = n // 2  # This gives us the median (50th percentile)
        threshold = valid_accuracies[percentile_50_index]

        # Get this node's accuracy (if it exists in the frontier)
        node_accuracy = self.pareto_frontier.plans_accuracy.get(node, float("-inf"))

        # If node accuracy is above the 50th percentile, optimize for cost
        # Otherwise, optimize for accuracy
        if node_accuracy > threshold:
            return "cost"
        else:
            return "acc"

    def expand(self, node: Node, optimize_goal: str) -> List[Node]:
        """
        Expand a node with a specific optimization goal.

        Args:
            node: Node to expand
            optimize_goal: The optimization goal to use

        Returns:
            List of newly created child nodes
        """
        max_retries = 3
        retry_count = 0

        # Build action options and initial prompt once
        op_list = list(node.op_dict.keys())
        banned_directives = set()
        last_step = None
        last_op = None
        if len(node.memo) > 0:
            last_step = node.memo[-1]
            if node.value < 0:
                last_directive = last_step[0]
                last_op = last_step[1]
                for group in DIRECTIVE_GROUPS:
                    directive = self.directive_name_to_obj.get(last_directive)
                    if directive in DIRECTIVE_GROUPS[group]:
                        banned_directives = set(DIRECTIVE_GROUPS[group])

        if optimize_goal == "acc":
            action_options = []  # a list of tuple
            for op_name in op_list:
                if op_name in node.used_actions:
                    used_actions = node.used_actions[op_name]
                else:
                    used_actions = set()

                # Get compression directives to exclude for code_map and extract operations
                compression_exclusions = get_excluded_directives_for_operation(
                    node, op_name
                )

                if last_op is None or last_op != op_name:
                    banned_directives = set()

                # Filter actions to only include cheaper models for cost optimization
                op_config = node.op_dict[op_name]
                current_model = op_config.get("model", "gpt-5")
                dynamic_actions = []

                for directive in self.available_actions:
                    if isinstance(directive, ChangeModelCostDirective):
                        continue
                    else:
                        # Keep other directives as-is
                        dynamic_actions.append(directive)

                action_space = (
                    set(dynamic_actions)
                    - banned_directives
                    - used_actions
                    - compression_exclusions
                )  # The actions that are not used on this operator and not excluded by group
                for action in action_space:
                    action_options.append((op_name, action.name))
            if len(action_options) < 1:
                raise RuntimeError(
                    "No applicable action found for expansion. Action space may be exhausted or all actions are inapplicable."
                )
            self.console.log("[bold cyan]OPTIMIZING ACC:[/bold cyan]")
            user_message, condensed_user_message = self.expansion_prompt_acc(
                node, action_options=action_options, input_query=node.parsed_yaml
            )

        elif optimize_goal == "cost":
            action_options = []  # a list of tuple
            for op_name in op_list:
                if op_name in node.used_actions:
                    used_actions = node.used_actions[op_name]
                else:
                    used_actions = set()

                # Get compression directives to exclude for code_map and extract operations
                compression_exclusions = get_excluded_directives_for_operation(
                    node, op_name
                )

                if last_op is None or last_op != op_name:
                    banned_directives = set()

                # Filter cost directives to only include cheaper models
                op_config = node.op_dict[op_name]
                default_model = node.parsed_yaml.get("default_model", "gpt-5")
                current_model = op_config.get("model", default_model)
                dynamic_cost_directives = []
                for directive in self.available_actions:
                    if (
                        isinstance(directive, ChangeModelCostDirective)
                        and directive.target_model
                    ):
                        if (
                            self._is_cheaper_model(
                                directive.target_model, current_model
                            )
                            and directive.target_model in self.frontier_models
                        ):
                            dynamic_cost_directives.append(directive)
                    else:
                        dynamic_cost_directives.append(directive)
                action_space = (
                    set(dynamic_cost_directives)
                    - banned_directives
                    - used_actions
                    - compression_exclusions
                )  # The cost actions that are not used on this operator and not excluded by group
                for action in action_space:
                    action_options.append((op_name, action.name))
            if len(action_options) < 1:
                raise RuntimeError(
                    "No applicable action found for expansion. Action space may be exhausted or all actions are inapplicable."
                )
            user_message, condensed_user_message = self.expansion_prompt_cost(
                node, action_options=action_options, input_query=node.parsed_yaml
            )

        # Initialize messages with accumulated message history from the path to this node
        messages = node.message_history.copy()
        message_condensed = node.message_history.copy()

        # Add the current system message and user message for this expansion
        if not messages or messages[0]["role"] != "system":
            messages.insert(
                0,
                {
                    "role": "system",
                    "content": "You are an expert query optimization agent for document processing pipelines. Your role is to analyze user queries and apply rewrite directives to create more accurate and cost effective execution plans. Your output must follow the structured output format.",
                },
            )

        messages.append({"role": "user", "content": user_message})
        if len(message_condensed) == 0 or message_condensed[0]["role"] != "system":
            message_condensed.append(
                {
                    "role": "system",
                    "content": "You are an expert query optimization agent for document processing pipelines. Your role is to analyze user queries and apply rewrite directives to create more accurate and cost effective execution plans. Your output must follow the structured output format.",
                }
            )
        message_condensed.append({"role": "user", "content": condensed_user_message})

        # Trim the history to prevent context window overflow before sending to the model
        messages = trim_history(messages)
        message_condensed = trim_history(message_condensed)

        while retry_count < max_retries:

            response = litellm.completion(
                model=self.model,
                messages=messages,
                api_key=os.environ.get("AZURE_API_KEY"),
                api_base=os.environ.get("AZURE_API_BASE"),
                api_version=os.environ.get("AZURE_API_VERSION"),
                azure=True,
                response_format=ExpandResponseFormat,
            )
            call_cost = response._hidden_params["response_cost"]
            with self.tree_lock:
                self.console.log(
                    f"[green]💰 Adding LLM call cost:[/green] ${call_cost:.4f} (total before: ${self.total_search_cost:.4f})"
                )
                self.total_search_cost += call_cost
            reply = response.choices[0].message.content

            try:
                parsed = json.loads(reply)
                directive_name = parsed.get("directive")
                target_op_list = parsed.get("operators")
                self.console.log(
                    f"[cyan]Directive:[/cyan] {directive_name}, [cyan]Target ops:[/cyan] {target_op_list}"
                )
                messages.append({"role": "assistant", "content": reply})
                message_condensed.append({"role": "assistant", "content": reply})
            except Exception as e:
                self.console.log(
                    f"[yellow]Failed to parse agent response: {e}[/yellow]"
                )
                retry_count += 1
                continue

            # Check if directive is already used for this plan + target ops
            directive = self.directive_name_to_obj.get(directive_name)
            if directive is None:
                self.console.log(
                    f"[yellow]Unknown directive name: {directive_name}[/yellow]"
                )
                retry_count += 1
                continue

            # Check if already used
            already_used = False
            for target_op in target_op_list:
                if directive in node.used_actions[target_op]:
                    already_used = True
                    break

            if already_used:
                retry_count += 1
                self.console.log(
                    f"[yellow]Directive '{directive_name}' already used for these ops. Retry {retry_count}/{max_retries}[/yellow]"
                )
                # Append feedback message as new user message
                feedback_message = f"We have already tried the directive '{directive_name}' on these operators: {target_op_list}. Please pick another directive from the available options we listed in the previous message."
                messages.append({"role": "user", "content": feedback_message})
                message_condensed.append({"role": "user", "content": feedback_message})
                continue

            # Valid directive found - break out of directive selection loop
            break

        # If we've exhausted directive selection retries
        if retry_count >= max_retries:
            raise RuntimeError(
                f"Failed to find unused directive after {max_retries} retries"
            )

        orig_default_model = node.parsed_yaml.get("default_model")

        datasets = node.parsed_yaml.get("datasets", {})
        input_file_path = None
        if isinstance(datasets, dict) and datasets:
            first_dataset = next(iter(datasets.values()))
            if isinstance(first_dataset, dict):
                input_file_path = first_dataset.get("path")

        # Mark action as used and increment use count immediately
        for target_op in target_op_list:
            node.mark_action_used(target_op, directive)

        # Increment action count immediately to prevent race conditions
        self.action_counts[directive] += 1

        message_length = len(messages)

        # Check if this directive supports multiple instantiations
        is_multi_instance = directive in MULTI_INSTANCE_DIRECTIVES
        num_instantiations = 2 if is_multi_instance else 1

        self.console.log(
            f"[bold blue]Creating {num_instantiations} instantiation(s) for directive '{directive_name}'[/bold blue]"
        )

        children = []
        instantiation_messages = messages.copy()
        for i in range(num_instantiations):
            try:
                self.console.log(
                    f"[dim]Creating instantiation {i+1}/{num_instantiations}[/dim]"
                )

                # For multi-instance directives, add variation to each instantiation
                if is_multi_instance:
                    # Use accumulated message history (includes previous instantiations)
                    # Add instantiation-specific context to encourage variation
                    if i == 0:
                        variation_prompt = f"This is instantiation {i+1} of {num_instantiations} for the '{directive_name}' directive. Focus on creating a distinct approach by exploring different parameter combinations or implementation strategies. For example, you can try different models, different parameter settings (chunk size, top k, etc.), or different implementation strategies."
                    else:
                        variation_prompt = f"This is instantiation {i+1} of {num_instantiations} for the '{directive_name}' directive. Based on the previous {i} instantiation(s) shown above, create a DIFFERENT approach that explores alternative parameters, strategies, or implementations. AVOID repeating the exact same configs asthe previous instantiations."

                    # Insert variation context before the last user message
                    variation_msg = {"role": "system", "content": variation_prompt}
                    instantiation_messages.append(variation_msg)

                new_ops_list, updated_message_history, cost = directive.instantiate(
                    operators=node.parsed_yaml["operations"],
                    target_ops=target_op_list,
                    agent_llm=self.model,
                    optimize_goal=optimize_goal,
                    global_default_model=orig_default_model,
                    message_history=instantiation_messages,
                    input_file_path=input_file_path,
                    pipeline_code=node.parsed_yaml,
                    dataset=self.dataset_name,
                    model_stats=self.model_stats,
                    allowed_model_list=self.available_models,
                )
                with self.tree_lock:
                    self.console.log(
                        f"[green]💰 Adding agent instantiation cost:[/green] ${cost:.4f} (total before: ${self.total_search_cost:.4f})"
                    )
                    self.total_search_cost += cost
                if new_ops_list is None:
                    self.console.log(
                        f"[yellow]Instantiation {i+1} failed: no ops list returned[/yellow]"
                    )
                    continue

                instantiation_messages = updated_message_history

                # Create child node for this instantiation
                # For multi-instance directives, use parent_id-instantiation_num format
                custom_id = (
                    f"{node.get_id()}-{i+1}-{optimize_goal}"
                    if is_multi_instance
                    else None
                )
                child = self.instantiate_node(
                    node,
                    new_ops_list,
                    directive_name,
                    target_op_list,
                    optimize_goal,
                    message_condensed + updated_message_history[message_length:],
                    custom_id,
                    is_multi_instance,
                )

                children.append(child)
                self.console.log(
                    f"[green]Instantiation {i+1} created successfully[/green]"
                )

            except Exception as e:
                self.console.log(
                    f"[yellow]Instantiation {i+1} failed with error: {str(e)}[/yellow]"
                )
                continue

        if not children:
            raise RuntimeError(
                f"All {num_instantiations} instantiation(s) failed for directive '{directive_name}'"
            )

        return children

    def simulate(self, node: Node):
        """
        Simulate a node (plan). Execute the plan and evaluate it separately.

        Args:
            node: Node to start simulation from

        Returns:
            The accuracy and cost of the node
        """

        accuracy = float("-inf")
        cost = -1
        self.console.log(
            f"[dim]Simulating node {node.get_id()}, {node.yaml_file_path}[/dim]"
        )

        try:
            # Step 1: Execute the plan (this will set node.cost)
            node.execute_plan()
        except Exception as e:
            self.console.log(
                f"[yellow]Failed to execute plan for node {node.get_id()}: {str(e)}[/yellow]"
            )
            # Log -inf occurrence for debugging
            node._log_inf_occurrence(
                "simulation_execution_failure", str(e), node.yaml_file_path
            )
            # Set cost to -1 to indicate failure (this is already done in Node.execute_plan)
            # Do not add failed plans to the frontier
            return cost, accuracy

        # Step 2: Evaluate the plan (this will call the evaluation function)
        try:
            accuracy = self.evaluate_node(node)
            cost = node.cost
            self.console.log(
                f"[green]Node {node.get_id()} evaluation -[/green] [dim]cost:[/dim] ${cost:.2f}, [dim]accuracy:[/dim] {accuracy:.4f}"
            )
        except Exception as e:
            self.console.log(
                f"[yellow]Failed to evaluate plan for node {node.get_id()}: {str(e)}[/yellow]"
            )
            # Log -inf occurrence for debugging
            node._log_inf_occurrence(
                "simulation_evaluation_failure", str(e), node.yaml_file_path
            )
            return cost, accuracy

        return cost, accuracy

    def add_to_frontier(self, node: Node, accuracy: float):
        # Decrement action count for failed plans
        if node.cost == -1 and node.latest_action in self.action_counts:
            self.action_counts[node.latest_action] -= 1

        # Step 3: Add to frontier
        affected_nodes, is_frontier_updated = self.pareto_frontier.add_plan_f1(
            node, accuracy
        )
        self.action_rewards = self.pareto_frontier.action_rewards
        self.action_cost_changes = self.pareto_frontier.action_cost_changes
        self.action_accuracy_changes = self.pareto_frontier.action_accuracy_changes
        return affected_nodes, is_frontier_updated

    def increment_visits_up_tree(self, node: Node):
        """
        Increment visit count up the tree from the given node to root.
        Called immediately after expansion.
        """
        current = node
        while current is not None:
            current.update_visit()
            current = current.parent

    def backpropagate(self, affected_nodes: Dict[Node, int], visit_node):
        """
        Backpropagate only the simulation value changes up the tree.
        Visit counts are updated separately in increment_visits_up_tree.
        """
        for node, val in affected_nodes.items():
            current = node
            while current is not None:
                current.update_value(val)
                current = current.parent

        visit_node.update_visit()

    def should_continue(self) -> bool:
        """Check if MCTS should continue running."""
        if self.iteration_count >= self.max_iterations:
            return False

        # Early stopping: return False if last 10 iterations found no Pareto optimal plans
        if self.iterations_without_improvement >= 10:
            self.console.log(
                f"[yellow]Early stopping: No Pareto optimal plans found in last {self.iterations_without_improvement} iterations[/yellow]"
            )
            return False

        return True

    def get_frontier_summary(self) -> List[Dict[str, Any]]:
        """Get summary of all plans in the Pareto frontier."""
        return self.pareto_frontier.get_all_plans_summary()

    def _is_cheaper_model(self, target_model: str, current_model: str) -> bool:
        """Check if target_model is cheaper than current_model for this dataset."""

        current_cost = self.model_stats.get(current_model)["cost"]
        target_cost = self.model_stats.get(target_model)["cost"]

        if current_cost is None or target_cost is None:
            return False

        return target_cost < current_cost

    def print_tree_visits_and_values(self, node=None, depth=0, file_handle=None):
        """
        Recursively print every node's visits and value in the MCTS tree.
        Args:
            node: The node to start from (default: root)
            depth: Current depth for indentation
            file_handle: If provided, write to this file instead of printing
        """
        if node is None:
            node = self.root
        indent = "  " * depth

        # Include action information if available
        action_info = ""
        if hasattr(node, "latest_action") and node.latest_action is not None:
            action_info = f", Action: {node.latest_action.name}"
        elif node == self.root:
            action_info = ", Action: ROOT"

        output = f"{indent}Node ID: {node.get_id()}, Visits: {node.visits}, Value: {node.value}{action_info}"

        if file_handle:
            file_handle.write(output + "\n")
        else:
            self.console.log(output)

        for child in node.children:
            self.print_tree_visits_and_values(child, depth + 1, file_handle)

    def log_tree_to_file(self, iteration_num):
        """
        Append the current tree state to a log file with iteration information.
        Args:
            iteration_num: Current iteration number
        """
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(f"\n{'='*50}\n")
            f.write(f"ITERATION {iteration_num}\n")
            f.write(f"{'='*50}\n")

            # Add action performance statistics
            f.write("Action Performance Statistics:\n")
            for action in self.available_actions:
                cost_change = self.action_cost_changes.get(action, 0)
                accuracy_change = self.action_accuracy_changes.get(action, 0)
                count = self.action_counts.get(action, 0)

                if count > 0:
                    avg_cost_change = cost_change / count
                    avg_accuracy_change = accuracy_change / count
                    f.write(
                        f"- {action.name}: {count} uses, avg change in cost: {avg_cost_change:+.2f}, avg change in accuracy: {avg_accuracy_change:+.4f}\n"
                    )
                else:
                    f.write(
                        f"- {action.name}: {count} uses, avg change in cost: Unknown (never tried), avg change in accuracy: Unknown (never tried)\n"
                    )
            f.write("\n")

            # Add tree structure
            f.write("Tree Structure:\n")
            self.print_tree_visits_and_values(file_handle=f)
            f.write("\n")
            f.write(f"Total search cost: ${self.total_search_cost:.2f}\n")

    def instantiate_node(
        self,
        node,
        new_ops_list,
        directive_name,
        target_op_list,
        optimize_goal,
        message_condensed,
        custom_id=None,
        is_multi_instance=False,
    ):
        """
        Instantiate a new child node by applying the directive to the given node and target operations.
        Args:
            node: The parent node
            new_ops_list: The entire pipeline operations list (not a subset)
            directive_name: The name of the directive
            target_op_list: List of target operations
            optimize_goal: The optimization goal (e.g., 'acc' or 'cost')
        Returns:
            The newly created child node
        """

        new_parsed_yaml = deepcopy(node.parsed_yaml)
        new_parsed_yaml["operations"] = new_ops_list
        new_parsed_yaml["bypass_cache"] = True
        new_parsed_yaml = update_pipeline(new_parsed_yaml, new_ops_list, target_op_list)

        # Update dataset path to use sample dataset if provided
        if self.sample_dataset_path and "datasets" in new_parsed_yaml:
            datasets = new_parsed_yaml["datasets"]
            if isinstance(datasets, dict) and datasets:
                # Update the first dataset's path
                first_dataset_key = next(iter(datasets.keys()))
                if isinstance(datasets[first_dataset_key], dict):
                    datasets[first_dataset_key]["path"] = self.sample_dataset_path

        fix_models(new_parsed_yaml)

        # Determine the node ID to use for filename
        if custom_id is not None:
            node_id_for_file = custom_id
        else:
            node_id_for_file = Node.get_next_id()

        # Determine where to save the new pipeline file
        if self.output_dir:
            # Use output directory with original filename as base (strip existing node IDs)
            original_filename = os.path.basename(str(node.yaml_file_path)).removesuffix(
                ".yaml"
            )
            # Remove any existing node ID suffix
            if "_" in original_filename:
                # Split and take only the first part before any underscores with numbers
                parts = original_filename.split("_")
                base_name = parts[0]
                # Check if subsequent parts are node IDs (numbers or multi-instance format)
                for part in parts[1:]:
                    if not (part.isdigit() or "-" in part):
                        base_name += "_" + part
            else:
                base_name = original_filename
            base_path = os.path.join(self.output_dir, base_name)
            os.makedirs(self.output_dir, exist_ok=True)
        else:
            # Use same directory as original pipeline
            base_path = str(node.yaml_file_path).removesuffix(".yaml")

        new_yaml_path = f"{base_path}_{node_id_for_file}.yaml"
        new_parsed_yaml["pipeline"]["output"][
            "path"
        ] = f"{base_path}_{node_id_for_file}.json"

        with open(new_yaml_path, "w") as file:
            yaml.dump(
                new_parsed_yaml,
                file,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            )

        # generate the child node
        child = Node(
            yaml_file_path=new_yaml_path,
            parent=node,
            console=self.console,
            message_history=message_condensed,
            id=custom_id,
            is_multi_instance=is_multi_instance,
        )
        action = self.directive_name_to_obj.get(directive_name)
        child.latest_action = action

        # Copy parent's memo and add new entries for this action
        child.memo = node.memo.copy()
        for target_op in target_op_list:
            child.add_memo_entry(directive_name, target_op)

        if directive_name == "gleaning":
            chaining = self.directive_name_to_obj.get("chaining")
            assert chaining
            chunking = self.directive_name_to_obj.get("doc_chunking")
            assert chunking
            gleaning = self.directive_name_to_obj.get("gleaning")
            assert gleaning
            for op in target_op_list:
                child.mark_action_used(op, chaining)
                child.mark_action_used(op, chunking)
                child.mark_action_used(op, gleaning)

        elif directive_name == "change model":
            change_model = self.directive_name_to_obj.get("change model")
            assert change_model
            for op in target_op_list:
                child.mark_action_used(op, change_model)

        elif directive_name == "doc_chunking":
            doc_chunking = self.directive_name_to_obj.get("doc_chunking")
            assert doc_chunking
            for op in child.parsed_yaml["operations"]:
                op_name = op["name"]
                child.mark_action_used(op_name, doc_chunking)

        elif directive_name == "deterministic_doc_compression":
            d_comp = self.directive_name_to_obj.get("deterministic_doc_compression")
            assert d_comp
            for op in child.parsed_yaml["operations"]:
                op_name = op["name"]
                child.mark_action_used(op_name, d_comp)

        elif directive_name == "reduce_chaining":
            reduce_chain = self.directive_name_to_obj.get("reduce_chaining")
            assert reduce_chain
            for op in child.parsed_yaml["operations"]:
                op_name = op["name"]
                child.mark_action_used(op_name, reduce_chain)

        # Check if this was a compression directive and mark newly generated operators
        # to exclude all compression directives from their action space
        compression_directive_names = [
            d.name for d in DIRECTIVE_GROUPS.get("compression", [])
        ]
        if directive_name in compression_directive_names:
            # Find newly created operators by comparing old and new operation lists
            old_op_names = {op["name"] for op in node.parsed_yaml["operations"]}
            new_op_names = {op["name"] for op in new_ops_list}
            newly_created_ops = new_op_names - old_op_names

            # Mark all compression directives as "used" for newly created operators
            # This prevents compression directives from being applied to operators
            # that were themselves generated by compression directives
            compression_directives = DIRECTIVE_GROUPS.get("compression", [])
            for new_op_name in newly_created_ops:
                for compression_directive in compression_directives:
                    child.mark_action_used(new_op_name, compression_directive)

        node.add_child(child)
        return child
