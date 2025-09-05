import json
import math
import os
import re
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from typing import Any, Dict, List, Optional

import litellm
from litellm import completion_cost
import yaml
import threading

from docetl.reasoning_optimizer.directives import (
    ALL_COST_DIRECTIVES,
    ALL_DIRECTIVES,
    DIRECTIVE_GROUPS,
    MULTI_INSTANCE_DIRECTIVES,
    Directive,
    get_all_cost_directive_strings,
    get_all_directive_strings,
)
from docetl.reasoning_optimizer.directives.change_model_cost import (
    ChangeModelCostDirective,
    create_model_specific_directives,
    MODEL_COSTS,
    MODEL_STATS,
    first_layer_yaml_paths,
    FRONTIER_MODELS
)
from docetl.reasoning_optimizer.load_data import load_input_doc
from docetl.reasoning_optimizer.op_descriptions import *
from docetl.utils import extract_output_from_json

from .acc_comparator import AccuracyComparator
from .Node import Node
from .ParetoFrontier import ParetoFrontier
from .mcts_utils import *
from pathlib import Path
from experiments.reasoning.utils import VOLUME_MOUNT_PATH

# Import evaluation function lookup
import sys
sys.path.append("../../experiments/reasoning")
try:
    from experiments.reasoning.evaluation.utils import get_evaluate_func
except ImportError:
    # Fallback import path
    sys.path.append(
        os.path.join(os.path.dirname(__file__), "../../experiments/reasoning")
    )
    from evaluation.utils import get_evaluate_func



class MCTS:
    """
    Monte Carlo Tree Search (MCTS) implementation for DocETL query plan optimization.

    This class implements the four phases of MCTS with Pareto frontier integration:
    1. Selection: Choose the best child using UCB with Pareto value consideration
    2. Expansion: Add new children to the tree
    3. Simulation: Execute a random policy to get a value
    4. Backpropagation: Update node values up the tree

    The MCTS optimizes for both cost and accuracy using the Pareto frontier.
    """

    def __init__(
        self,
        root_yaml_path: str,
        accuracy_comparator: AccuracyComparator,
        available_actions: set[Directive],
        sample_input,
        dataset_stats: str,
        dataset_name: str,
        exploration_constant: float = 1.414,
        max_iterations: int = 20,
        max_time: Optional[float] = 600.0,
        expansion_count: int = 6,
        model="gpt-4.1",
        output_dir: Optional[str] = None,
        original_query_result: Optional[Dict[str, Any]] = None,
        build_first_layer: Optional[bool] = False,
    ):
        """
        Initialize the MCTS algorithm with Pareto frontier integration.

        Args:
            root_yaml_path: Path to the initial YAML configuration file
            accuracy_comparator: Comparator for evaluating plan accuracy
            available_actions: List of available actions for expansion
            exploration_constant: UCB exploration constant (default: sqrt(2))
            max_iterations: Maximum number of MCTS iterations
            max_time: Maximum time to run MCTS in seconds (None for no limit)
            sample_input: sample input data
            output_dir: Directory to save new pipeline files (None means same dir as original)
            original_query_result: Pre-executed original query result to avoid re-execution
        """

        self.root = Node(root_yaml_path, c=exploration_constant)
        self.tree_lock = threading.RLock()
        self.max_concurrent_agents = 3
        
        # Start with base available actions
        self.available_actions = set(available_actions)

        # Add all possible model-specific directives to available actions
        all_models = list(MODEL_COSTS.get(dataset_name, {}).keys()) if dataset_name in MODEL_COSTS else []
        
        for current_model in all_models:
            model_specific_directive = create_model_specific_directives(current_model)
            self.available_actions.add(model_specific_directive)
        
        # Initialize root node's action_used to include all change model directives
        for op_name in self.root.op_dict.keys():
            for directive in self.available_actions:
                if isinstance(directive, ChangeModelCostDirective):
                    self.root.mark_action_used(op_name, directive)
        
        # Initialize action tracking for all actions (base + model-specific)
        self.action_rewards = {action: 0.0 for action in self.available_actions}
        self.action_cost_changes = {action: 0.0 for action in self.available_actions}
        self.action_accuracy_changes = {action: 0.0 for action in self.available_actions}
        self.action_counts = {action: 0.0 for action in self.available_actions}
        self.exploration_constant = exploration_constant
        self.max_iterations = max_iterations
        self.max_time = max_time
        self.iteration_count = 0
        self.expansion_count = expansion_count
        self.start_time = None
        self.model = model
        self.sample_input = sample_input
        self.dataset_stats = dataset_stats
        self.dataset_name = dataset_name
        self.output_dir = output_dir
        
        # Set up evaluation function and dataset metrics
        self.evaluate_func = get_evaluate_func(dataset_name)
        self.dataset_metrics = {
            "cuad": "avg_f1",
            "blackvault": "avg_distinct_locations",
            "game_reviews": "weighted_score",
            "medec": "combined_score", 
            "sustainability": "economic_activity_accuracy",
            "biodex": "avg_rp_at_5",
        }
        # Set up log file path
        if self.output_dir:
            self.log_path = os.path.join(self.output_dir, "mcts_tree_log.txt")
        else:
            self.log_path = "mcts_tree_log.txt"

        # Initialize log file (clear it)
        with open(self.log_path, "w", encoding="utf-8") as f:
            f.write(f"MCTS Tree Visits and Values Log\n")
            f.write(f"Root YAML: {root_yaml_path}\n")
            f.write(f"Max iterations: {max_iterations}\n")
            f.write(f"{'='*50}\n")

        # Initialize Pareto frontier
        self.pareto_frontier = ParetoFrontier(
            accuracy_comparator, self.action_rewards, self.action_cost_changes, 
            self.action_accuracy_changes, dataset_name
        )

        # Create comprehensive directive mapping 
        self.directive_name_to_obj = {
            action.name: action for action in self.available_actions
        }
       
        # Track iterations without new Pareto optimal plans for early stopping
        self.iterations_without_improvement = 0
        
        # Track total cost for all completion calls during search
        self.total_search_cost = 0.0

        if build_first_layer:
            print("🔄 Building first layer of the pipeline manually...")
            output_dir = str(Path(VOLUME_MOUNT_PATH) / "outputs")
            for model_yaml_path in first_layer_yaml_paths[dataset_name]:
                model_yaml_path = Path(output_dir) / f"{dataset_name}" / model_yaml_path
                child = Node(model_yaml_path)
                child_model = child.parsed_yaml["default_model"]
                child_model = child_model.replace("azure/", "").replace("gemini/", "")
                child.cost = MODEL_COSTS[dataset_name][child_model]
                child.value = 0
                child.visits = 1
                action_name = "change to " + child_model
                action = self.directive_name_to_obj[action_name]
                child.latest_action = action
                child.sample_result = extract_output_from_json(model_yaml_path)
                
                # Mark all change model directives as used for first layer nodes
                # since they are already model-specific plans
                for op_name in child.op_dict.keys():
                    for directive in self.available_actions:
                        if isinstance(directive, ChangeModelCostDirective):
                            child.mark_action_used(op_name, directive)
                
                self.root.add_child(child)
                accuracy = MODEL_STATS[dataset_name][child_model]["acc"]
                self.pareto_frontier.add_plan_f1(child, accuracy)
            
            print(self.pareto_frontier.plans)
            self.root.visits = len(first_layer_yaml_paths[dataset_name])
            self.log_tree_to_file(0)
            return
        
        # Use original query result if provided, otherwise execute root node
        if original_query_result and original_query_result.get("success"):
            print("🔄 Using pre-executed original query result for MCTS root node")
            # Set root node properties from original query result
            self.root.cost = original_query_result["cost"]
            self.root.sample_result = original_query_result.get("sample_output", [])

            # Update the root node's YAML to point to the original query result file
            if original_query_result.get("output_file_path"):
                print(
                    f"📁 Setting root node output path to: {original_query_result['output_file_path']}"
                )
                try:
                    self.root.parsed_yaml["pipeline"]["output"]["path"] = (
                        original_query_result["output_file_path"]
                    )
                    self.root.result_path = original_query_result["output_file_path"]
                except Exception as e:
                    print(f"⚠️ Could not update root node output path: {e}")
                    
            # Evaluate root node accuracy and add to pareto frontier
            root_accuracy = self.evaluate_node(self.root)
            affected_nodes, is_frontier_updated = self.pareto_frontier.add_plan_f1(self.root, root_accuracy)
            self.root.visits = 1
        else:
            print(
                "▶️ Executing root node for MCTS (original query result not available)"
            )
            # execute root node and add it to the pareto frontier
            cost, accuracy = self.simulate(self.root)
            with self.tree_lock:
                cost_to_add = max(cost,0)
                print(f"💰 Adding simulation cost: ${cost_to_add:.4f} (total before: ${self.total_search_cost:.4f})")
                self.total_search_cost += cost_to_add
            affected_nodes, is_frontier_updated = self.add_to_frontier(self.root, accuracy)
            with self.tree_lock:
                self.backpropagate(affected_nodes, self.root)
            self.root.visits = 1


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
        print("result_file_path", result_file_path)

        try:
            results = self.evaluate_func("docetl_preprint", result_file_path)

            # Extract the appropriate metric based on dataset
            primary_metric = self.dataset_metrics.get(self.dataset_name)
            if primary_metric and primary_metric in results:
                true_accuracy = results[primary_metric]
            else:
                # Fallback to first numerical value found if dataset unknown or metric missing
                true_accuracy = next(
                    (v for v in results.values() if isinstance(v, (int, float))), 0.5
                )
                
        except Exception as e:
            print(f"⚠️ Evaluation failed for node {node.get_id()}: {e}")
            # Return -inf when evaluation fails, same as unexecutable plans
            return float("-inf")
            
        # Guard against NaN values
        if true_accuracy is None or (isinstance(true_accuracy, float) and (true_accuracy != true_accuracy)):  # NaN check
            print(f"⚠️ Evaluation returned NaN for node {node.get_id()}, setting to -inf")
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

        print(f"Starting concurrent MCTS search with {self.max_iterations} iterations and {self.max_concurrent_agents} agents...")
        print(f"Root node cost: ${self.root.cost:.2f}")

        with ThreadPoolExecutor(max_workers=self.max_concurrent_agents) as executor:
            while self.should_continue():
                # Submit up to max_concurrent_agents tasks
                futures = []
                agents_to_submit = min(self.max_concurrent_agents, self.max_iterations - self.iteration_count)
                
                for _ in range(agents_to_submit):
                    future = executor.submit(self.mcts_iteration)
                    futures.append(future)
                
                # Wait for at least one agent to complete
                for future in as_completed(futures):
                    try:
                        success = future.result()
                        if success:
                            with self.tree_lock:
                                self.iteration_count += 1
                    except Exception as e:
                        print(f"Agent failed with error: {e}")
                    
                    # Check if we should continue after each completion
                    if not self.should_continue():
                        break
                
                # Cancel remaining futures if stopping
                if not self.should_continue():
                    for future in futures:
                        future.cancel()

        # Final statistics
        print(f"\nConcurrent MCTS search completed!")
        print(f"Total iterations: {self.iteration_count}")
        print(f"Pareto frontier size: {len(self.pareto_frontier)}")
        print(f"Frontier plans: {len(self.pareto_frontier.frontier_plans)}")
        print("pareto frontier yaml files: ")
        for plan in self.pareto_frontier.frontier_plans:
            print(plan.yaml_file_path)

        # Return all frontier plans
        frontier_plans = [
            summary["node"]
            for summary in self.pareto_frontier.get_all_plans_summary()
            if summary["is_frontier"]
        ]

        return frontier_plans


    def mcts_iteration(self):
        """Perform one complete MCTS iteration with thread safety."""
        thread_id = threading.current_thread().ident
        
        # 1. Selection: Find the best leaf node (requires tree lock)
        with self.tree_lock:
            print(f"[Thread {thread_id}] SELECTION")
            leaf = self.select(self.root)
            print(f"[Thread {thread_id}] SELECTED NODE: {leaf.get_id()}")

        # 2. Expansion: Always attempt to expand the leaf, catch errors (requires tree lock)
        with self.tree_lock:
            print(f"[Thread {thread_id}] EXPANSION")
            acc_children = []

            has_leaf_acc = 1
            try:
                acc_children = self.expand(leaf, optimize_goal="acc")
                # Increment visit count immediately after successful expansion
                if acc_children:
                    self.increment_visits_up_tree(leaf)
            except RuntimeError as e:
                print(e)
                has_leaf_acc = 0

        # 3. Simulation: Run simulations from the leaf

        is_frontier_updated = False
        if has_leaf_acc:
            print("HAS LEAF ACC SIMULATION")
            
            if len(acc_children) > 1:
                print(f"Handling {len(acc_children)} multi-instance candidates")
                candidate_results = []
                
                # Simulate all candidates
                for num, candidate in enumerate(acc_children):
                    cost, accuracy = self.simulate(candidate)
                    with self.tree_lock:
                        cost_to_add = max(cost,0)
                        print(f"💰 Adding multi-instance acc candidate cost: ${cost_to_add:.4f} (total before: ${self.total_search_cost:.4f})")
                        self.total_search_cost += cost_to_add
                    print(f"Multi-instance candidate {num} - Cost: ${cost:.2f}, Accuracy: {accuracy:.4f}")
                    if cost != -1 and accuracy != float("-inf"):  # Valid plan
                        candidate_results.append((candidate, accuracy, cost))
                    else:
                        print(f"Multi-instance candidate {num} failed during simulation")
                
                if candidate_results:
                    # Select the best candidate based on accuracy
                    best_candidate, best_accuracy, best_cost = max(candidate_results, key=lambda x: x[1])
                    print(f"Selected best multi-instance candidate {best_candidate.get_id()} with accuracy {best_accuracy:.4f} and cost ${best_cost:.2f}")
                    
                    # Change the best candidate's ID back to a proper counter ID
                    old_id = best_candidate.get_id()
                    new_id = best_candidate.set_id_to_counter()
                    print(f"Updated best candidate ID from {old_id} to {new_id}")
                    
                    # Delete ALL non-selected candidates (both failed and successful ones that weren't chosen)
                    for candidate in acc_children:
                        if candidate != best_candidate:
                            candidate.delete(selected_node_final_id=new_id)
                    
                    # Process only the best candidate (requires tree lock for backprop)
                    affected_nodes, is_frontier_updated = self.add_to_frontier(best_candidate, best_accuracy)
                    with self.tree_lock:
                        self.backpropagate(affected_nodes, best_candidate)
                else:
                    # If no candidates were successful, delete all of them
                    print("No successful candidates found, deleting all multi-instance candidates")
                    for candidate in acc_children:
                        candidate.delete(selected_node_final_id=None)
                    
            else:
                # Original logic for single instantiations
                for leaf_acc in acc_children:
                    cost, accuracy = self.simulate(leaf_acc)
                    with self.tree_lock:
                        cost_to_add = max(cost,0)
                        print(f"💰 Adding leaf acc simulation: ${cost_to_add:.4f} (total before: ${self.total_search_cost:.4f})")
                        self.total_search_cost += cost_to_add
                    affected_nodes, temp_updated = self.add_to_frontier(leaf_acc, accuracy)
                    if temp_updated:
                        is_frontier_updated = True
                    # Check if any node was added to the frontier (value = 1)
                    with self.tree_lock:
                        self.backpropagate(affected_nodes, leaf_acc)

            # Update counter for early stopping (requires tree lock)
            with self.tree_lock:
                if is_frontier_updated:
                    self.iterations_without_improvement = 0
                else:
                    self.iterations_without_improvement += 1

                self.log_tree_to_file(self.iteration_count + 1)
        return has_leaf_acc

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
            print(f"Node {current.get_id()} is fully explored, selecting best child")
            current = current.best_child()

        return current


    def is_fully_explored(self, node: Node) -> bool:
        """Check if a node has been fully explored based on visit count."""
        return is_fully_explored(node)

    def expansion_prompt_acc(self, node, action_options, input_query) -> tuple[str, str]:
        return create_expansion_prompt_acc(
            node, action_options, input_query, self.available_actions, 
            self.action_cost_changes, self.action_accuracy_changes, self.action_counts, self.sample_input, 
            self.root, node.yaml_file_path, self.dataset_name, self.pareto_frontier.plans_accuracy
        )

    def expansion_prompt_cost(self, node, action_options, input_query) -> tuple[str, str]:
        return create_expansion_prompt_cost(
            node, action_options, input_query, self.available_actions,
            self.action_cost_changes, self.action_accuracy_changes, self.action_counts, self.sample_input,
            self.root, node.yaml_file_path, self.dataset_name, self.pareto_frontier.plans_accuracy
        )

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
        Expand a leaf node by adding one new child and return the child.

        Args:
            node: Leaf node to expand
            optimize_goal: The optimization goal, e.g., 'acc' or 'cost'
        Returns:
            The newly created child
        """

        print("INSIDE EXPAND for node: ", node.get_id())

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

        optimize_goal = self.get_optimize_goal(node)
        print(f"Optimize goal: {optimize_goal}")
        if optimize_goal == "acc":
            action_options = []  # a list of tuple
            for op_name in op_list:
                if op_name in node.used_actions:
                    used_actions = node.used_actions[op_name]
                else:
                    used_actions = set()
                
                # Get compression directives to exclude for code_map and extract operations
                compression_exclusions = get_excluded_directives_for_operation(node, op_name)
                
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
                    set(dynamic_actions) - banned_directives - used_actions - compression_exclusions
                )  # The actions that are not used on this operator and not excluded by group
                for action in action_space:
                    action_options.append((op_name, action.name))
            if len(action_options) < 1:
                print("NO ACTION FOUND")
                raise RuntimeError(
                    "No applicable action found for expansion. Action space may be exhausted or all actions are inapplicable."
                )
            print("OPTIMIZING ACC:")
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
                compression_exclusions = get_excluded_directives_for_operation(node, op_name)
                
                if last_op is None or last_op != op_name:
                    banned_directives = set()

                # Filter cost directives to only include cheaper models
                op_config = node.op_dict[op_name]
                current_model = op_config.get("model", "gpt-5")
                dynamic_cost_directives = []
                
                for directive in self.available_actions:
                    if isinstance(directive, ChangeModelCostDirective) and directive.target_model:
                        if self._is_cheaper_model(directive.target_model, current_model) and directive.target_model in FRONTIER_MODELS[self.dataset_name]:
                            dynamic_cost_directives.append(directive)
                    else:
                        dynamic_cost_directives.append(directive)
                
                action_space = (
                    set(dynamic_cost_directives) - banned_directives - used_actions - compression_exclusions
                )  # The cost actions that are not used on this operator and not excluded by group
                for action in action_space:
                    action_options.append((op_name, action.name))
            print("OPTIMIZING COST:")
            if len(action_options) < 1:
                print("NO ACTION FOUND")
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
                print(f"💰 Adding LLM call cost: ${call_cost:.4f} (total before: ${self.total_search_cost:.4f})")
                self.total_search_cost += call_cost
            reply = response.choices[0].message.content

            try:
                parsed = json.loads(reply)
                directive_name = parsed.get("directive")
                target_op_list = parsed.get("operators")
                print(f"Directive: {directive_name}, Target ops: {target_op_list}")
                messages.append({"role": "assistant", "content": reply})
                message_condensed.append({"role": "assistant", "content": reply})
            except Exception as e:
                print(f"Failed to parse agent response: {e}")
                retry_count += 1
                continue


            # Check if directive is already used for this plan + target ops
            directive = self.directive_name_to_obj.get(directive_name)
            if directive is None:
                print(f"Unknown directive name: {directive_name}")
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
                print(
                    f"Directive '{directive_name}' already used for these ops. Retry {retry_count}/{max_retries}"
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
        
        print(f"Creating {num_instantiations} instantiation(s) for directive '{directive_name}'")
        
        children = []
        instantiation_messages = messages.copy()
        for i in range(num_instantiations):
            try:
                print(f"Creating instantiation {i+1}/{num_instantiations}")
                
                # For multi-instance directives, add variation to each instantiation
                if is_multi_instance:
                    # Use accumulated message history (includes previous instantiations)
                    # Add instantiation-specific context to encourage variation
                    if i == 0:
                        variation_prompt = f"This is instantiation {i+1} of {num_instantiations} for the '{directive_name}' directive. Focus on creating a distinct approach by exploring different parameter combinations or implementation strategies. For example, you can try different models, different parameter settings (chunk size, top k, etc.), or different implementation strategies."
                    else:
                        variation_prompt = f"This is instantiation {i+1} of {num_instantiations} for the '{directive_name}' directive. Based on the previous {i} instantiation(s) shown above, create a DIFFERENT approach that explores alternative parameters, strategies, or implementations. AVOID repeating the exact same configs asthe previous instantiations."
                    
                    
                    # Insert variation context before the last user message
                    variation_msg = {
                        "role": "system",
                        "content": variation_prompt
                    }
                    instantiation_messages.append(variation_msg)

                with open("instantiation_messages_debug.txt", "w", encoding="utf-8") as f:
                    f.write(f"i: {i}\n")
                    f.write("instantiation_messages: \n")
                    f.write(str(instantiation_messages))
                    f.write("\n\n")
                
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
                )
                with self.tree_lock:
                    print(f"💰 Adding agent instantiation cost: ${cost:.4f} (total before: ${self.total_search_cost:.4f})")
                    self.total_search_cost += cost
                if new_ops_list is None:
                    print(f"Instantiation {i+1} failed: no ops list returned")
                    continue

                instantiation_messages = updated_message_history

                # Create child node for this instantiation
                # For multi-instance directives, use parent_id-instantiation_num format
                custom_id = f"{node.get_id()}-{i+1}" if is_multi_instance else None
                child = self.instantiate_node(
                    node, new_ops_list, directive_name, target_op_list, optimize_goal, 
                    message_condensed + updated_message_history[message_length:], custom_id
                )
                
                children.append(child)
                print(f"Instantiation {i+1} created successfully")

            except Exception as e:
                print(f"Instantiation {i+1} failed with error: {str(e)}")
                continue

        if not children:
            raise RuntimeError(f"All {num_instantiations} instantiation(s) failed for directive '{directive_name}'")
        
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

        try:
            # Step 1: Execute the plan (this will set node.cost)
            node.execute_plan()
        except Exception as e:
            print(f"Failed to execute plan for node {node.get_id()}: {str(e)}")
            # Set cost to -1 to indicate failure (this is already done in Node.execute_plan)
            # Do not add failed plans to the frontier
            return cost, accuracy

        # Step 2: Evaluate the plan (this will call the evaluation function)
        try:
            accuracy = self.evaluate_node(node)
            cost = node.cost
            print(f"Node {node.get_id()} evaluation - cost: ${cost:.2f}, accuracy: {accuracy:.4f}")
        except Exception as e:
            print(f"Failed to evaluate plan for node {node.get_id()}: {str(e)}")
            return cost, accuracy
        
        return cost, accuracy

    def add_to_frontier(self, node: Node, accuracy: float):
        # Only decrement action count for failed plans

        if node.cost == -1 and node.latest_action in self.action_counts:
            self.action_counts[node.latest_action] -= 1

        # Step 3: Add to frontier (this only manages frontier state, no evaluation)
        affected_nodes, is_frontier_updated = self.pareto_frontier.add_plan_f1(node, accuracy)
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
            print(
                f"Early stopping: No Pareto optimal plans found in last {self.iterations_without_improvement} iterations"
            )
            return False

        return True

    def get_frontier_summary(self) -> List[Dict[str, Any]]:
        """Get summary of all plans in the Pareto frontier."""
        return self.pareto_frontier.get_all_plans_summary()
    
    def _is_cheaper_model(self, target_model: str, current_model: str) -> bool:
        """Check if target_model is cheaper than current_model for this dataset."""
        if self.dataset_name not in MODEL_COSTS:
            return False
        
        dataset_costs = MODEL_COSTS[self.dataset_name]
        # Remove 'azure/' or 'gemini/' prefix from current_model if present
        if current_model.startswith("azure/"):
            current_model = current_model[len("azure/") :]
        if current_model.startswith("gemini/"):
            current_model = current_model[len("gemini/") :]
        # Remove 'azure/' or 'gemini/' prefix from target_model if present
        if target_model.startswith("azure/"):
            target_model = target_model[len("azure/") :]
        if target_model.startswith("gemini/"):
            target_model = target_model[len("gemini/") :]
            
        current_cost = dataset_costs.get(current_model)
        target_cost = dataset_costs.get(target_model)

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
            print(output)

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
            f.write(f"Action Performance Statistics:\n")
            for action in self.available_actions:
                cost_change = self.action_cost_changes.get(action, 0)
                accuracy_change = self.action_accuracy_changes.get(action, 0)
                count = self.action_counts.get(action, 0)
                
                if count > 0:
                    avg_cost_change = cost_change / count
                    avg_accuracy_change = accuracy_change / count
                    f.write(f"- {action.name}: {count} uses, avg change in cost: {avg_cost_change:+.2f}, avg change in accuracy: {avg_accuracy_change:+.4f}\n")
                else:
                    f.write(f"- {action.name}: {count} uses, avg change in cost: Unknown (never tried), avg change in accuracy: Unknown (never tried)\n")
            f.write(f"\n")

            # Add tree structure
            f.write(f"Tree Structure:\n")
            self.print_tree_visits_and_values(file_handle=f)
            f.write(f"\n")
            f.write(f"Total search cost: ${self.total_search_cost:.2f}\n")

    def instantiate_node(
        self, node, new_ops_list, directive_name, target_op_list, optimize_goal, message_condensed, custom_id=None
    ):
        """
        Instantiate a new child node by applying the directive to the given node and target operations.
        Args:
            node: The parent node
            directive: The directive object to apply
            directive_name: The name of the directive
            target_op_list: List of target operations
            optimize_goal: The optimization goal (e.g., 'acc' or 'cost')
        Returns:
            The newly created child node
        """

        new_parsed_yaml = deepcopy(node.parsed_yaml)
        new_parsed_yaml["operations"] = new_ops_list
        new_parsed_yaml["bypass_cache"] = True
        new_parsed_yaml = update_pipeline(
            new_parsed_yaml, new_ops_list, target_op_list
        )

        fix_models(new_parsed_yaml)

        # Determine the node ID to use for filename
        if custom_id is not None:
            node_id_for_file = custom_id
        else:
            node_id_for_file = Node.get_next_id()
        
        # Determine where to save the new pipeline file
        if self.output_dir:
            # Use output directory with original filename as base (strip existing node IDs)
            original_filename = os.path.basename(str(node.yaml_file_path)).removesuffix(".yaml")
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

        print("NEW YAML FILE: ", new_yaml_path)

        # generate the child node
        child = Node(yaml_file_path=new_yaml_path, parent=node, message_history=message_condensed, id=custom_id)
        action = self.directive_name_to_obj.get(directive_name)
        child.latest_action = action

        # Copy parent's memo and add new entries for this action
        child.memo = node.memo.copy()
        for target_op in target_op_list:
            child.add_memo_entry(directive_name, target_op)
            
        print("last action: ", child.latest_action.name, "child.id: ", child.id)
        
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
        compression_directive_names = [d.name for d in DIRECTIVE_GROUPS.get("compression", [])]
        if directive_name in compression_directive_names:
            # Find newly created operators by comparing old and new operation lists
            old_op_names = {op["name"] for op in node.parsed_yaml["operations"]}
            new_op_names = {op["name"] for op in new_ops_list}
            newly_created_ops = new_op_names - old_op_names
            print("newly_created_ops: ", newly_created_ops)
            
            # Mark all compression directives as "used" for newly created operators
            # This prevents compression directives from being applied to operators
            # that were themselves generated by compression directives
            compression_directives = DIRECTIVE_GROUPS.get("compression", [])
            for new_op_name in newly_created_ops:
                for compression_directive in compression_directives:
                    child.mark_action_used(new_op_name, compression_directive)
                print(f"Marked compression directives as excluded for newly created operator: {new_op_name}")

        node.add_child(child)
        return child
