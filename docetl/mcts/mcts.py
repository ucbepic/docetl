import json
import math
import os
import time
from copy import deepcopy
from typing import Any, Dict, List, Optional

import litellm
import yaml
import re

from docetl.reasoning_optimizer.directives import (
    ALL_DIRECTIVES,
    ALL_COST_DIRECTIVES,
    Directive,
    get_all_directive_strings,
    get_all_cost_directive_strings,
)
from docetl.reasoning_optimizer.load_data import load_input_doc
from docetl.reasoning_optimizer.op_descriptions import *

from .acc_comparator import AccuracyComparator
from .Node import Node
from .ParetoFrontier import ParetoFrontier


def count_num_pass(mcts_instance, node: Node) -> int:
    """
    Count the number of times the main content key is mentioned in prompts of 
    map/filter/extract/reduce operators.
    
    Args:
        mcts_instance: The MCTS instance containing the main_content_key
        node: The node containing the parsed_yaml to analyze
        
    Returns:
        Total count of main content key mentions across all relevant operators
    """
    relevant_ops = ["map", "filter", "extract", "reduce"]
    total_count = 0
    
    # Use the pre-identified main content key from MCTS instance
    main_content_key = mcts_instance.main_content_key
    
    if not main_content_key:
        print("No main content key available")
        return 0
    
    print("--"*50)
    print(f"Using main content key: {main_content_key}")
    
    # Count mentions in relevant operators
    operations = node.parsed_yaml.get("operations", [])
    for op in operations:
        op_type = op.get("type", "").lower()
        if op_type in relevant_ops:
            prompt = op.get("prompt", "")
            if isinstance(prompt, str):
                # Count occurrences of {{ anything.main_content_key }}
                pattern = rf"\{{\{{\s*\w+\.{re.escape(main_content_key)}\s*\}}\}}"
                matches = re.findall(pattern, prompt)
                count_in_op = len(matches)
                total_count += count_in_op
                print(f"Operator {op.get('name', 'unnamed')} ({op_type}): {count_in_op} mentions of {main_content_key}")
    
    print(f"Total mentions of main content key '{main_content_key}': {total_count}")
    print("--"*50)
    return total_count


class ExpandResponseFormat(BaseModel):
    directive: str
    operators: List[str]


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
        """
        self.root = Node(root_yaml_path, c=exploration_constant)
        self.available_actions = available_actions
        self.action_rewards = {action: 0.0 for action in available_actions}
        self.action_counts = {
            action: 0.0 for action in available_actions
        }  # number of times an action has been applied
        self.exploration_constant = exploration_constant
        self.max_iterations = max_iterations
        self.max_time = max_time
        self.iteration_count = 0
        self.expansion_count = expansion_count
        self.start_time = None
        self.model = model
        self.sample_input = sample_input
        self.dataset_stats = dataset_stats
        self.output_dir = output_dir
        
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
            accuracy_comparator, self.action_rewards, dataset_name
        )
        self.directive_name_to_obj = {
            action.name: action for action in self.available_actions
        }

        # Track iterations without new Pareto optimal plans for early stopping
        self.iterations_without_improvement = 0

        # Identify main content key once during initialization
        self.main_content_key = self._identify_main_content_key()

        # execute root node and add it to the pareto frontier
        _ = self.simulate(self.root)
        self.root.visits = 1

    def _identify_main_content_key(self) -> str:
        """
        Identify the main content key (key with longest value) from the input dataset.
        
        Returns:
            The key name with the longest average value, or None if not found
        """
        try:
            datasets = self.root.parsed_yaml.get("datasets", {})
            if datasets:
                # Get the first dataset's path
                for dataset_name, dataset_config in datasets.items():
                    if isinstance(dataset_config, dict) and "path" in dataset_config:
                        input_file_path = dataset_config["path"]
                        # Load a sample to find the key with longest value
                        with open(input_file_path, 'r') as f:
                            data = json.load(f)
                            if data and isinstance(data, list) and data[0]:
                                sample_item = data[0]
                                if isinstance(sample_item, dict):
                                    # Find key with longest average value length
                                    max_avg_length = 0
                                    main_content_key = None
                                    for key, value in sample_item.items():
                                        if isinstance(value, str):
                                            if len(value) > max_avg_length:
                                                max_avg_length = len(value)
                                                main_content_key = key
                                    print(f"Main content key identified: {main_content_key}")
                                    return main_content_key
                        break
        except Exception as e:
            print(f"Could not determine main content key: {e}")
        
        print("No main content key found")
        return None

    def search(self):
        """
        Perform MCTS search to find the optimal query plan.

        Returns:
            Tuple of (best_node, search_statistics)
        """
        self.start_time = time.time()
        self.iteration_count = 0

        print(f"Starting MCTS search with {self.max_iterations} iterations...")
        print(f"Root node cost: ${self.root.cost:.2f}")

        while self.should_continue():
            if self.iteration_count < 5: 
            # if self.iteration_count >= self.max_iterations - 5:
                if self.mcts_cost_iteration():
                    self.iteration_count += 1
            else: 
                if self.mcts_iteration():
                    self.iteration_count += 1

        # Final statistics
        print(f"\nMCTS search completed!")
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

    def mcts_cost_iteration(self):
        """Perform one complete MCTS iteration optimizing for cost."""
        # Track if any new Pareto optimal plans are found in this iteration
        found_new_pareto_plan = False

        # 1. Selection: Find the best leaf node
        print("SELECTION (COST)")
        leaf = self.select(self.root)
        print("SELECTED NODE: ", leaf.get_id())

        # 2. Expansion: Always attempt to expand the leaf, catch errors
        print("EXPANSION (COST)")
        cost_children = []

        has_leaf_cost = 1
        try:
            cost_children = self.expand(leaf, optimize_goal="cost")
        except RuntimeError as e:
            print(e)
            has_leaf_cost = 0

        # 3. Simulation: Run simulations from the leaf
        is_frontier_updated = False
        if has_leaf_cost:
            print("HAS LEAF COST SIMULATION")
            for leaf_cost in cost_children:
                affected_nodes, is_frontier_updated = self.simulate(leaf_cost)
                # Check if any node was added to the frontier (value = 1)
                self.backpropagate(affected_nodes, leaf_cost)

            # Update counter for early stopping
            if is_frontier_updated:
                self.iterations_without_improvement = 0
            else:
                self.iterations_without_improvement += 1

        self.log_tree_to_file(self.iteration_count + 1)
        return has_leaf_cost

    def mcts_iteration(self):
        """Perform one complete MCTS iteration."""
        # Track if any new Pareto optimal plans are found in this iteration
        found_new_pareto_plan = False

        # 1. Selection: Find the best leaf node
        print("SELECTION")
        leaf = self.select(self.root)
        print("SELECTED NODE: ", leaf.get_id())

        # 2. Expansion: Always attempt to expand the leaf, catch errors
        print("EXPANSION")
        acc_children = []
        # cost_children = []

        has_leaf_acc = 1
        try:
            acc_children = self.expand(leaf, optimize_goal="acc")
        except RuntimeError as e:
            print(e)
            has_leaf_acc = 0

        # 3. Simulation: Run simulations from the leaf

        is_frontier_updated = False
        if has_leaf_acc:
            print("HAS LEAF ACC SIMULATION")
            for leaf_acc in acc_children:
                affected_nodes, is_frontier_updated = self.simulate(leaf_acc)
                # Check if any node was added to the frontier (value = 1)
                self.backpropagate(affected_nodes, leaf_acc)

            # Update counter for early stopping
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
            current = current.best_child()

        return current

    # Dummy implementation for now
    def is_action_applicable(self, node: Node, action: Directive) -> bool:
        return True

    def update_pipeline(self, orig_config, new_ops_list, target_ops):
        """
        Update the pipeline configuration with new operations.

        Args:
            orig_config (dict): The original pipeline configuration
            new_ops_list (list): List of new operations to add
            target_ops (list): List of target operation names to replace

        Returns:
            dict: Updated pipeline configuration
        """
        if new_ops_list is not None:
            op_names = [op.get("name") for op in new_ops_list if "name" in op]

        # Update the pipeline steps to use the new operation names
        if "pipeline" in orig_config and "steps" in orig_config["pipeline"]:
            for step in orig_config["pipeline"]["steps"]:
                if "operations" in step:
                    new_ops = []
                    for op in step["operations"]:
                        if op == target_ops[0]:
                            new_ops.extend(op_names)
                    step["operations"] = new_ops

        return orig_config

    def fix_models_azure(self, parsed_yaml):
        def traverse(obj):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if key == "model" and isinstance(value, str):
                        if not value.startswith("azure"):
                            obj[key] = f"azure/{value}"
                    else:
                        traverse(value)
            elif isinstance(obj, list):
                for item in obj:
                    traverse(item)

        traverse(parsed_yaml)

    def is_fully_explored(self, node: Node) -> bool:
        allowed_children = max(2, 1 + math.floor(math.sqrt(float(node.visits))))
        print(
            "# of children: ",
            len(node.children),
            " allowed_childre: ",
            allowed_children,
        )
        if len(node.children) >= allowed_children:
            return True
        for op in node.parsed_yaml["operations"]:
            op_name = op.get("name")
            print(op_name, len(node.used_actions[op_name]))
            if len(node.used_actions[op_name]) < len(ALL_DIRECTIVES):
                return False
        return True

    def expansion_prompt_acc(self, node, action_options, input_query) -> tuple[str, str]:

        availabel_actions_str = ""
        for item in action_options:
            op_name = item[0]
            action_name = item[1]
            action_str = f"Operator: {op_name}, Rewrite directive: {action_name}\n"
            availabel_actions_str += action_str

        print(availabel_actions_str)
        action_stats = []
        for action in self.available_actions:
            reward = self.action_rewards.get(action, 0)
            count = self.action_counts.get(action, 0)
            avg_reward = reward / count if count > 0 else "Unknown (never tried)"
            action_stats.append(
                f"- {action.name}: {count} uses, avg reward: {avg_reward}"
            )

        action_stats_str = "\n".join(action_stats)

        print(action_stats_str)

        input_schema = load_input_doc(node.yaml_file_path)

        user_message = f"""
        I have a set of operations used to process long documents, along with a list of possible rewrite directives aimed at improving the quality of the query result.
        Given a query pipeline made up of these operations, recommend one specific rewrite directive (specify by its name) that would improve accuracy and specify which operators (specify by their names) in the pipeline the directive should be applied to.
        Make sure that your chosen directive is in the provided list of rewrite directives.

        Pipeline:
        Pipelines in DocETL are the core structures that define the flow of data processing. A pipeline consists of five main components: \n
        - Default Model: The language model to use for the pipeline. Limit your choice of model to gpt-4.1-nano, gpt-4o-mini, gpt-4o, gpt-4.1 \n
        - System Prompts: A description of your dataset and the "persona" you'd like the LLM to adopt when analyzing your data. \n
        - Datasets: The input data sources for your pipeline. \n
        - Operators: The processing steps that transform your data. \n
        - Pipeline Specification: The sequence of steps and the output configuration. \n

        Operators:
        Operators form the building blocks of data processing pipelines. Below is the list of operators:
        {op_map.to_string()}\n
        {op_extract.to_string()}\n
        {op_parallel_map.to_string()}\n
        {op_filter.to_string()}\n
        {op_reduce.to_string()}\n
        {op_split.to_string()}\n
        {op_gather.to_string()}\n
        {op_unnest.to_string()}\n
        {op_sample.to_string()}\n
        {op_resolve.to_string()}\n

        Rewrite directives:
        {get_all_directive_strings()}\n

        Your valid choice of operation and rewrite directive combination. Only choose one of these:\n
        {availabel_actions_str}

        Action Performance History:
        Based on previous executions across DIFFERENT query pipelines, here's how each action has performed:\n
        {action_stats_str}

        Note: These statistics come from applying actions to various other query pipelines, not the current one. Use this as general guidance about action effectiveness, but consider that performance may vary significantly for your specific pipeline structure and data.

        Selection Strategy:
        Consider the current query pipeline, which directive can best improve the accuracy.
        Prioritize exploration of untested actions while balancing with exploitation of proven performers:
        - Actions with 0 uses have unknown potential, so you should explore them if applicable. Try change model directive if it has not been used in the past iterations. 
        - High average reward indicates good historical performance
        - Consider both immediate improvement and learning about the action space

        Make sure you read every rewrite directive carefully.
        Make sure you only choose from the valid choices above and avoid already used combinations.

        Input document schema with token statistics: {input_schema} \n
        Input data sample: {json.dumps(self.sample_input, indent=2)[:5000]} \n
        The original query in YAML format using our operations: {input_query} \n
        """
        
        # Create a condensed version for message history (without full operator/directive descriptions)
        condensed_user_message = f"""
        Recommend one specific rewrite directive for accuracy optimization.
        
        Valid choices:
        {availabel_actions_str}
        
        Action Performance History:
        {action_stats_str}
        
        Current pipeline: {input_query} 
        """
        
        return user_message, condensed_user_message


    def expansion_prompt_cost(self, node, action_options, input_query, num_of_passes=0) -> tuple[str, str]:

        availabel_actions_str = ""
        for item in action_options:
            op_name = item[0]
            action_name = item[1]
            action_str = f"Operator: {op_name}, Rewrite directive: {action_name}\n"
            availabel_actions_str += action_str

        print(availabel_actions_str)
        action_stats = []
        for action in self.available_actions:
            reward = self.action_rewards.get(action, 0)
            count = self.action_counts.get(action, 0)
            avg_reward = reward / count if count > 0 else "Unknown (never tried)"
            action_stats.append(
                f"- {action.name}: {count} uses, avg reward: {avg_reward}"
            )

        action_stats_str = "\n".join(action_stats)

        print(action_stats_str)

        input_schema = load_input_doc(node.yaml_file_path)

        user_message = f"""
        I have a set of operations used to process long documents, along with a list of possible rewrite directives designed to improve the cost effectiveness of the pipeline, while maintaining similar or better accuracy.
        Given a query pipeline composed of these operations, recommend one specific rewrite directive (identified by its name from the provided list) that would improve cost effectiveness. Also, specify which operator(s) (by name) in the pipeline the directive should be applied to.
        Make sure your recommended directive is selected from the provided list.

        Pipeline:
        Pipelines in DocETL are the core structures that define the flow of data processing. A pipeline consists of five main components: \n
        - Default Model: The language model to use for the pipeline. Limit your choice of model to gpt-4.1-nano, gpt-4o-mini, gpt-4o, gpt-4.1 \n
        - System Prompts: A description of your dataset and the "persona" you'd like the LLM to adopt when analyzing your data. \n
        - Datasets: The input data sources for your pipeline. \n
        - Operators: The processing steps that transform your data. \n
        - Pipeline Specification: The sequence of steps and the output configuration. \n

        Operators:
        Operators form the building blocks of data processing pipelines. Below is the list of operators:
        {op_map.to_string()}\n
        {op_extract.to_string()}\n
        {op_parallel_map.to_string()}\n
        {op_filter.to_string()}\n
        {op_reduce.to_string()}\n
        {op_split.to_string()}\n
        {op_gather.to_string()}\n
        {op_unnest.to_string()}\n
        {op_sample.to_string()}\n
        {op_resolve.to_string()}\n

        Rewrite directives:
        {get_all_cost_directive_strings()}\n

        Your valid choice of operation and rewrite directive combination. Only choose one of these:\n
        {availabel_actions_str}

        Action Performance History:
        Based on previous executions across DIFFERENT query pipelines, here's how each action has performed:\n
        {action_stats_str}

        Note: These statistics come from applying actions to various other query pipelines, not the current one. Use this as general guidance about action effectiveness, but consider that performance may vary significantly for your specific pipeline structure and data.

        Selection Strategy:
        Consider the current query pipeline, which directive can best improve cost effectiveness. 
        Prioritize exploration of untested actions while balancing with exploitation of proven performers:
        - Actions with 0 uses have unknown potential, so you should explore them if applicable.
        - High average reward indicates good historical performance
        - Consider both immediate improvement and learning about the action space

        Make sure you only choose from the valid choices above and avoid already used combinations.

        Input document schema with token statistics: {input_schema} \n
        Input data sample: {json.dumps(self.sample_input, indent=2)[:5000]} \n
        The original query in YAML format using our operations: {input_query} \n
        """
        
        # Create a condensed version for message history (without full operator/directive descriptions)
        condensed_user_message = f"""
        Recommend one specific rewrite directive for cost optimization.
        
        Valid choices:
        {availabel_actions_str}
        
        Action Performance History:
        {action_stats_str}
        
        Current pipeline: {input_query}
        """
        
        return user_message, condensed_user_message


    def expand(self, node: Node, optimize_goal: str) -> List[Node]:
        """
        Expand a leaf node by adding one new child and return the child.

        Args:
            node: Leaf node to expand
            optimize_goal: The optimization goal, e.g., 'acc' or 'cost'
        Returns:
            The newly created child
        """

        print("INSIDE EXPAND")
        print(node.get_id())

        max_retries = 3
        retry_count = 0

        num_of_passes = count_num_pass(self, node)

        # Build action options and initial prompt once
        op_list = list(node.op_dict.keys())

        if optimize_goal == "acc":
            action_options = []  # a list of tuple
            for op_name in op_list:
                if op_name in node.used_actions:
                    used_actions = node.used_actions[op_name]
                else:
                    used_actions = set()
                action_space = (
                    set(self.available_actions) - used_actions
                )  # The actions that are not used on this operator
                for action in action_space:
                    action_options.append((op_name, action.name))
            print(action_options)
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
                action_space = (
                    set(ALL_COST_DIRECTIVES) - used_actions
                )  # The cost actions that are not used on this operator
                for action in action_space:
                    action_options.append((op_name, action.name))
            print(action_options)
            print("OPTIMIZING COST:")
            if len(action_options) < 1:
                print("NO ACTION FOUND")
                raise RuntimeError(
                    "No applicable action found for expansion. Action space may be exhausted or all actions are inapplicable."
                )
            user_message, condensed_user_message = self.expansion_prompt_cost(
                node, action_options=action_options, input_query=node.parsed_yaml, num_of_passes=num_of_passes
            )

        # Initialize messages with accumulated message history from the path to this node
        messages = node.message_history.copy()
        
        # Add the current system message and user message for this expansion
        if not messages or messages[0]["role"] != "system":
            messages.insert(0, {
                "role": "system",
                "content": "You are an expert query optimization agent for document processing pipelines. Your role is to analyze user queries and apply rewrite directives to create more accurate and cost effective execution plans. Your output must follow the structured output format.",
            })
        
        messages.append({"role": "user", "content": user_message})

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
            reply = response.choices[0].message.content

            try:
                parsed = json.loads(reply)
                directive_name = parsed.get("directive")
                target_op_list = parsed.get("operators")
                print(f"Directive: {directive_name}, Target ops: {target_op_list}")
                messages.append({"role": "assistant", "content": reply})
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

        rewrites = []

        # Mark action as used
        for target_op in target_op_list:
            node.mark_action_used(target_op, directive)
     
        try:
            new_ops_list, message_history = directive.instantiate(
                operators=node.parsed_yaml["operations"],
                target_ops=target_op_list,
                agent_llm=self.model,
                optimize_goal=optimize_goal,
                global_default_model=orig_default_model,
                message_history=messages,
                input_file_path=input_file_path,
                pipeline_code=node.parsed_yaml,
            )
            if new_ops_list is None:
                raise RuntimeError(
                    "Failed to instantiate directive: no new ops list returned after retries."
                )

            rewrites.append(new_ops_list)

        except Exception as e:
            print(e)
            raise RuntimeError(f"Failed to instantiate directive: {str(e)}")

        # Update the node's message history with the conversation from directive expansion
        # and the conversation from directive.instantiate
        node.message_history.extend(message_history[len(node.message_history):])
        
        children = []
        for new_ops in rewrites:
            child = self.instantiate_node(
                node, new_ops, directive_name, target_op_list, optimize_goal
            )
            children.append(child)
        return children

    def simulate(self, node: Node):
        """
        Simulate a node (plan). Execute the plan and add it to the pareto frontier.

        Args:
            node: Node to start simulation from

        Returns:
            The updated nodes by the change of the pareto frontier
        """

        try:
            node.execute_plan()
        except Exception as e:
            print(f"Failed to execute plan for node {node.get_id()}: {str(e)}")
            # Set cost to -1 to indicate failure (this is already done in Node.execute_plan)
            # Do not add failed plans to the frontier
            return {}, False

        # Only increment action count after successful execution
        if hasattr(node, 'latest_action') and node.latest_action is not None:
            self.action_counts[node.latest_action] += 1

        affected_nodes, is_frontier_updated = self.pareto_frontier.add_plan_f1(node)
        self.action_rewards = self.pareto_frontier.action_rewards
        return affected_nodes, is_frontier_updated

    def backpropagate(self, affected_nodes: Dict[Node, int], visit_node):
        """
        Backpropagate the simulation value change up the tree.
        """

        for node, val in affected_nodes.items():
            current = node
            while current is not None:
                current.update_value(val)
                current = current.parent

        current = visit_node
        while current is not None:
            current.update_visit()
            current = current.parent

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
        if hasattr(node, 'latest_action') and node.latest_action is not None:
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
            
            # Add action reward statistics
            f.write(f"Action Performance Statistics:\n")
            for action in self.available_actions:
                reward = self.action_rewards.get(action, 0)
                count = self.action_counts.get(action, 0)
                avg_reward = reward / count if count > 0 else "Unknown (never tried)"
                f.write(f"- {action.name}: {count} uses, avg reward: {avg_reward}\n")
            f.write(f"\n")
            
            # Add tree structure
            f.write(f"Tree Structure:\n")
            self.print_tree_visits_and_values(file_handle=f)
            f.write(f"\n")

    def instantiate_node(
        self, node, new_ops_list, directive_name, target_op_list, optimize_goal
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
        new_parsed_yaml = self.update_pipeline(
            new_parsed_yaml, new_ops_list, target_op_list
        )

        self.fix_models_azure(new_parsed_yaml)

        # Determine where to save the new pipeline file
        if self.output_dir:
            # Use output directory with original filename as base
            original_filename = os.path.basename(node.yaml_file_path).removesuffix(
                ".yaml"
            )
            base_path = os.path.join(self.output_dir, original_filename)
            os.makedirs(self.output_dir, exist_ok=True)
        else:
            # Use same directory as original pipeline
            base_path = node.yaml_file_path.removesuffix(".yaml")

        new_yaml_path = f"{base_path}_{len(node.children)+1}.yaml"
        new_parsed_yaml["pipeline"]["output"][
            "path"
        ] = f"{base_path}_{len(node.children)+1}.json"

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
        child = Node(yaml_file_path=new_yaml_path, parent=node)
        action = self.directive_name_to_obj.get(directive_name)
        child.latest_action = action
        print("last action: ", child.latest_action.name, child.id)
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

        node.add_child(child)
        return child
