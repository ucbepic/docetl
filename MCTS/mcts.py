from ast import Set
import math
import random
import time
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
import os
import yaml
import litellm
from copy import deepcopy
from Node import Node
from ParetoFrontier import ParetoFrontier
from acc_comparator import AccuracyComparator
from docetl.reasoning_optimizer.directive import Directive
from docetl.reasoning_optimizer.ChainingDirective import *
from docetl.reasoning_optimizer.GleaningDirective import *
from docetl.reasoning_optimizer.ChangeModelDirective import *
from docetl.reasoning_optimizer.op_descriptions import *


class ExpandResponseFormat(BaseModel):
    directive: str
    operator: str

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
        exploration_constant: float = 1.414,
        max_iterations: int = 20,
        max_time: Optional[float] = 600.0,
        expansion_count: int = 3,
        model = "gpt-4.1"
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
        """
        self.root = Node(root_yaml_path, c=exploration_constant)
        self.available_actions = available_actions
        self.exploration_constant = exploration_constant
        self.max_iterations = max_iterations
        self.max_time = max_time
        self.iteration_count = 0
        self.expansion_count = expansion_count
        self.start_time = None
        self.model = model
        self.sample_input = sample_input
        # Initialize Pareto frontier
        self.pareto_frontier = ParetoFrontier(accuracy_comparator)
        self.directive_name_to_obj = {action.name: action for action in self.available_actions}
        
        # execute root node and add it to the pareto frontier
        _ = self.simulate(self.root)


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
            self.mcts_iteration()
            self.iteration_count += 1
            
        # Final statistics
        print(f"\nMCTS search completed!")
        print(f"Total iterations: {self.iteration_count}")
        print(f"Pareto frontier size: {len(self.pareto_frontier)}")
        print(f"Frontier plans: {len(self.pareto_frontier.frontier_plans)}")
        self.pareto_frontier.plot_plans()
        print("pareto frontier yaml files: ")
        for plan in self.pareto_frontier.frontier_plans:
            print(plan.yaml_file_path)
        
        # Return all frontier plans
        frontier_plans = [summary['node'] for summary in self.pareto_frontier.get_all_plans_summary() 
                         if summary['is_frontier']]
        
        return frontier_plans
    
    def mcts_iteration(self):
        """Perform one complete MCTS iteration."""
        # 1. Selection: Find the best leaf node
        print("SELECTION")
        leaf = self.select(self.root)
        print("SELECTED NODE: ", leaf.get_id())
        
        # 2. Expansion: Always attempt to expand the leaf, catch errors
        print("EXPANSION")
        try:
            leaf = self.expand(leaf)
        except RuntimeError as e:
            print("OOOOOPS")
            return
        
        # 3. Simulation: Run simulations from the leaf
        print("SIMULATION")
       
        affected_nodes = self.simulate(leaf)
        
        # 4. Backpropagation: Update values up the tree
        print("BACKPROP")
        self.backpropagate(affected_nodes, leaf)

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
 
    def is_fully_explored(self, node:Node) -> bool:
        if len(node.children) >= self.expansion_count: return True
        return False

    def expansion_prompt(self, action_options, input_query) -> str:
        
        availabel_actions_str = ""
        for item in action_options:
            op_name = item[0]
            action_name = item[1]
            action_str = f"Operator: {op_name}, Rewrite directive: {action_name}\n"
            availabel_actions_str += action_str
        
        print(availabel_actions_str)

        input_schema = """
        Dataset: contracts_data
        Type: file
        Records loaded: 50
        Input schema:
            document: string (avg: 10993.9 tokens)
            id: string (avg: 22.9 tokens)
            name: string (avg: 27.6 tokens)
        Total tokens: 546,693
        """

        user_message = f"""
        I have a set of operations used to process long documents, along with a list of possible rewrite directives aimed at improving the quality or cost effectiveness of the query result.
        Given a query pipeline made up of these operations, recommend one specific rewrite directive (specify by its name) that would improve accuracy or cost effectiveness and pecify which single operator (specify by the name) in the pipeline the directive should be applied to.
        Make sure that your cosen directive is in the provided list of rewrite directives.
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
        {ChainingDirective().to_string_for_plan()}\n
        {GleaningDirective().to_string_for_plan()}\n
        {ChangeModelDirective().to_string_for_plan()}\n

        Your valid choice of operation and rewrite directive combination. Only choose one of these:
        {availabel_actions_str}

        Input document schema with token statistics: {input_schema} \n
        Input data sample: {json.dumps(self.sample_input, indent=2)[:5000]} \n
        The original query in YAML format using our operations: {input_query} \n
        """
        return user_message
    
    def expand(self, node: Node) -> Node:
        """
        Expand a leaf node by adding one new child and return the child. 
        
        Args:
            node: Leaf node to expand
            
        Returns:
            The newly created child
        """

        print("INSIDE EXPAND")
        print(node.get_id())
        
        op_list = list(node.op_dict.keys())
        action_options = [] # a list of tuple
        for op_name in op_list:
            if op_name in node.used_actions: 
                used_actions = node.used_actions[op_name]
            else: used_actions = set()
            print(op_name, " ***** ", used_actions)
            action_space = set(self.available_actions) - used_actions # The actions that are not used on this operator 
            for action in action_space:
                action_options.append((op_name, action.name))
        print(action_options)
        if len(action_options) < 1:
            print("NO ACTION FOUND")
            raise RuntimeError("No applicable action found for expansion. Action space may be exhausted or all actions are inapplicable.")
        
        user_message = self.expansion_prompt(action_options = action_options, input_query=node.parsed_yaml)
        messages = [
            {"role": "system", "content": "You are an expert query optimization agent for document processing pipelines. Your role is to analyze user queries and apply rewrite directives to create more accurate and cost effective execution plans. Your output must follow the structured output format."},
            {"role": "user", "content": user_message}
        ]

        response = litellm.completion(
            model=self.model,
            messages=messages,
            api_key=os.environ.get("AZURE_API_KEY"),
            api_base=os.environ.get("AZURE_API_BASE"),
            api_version=os.environ.get("AZURE_API_VERSION"),
            azure=True,
            response_format=ExpandResponseFormat
        )
        reply = response.choices[0].message.content

        try:
            parsed = json.loads(reply)
            directive_name = parsed.get("directive")
            target_op = parsed.get("operator")
            print(f"Directive: {directive_name}, Target ops: {target_op}")
        except Exception as e:
            print(f"Failed to parse agent response: {e}")

        # mark action used
        directive = self.directive_name_to_obj.get(directive_name)
        if directive is None:
            raise ValueError(f"Unknown directive name: {directive_name}")
        node.mark_action_used(target_op, directive)

        new_ops_list, message_history = directive.instantiate(operators = node.parsed_yaml["operations"], target_ops = [target_op], agent_llm = self.model)
        new_parsed_yaml = deepcopy(node.parsed_yaml)
        new_parsed_yaml["operations"] = new_ops_list
        new_parsed_yaml["bypass_cache"] = True
        new_parsed_yaml = self.update_pipeline(new_parsed_yaml, new_ops_list, [target_op])

        self.fix_models_azure(new_parsed_yaml)
        base_path = node.yaml_file_path.removesuffix('.yaml')
        new_yaml_path = f"{base_path}_{len(node.children)+1}.yaml"
        new_parsed_yaml["pipeline"]["output"]["path"] = f"{base_path}_{len(node.children)+1}.json"

        with open(new_yaml_path, 'w') as file:
            yaml.dump(new_parsed_yaml, file, default_flow_style=False, allow_unicode=True, sort_keys=False)

        print("NEW YAML FILE: ", new_yaml_path)
    
        # generate the child node
        child = Node(yaml_file_path=new_yaml_path, parent=node)
        node.add_child(child)

        # Return the child 
        return child
    
    def simulate(self, node: Node):
        """
        Simulate a node (plan). Execute the plan and add it to the pareto frontier.
        
        Args:
            node: Node to start simulation from
            
        Returns:
            The updated nodes by the change of the pareto frontier
        """
        
        node.execute_plan()
        affected_nodes = self.pareto_frontier.add_plan(node)
        return affected_nodes 

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
        
        # if self.max_time and self.start_time and time.time() - self.start_time >= self.max_time:
        #     return False
        
        return True
    
    
    def get_frontier_summary(self) -> List[Dict[str, Any]]:
        """Get summary of all plans in the Pareto frontier."""
        return self.pareto_frontier.get_all_plans_summary()
    
    
if __name__ == "__main__":

    # Example usage
    print("MCTS with Pareto Frontier Integration")
    print("This module provides MCTS optimization with multi-objective search.")
    print("Use run_mcts_optimization() to start optimization.") 

    user_query_yaml_path = "/Users/lindseywei/Documents/DocETL-optimizer/reasoning-optimizer/MCTS/execute_res/CUAD-map.yaml"
    with open('/Users/lindseywei/Documents/DocETL-optimizer/reasoning-optimizer/CUAD_random_sample.json', 'r') as f:
        sample_data = json.load(f)

    ac = AccuracyComparator(sample_data)

    action_chaining = ChainingDirective()
    action_gleaning = GleaningDirective()
    action_change_model = ChangeModelDirective()
    actions = set()
    actions.add(action_chaining)
    actions.add(action_gleaning)
    actions.add(action_change_model)

    mcts = MCTS(root_yaml_path=user_query_yaml_path, accuracy_comparator=ac, available_actions=actions, sample_input = sample_data)
    mcts.search()
