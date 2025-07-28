from __future__ import annotations
from platform import node
import yaml
import math
from typing import Optional, List, Dict, Any
import os
from dotenv import load_dotenv
from docetl.runner import DSLRunner
from docetl.reasoning_optimizer.directives import Directive

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

    def __init__(self, yaml_file_path: str, parent: Optional[Node] = None, c: float = 1.414):
        """
        Initialize a Node with YAML file information.
        
        Args:
            yaml_file_path: Path to the YAML configuration file
            parent: Parent node in the search tree
            c: Exploration constant for UCB calculation (default: sqrt(2))
        """
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
        self.used_actions_acc = {}
        self.used_actions_cost = {}

        self.op_dict = {} # Dict: op_name -> op
        for op in self.parsed_yaml["operations"]:
            op_name = op["name"]
            self.op_dict[op_name] = op
            self.used_actions_acc[op_name] = set()
            self.used_actions_cost[op_name] = set()
        self.visits = 0
        self.value = -math.inf
        self.parent = parent
        self.children = []
        self.c = c  # Exploration constant for UCB
        self.cost = -1.0
        self.sample_result = []

        
        # Assign a unique ID to this node
        self.id = Node._id_counter
        Node._id_counter += 1


        print("NODE ID: ", self.id)


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

        print("EXECUTING PLAN: ", self.yaml_file_path)
        
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
            self.sample_result = result_data
            return total_cost, result_data
            
        except Exception as e:
            self.cost = -1  # Indicate failure
            raise Exception(f"Failed to execute plan {self.yaml_file_path}: {str(e)}")


    def _load_yaml(self) -> Dict[str, Any]:
        """
        Load and parse the YAML file.
        
        Returns:
            Parsed YAML content as a dictionary
        """
        try:
            with open(self.yaml_file_path, 'r', encoding='utf-8') as file:
                return yaml.safe_load(file)
        except Exception as e:
            print(f"Error loading YAML file {self.yaml_file_path}: {e}")
            return {}

    
    def best_child(self) -> Node:
        """
        Return the child with the highest UCB (Upper Confidence Bound) value.
        
        UCB formula: value/visits + c * sqrt(ln(parent_visits) / visits)
        
        Returns:
            Child node with highest UCB, or None if no children exist
        """
            
        def ucb(child: Node) -> float:
            if child.visits == 0:
                return float('inf')  # Prioritize unvisited children
            exploitation = child.value / child.visits
            exploration = self.c * math.sqrt(math.log(self.visits) / child.visits)
            return exploitation + exploration
        
        # Print visits and value for each child
        for child in self.children:
            print(f"Child {child.yaml_file_path}: visits = {child.visits}, value = {child.value}")
        
        return max(self.children, key=ucb)
    
    def add_child(self, child: Node):
        """
        Add a new child node during tree expansion.
        
        Args:
            yaml_file_path: Path to the YAML file for the new child node
            
        Returns:
            The newly created child node
        """

        self.children.append(child)
    
    def is_leaf(self) -> bool:
        """
        Check if this node is a leaf (has no children).
        
        Returns:
            True if the node has no children, False otherwise
        """
        return len(self.children) == 0
    
    def mark_action_used_acc(self, op_name, action: Directive):
        """
        Mark a rewrite action as used.
        
        Args:
            action: The action identifier to mark as used
        """
        self.used_actions_acc[op_name].add(action)

    def mark_action_used_cost(self, op_name, action: Directive):
        """
        Mark a rewrite action as used.
        
        Args:
            action: The action identifier to mark as used
        """
        self.used_actions_cost[op_name].add(action)
    
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
        print("***", self.id, self.value, value)
        self.value = max(self.value, value)
    
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
            return float('inf')
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
    
    
