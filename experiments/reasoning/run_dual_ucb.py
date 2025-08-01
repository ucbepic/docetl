#!/usr/bin/env python3
"""
Dual UCB Bandit Experiment Runner

This script implements a dual UCB bandit approach for DocETL pipeline optimization:
- Outer bandit: selects which pipeline to rewrite (arms = pipelines)
- Inner bandit: selects rewrite directive (arms = rewrite directives)  
- Different rewards: outer uses pareto frontier hypervolume, inner uses pipeline improvement
"""

import os
import json
import argparse
import time
import math
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Set
from pydantic import BaseModel

from docetl.mcts import Node, ParetoFrontier, AccuracyComparator
from docetl.reasoning_optimizer.directives import (
    DEFAULT_MODEL, DEFAULT_OUTPUT_DIR, ALL_DIRECTIVES, instantiate_directive, get_all_directive_strings
)
from docetl.reasoning_optimizer.op_descriptions import *
import litellm
from experiments.reasoning.evaluation.cuad import evaluate_results as cuad_evaluate


class DirectiveApplicabilityResponse(BaseModel):
    """Response format for checking directive applicability"""
    can_apply: bool
    directive: str
    operators: List[str] = []


class UCBBandit:
    """Upper Confidence Bound bandit implementation"""
    
    def __init__(self, arms: List, exploration_constant: float = 1.414):
        self.arms = arms
        self.arm_counts = {arm: 0 for arm in arms}
        self.arm_rewards = {arm: 0.0 for arm in arms}
        self.c = exploration_constant
        self.total_pulls = 0
        
    def select_arm(self):
        """Select arm using UCB algorithm"""
        if self.total_pulls == 0:
            return self.arms[0]
            
        # Calculate UCB for each arm
        ucb_values = {}
        for arm in self.arms:
            if self.arm_counts[arm] == 0:
                # Unvisited arms get infinite UCB value (pure exploration)
                ucb_values[arm] = float('inf')
            else:
                # Standard UCB formula: exploitation + exploration
                exploitation = self.arm_rewards[arm] / self.arm_counts[arm]
                exploration = self.c * math.sqrt(math.log(self.total_pulls) / self.arm_counts[arm])
                ucb_values[arm] = exploitation + exploration
                
        return max(ucb_values, key=ucb_values.get)


class DualUCB:
    """
    Dual UCB implementation for DocETL pipeline optimization.
    
    Uses two bandits:
    - Outer bandit: selects pipelines to rewrite 
    - Inner bandit: selects rewrite directives
    """
    
    def __init__(
        self,
        root_yaml_path: str,
        accuracy_comparator: AccuracyComparator,
        available_actions: Set,
        sample_input,
        exploration_constant: float = 1.414,
        max_iterations: int = 100,
        model: str = DEFAULT_MODEL,
        output_dir: str = None,
    ):
        """Initialize dual UCB algorithm"""
        self.accuracy_comparator = accuracy_comparator
        self.available_actions = available_actions
        self.sample_input = sample_input
        self.exploration_constant = exploration_constant
        self.max_iterations = max_iterations
        self.model = model
        self.output_dir = output_dir
        
        # Initialize root node and execute it
        self.root = Node(root_yaml_path, c=exploration_constant)
        self.root.execute_plan()
        
        # Track all executed pipelines
        self.all_pipelines = [self.root]
        self.pipeline_id_to_node = {self.root.get_id(): self.root}
        
        # Initialize Pareto frontier  
        self.pareto_frontier = ParetoFrontier(accuracy_comparator, {})
        self.pareto_frontier.add_plan_f1(self.root)
        
        # Track min/max values for normalization - initialize before first use
        self.min_accuracy = float('inf')
        self.max_accuracy = float('-inf')
        self.min_cost = float('inf') 
        self.max_cost = float('-inf')
        
        # Initialize normalization bounds after root execution
        self._update_normalization_bounds()
        
        # Initialize outer bandit (pipeline selection)
        self.outer_bandit = UCBBandit([self.root.get_id()], exploration_constant)
        
        # Initialize inner bandit (directive selection)  
        self.inner_bandit = UCBBandit(list(available_actions), exploration_constant)
        
        # Track hypervolume history for outer bandit rewards
        self.hypervolume_history = []
        self.current_hypervolume = self._compute_hypervolume()
        self.hypervolume_history.append(self.current_hypervolume)
        
        # Statistics
        self.iteration_count = 0
        self.start_time = None
        
        # Track directive applications for reward recomputation
        self.directive_applications = {}  # directive -> [(parent_node, child_node), ...]
        
    def _update_normalization_bounds(self):
        """Update min/max bounds for normalization based on all pipelines"""
        for node in self.all_pipelines:
            if node.cost > 0:  # Valid cost
                self.min_cost = min(self.min_cost, node.cost)
                self.max_cost = max(self.max_cost, node.cost)
            
            acc = self.pareto_frontier.plans_accuracy.get(node, 0.0)
            self.min_accuracy = min(self.min_accuracy, acc)
            self.max_accuracy = max(self.max_accuracy, acc)
            
        # Handle edge cases
        if self.min_accuracy == self.max_accuracy:
            self.max_accuracy = self.min_accuracy + 1e-6
        if self.min_cost == self.max_cost:
            self.max_cost = self.min_cost + 1e-6
            
    def _normalize_accuracy(self, acc: float) -> float:
        """Normalize accuracy to [0,1] for scalarized reward computation"""
        if self.max_accuracy == self.min_accuracy:
            return 0.5
        return (acc - self.min_accuracy) / (self.max_accuracy - self.min_accuracy)
    
    def _normalize_cost(self, cost: float) -> float:
        """Normalize cost to [0,1] for scalarized reward computation"""
        if self.max_cost == self.min_cost:
            return 0.5
        return (cost - self.min_cost) / (self.max_cost - self.min_cost)
        
    def _compute_hypervolume(self) -> float:
        """
        Compute hypervolume of current Pareto frontier using proper formula:
        HV(F) = Σ(A_i - A_ref)(C_{i+1} - C_i) where C_{n+1} := C_ref
        Uses normalized accuracy and cost values.
        """
        return self._compute_hypervolume_for_nodes(self.pareto_frontier.frontier_plans)
    
    
    def _recompute_all_rewards(self):
        """
        Recompute all bandit rewards based on current Pareto frontier.
        This is called whenever the frontier changes.
        """
        print("Recomputing all bandit rewards...")
        
        # 1. Recompute outer bandit rewards (hypervolume with/without each pipeline)
        self._recompute_outer_bandit_rewards()
        
        # 2. Recompute inner bandit rewards (scalarized improvement for all directive applications)
        self._recompute_inner_bandit_rewards()
    
    def _recompute_outer_bandit_rewards(self):
        """
        For each pipeline, compute reward as hypervolume improvement when that pipeline is added.
        """
        # Clear existing rewards but keep counts
        for pipeline_id in self.outer_bandit.arms:
            self.outer_bandit.arm_rewards[pipeline_id] = 0.0
        
        # For each pipeline, compute hypervolume with and without it
        for pipeline_node in self.all_pipelines:
            pipeline_id = pipeline_node.get_id()
            if pipeline_id not in self.outer_bandit.arms:
                continue
                
            # Compute hypervolume without this pipeline
            frontier_without = [p for p in self.pareto_frontier.frontier_plans if p != pipeline_node]
            hv_without = self._compute_hypervolume_for_nodes(frontier_without)
            
            # Compute hypervolume with this pipeline (current frontier)
            hv_with = self.current_hypervolume
            
            # Reward is the improvement
            reward = max(0.0, hv_with - hv_without)
            
            # Update the bandit with this reward
            if self.outer_bandit.arm_counts[pipeline_id] > 0:
                # Set the cumulative reward to maintain proper average
                self.outer_bandit.arm_rewards[pipeline_id] = reward * self.outer_bandit.arm_counts[pipeline_id]
            
            print(f"Pipeline {pipeline_id}: HV improvement = {reward:.4f}")
    
    def _recompute_inner_bandit_rewards(self):
        """
        For each directive, recompute rewards for all its (parent, child) pairs using fresh normalization.
        """
        # Clear existing rewards but keep counts
        for directive in self.inner_bandit.arms:
            self.inner_bandit.arm_rewards[directive] = 0.0
        
        # Sample a fresh λ for this recomputation
        import random
        lambda_min, lambda_max = 0.001, 1.0
        lambda_val = math.exp(random.uniform(math.log(lambda_min), math.log(lambda_max)))
        
        print(f"Recomputing inner rewards with λ={lambda_val:.4f}")
        
        # For each directive that has been applied
        for directive, applications in self.directive_applications.items():
            if directive not in self.inner_bandit.arms or not applications:
                continue
            
            total_reward = 0.0
            count = len(applications)
            
            # Compute reward for each (parent, child) pair
            for parent_node, child_node in applications:
                # Get raw accuracy and cost values
                parent_acc = self.pareto_frontier.plans_accuracy.get(parent_node, 0.0)
                child_acc = self.pareto_frontier.plans_accuracy.get(child_node, 0.0)
                parent_cost = parent_node.cost if parent_node.cost > 0 else 1.0
                child_cost = child_node.cost
                
                # Normalize using current bounds
                parent_acc_norm = self._normalize_accuracy(parent_acc)
                child_acc_norm = self._normalize_accuracy(child_acc)
                parent_cost_norm = self._normalize_cost(parent_cost)
                child_cost_norm = self._normalize_cost(child_cost)
                
                # Compute scalarized scores
                parent_score = parent_acc_norm - lambda_val * parent_cost_norm
                child_score = child_acc_norm - lambda_val * child_cost_norm
                
                # Reward is max(0, improvement)
                improvement = child_score - parent_score
                reward = max(0.0, improvement)
                total_reward += reward
            
            # Update bandit with average reward
            avg_reward = total_reward / count if count > 0 else 0.0
            if self.inner_bandit.arm_counts[directive] > 0:
                # Set cumulative reward to maintain proper average
                self.inner_bandit.arm_rewards[directive] = avg_reward * self.inner_bandit.arm_counts[directive]
            
            print(f"Directive {directive.name}: avg reward = {avg_reward:.4f} (from {count} applications)")
    
    def _compute_hypervolume_for_nodes(self, nodes: List[Node]) -> float:
        """Compute hypervolume for a specific set of nodes using normalized values"""
        if not nodes:
            return 0.0
            
        # Get frontier points (normalized accuracy, normalized cost)
        frontier_points = []
        for node in nodes:
            acc = self.pareto_frontier.plans_accuracy.get(node, 0.0)
            cost = node.cost if node.cost > 0 else 0.0
            
            # Apply 0-1 normalization
            norm_acc = self._normalize_accuracy(acc)
            norm_cost = self._normalize_cost(cost)
            
            frontier_points.append((norm_acc, norm_cost))
            
        if not frontier_points:
            return 0.0
            
        # Sort in ascending cost and descending accuracy (as per formula)
        frontier_points.sort(key=lambda p: (p[1], -p[0]))
        
        # Reference point: A_ref = 0, C_ref = max normalized cost on frontier
        A_ref = 0.0
        C_ref = max(point[1] for point in frontier_points) if frontier_points else 0.0
        
        # Compute hypervolume: Σ(A_i - A_ref)(C_{i+1} - C_i)
        hypervolume = 0.0
        for i in range(len(frontier_points)):
            A_i = frontier_points[i][0]
            C_i = frontier_points[i][1]
            C_next = frontier_points[i + 1][1] if i + 1 < len(frontier_points) else C_ref
            
            if A_i > A_ref and C_next > C_i:
                hypervolume += (A_i - A_ref) * (C_next - C_i)
                
        return hypervolume
    
    def search(self) -> List[Node]:
        """Run dual UCB search"""
        self.start_time = time.time()
        self.iteration_count = 0
        
        print(f"Starting Dual UCB search with {self.max_iterations} iterations...")
        print(f"Root node cost: ${self.root.cost:.2f}")
        print(f"Initial hypervolume: {self.current_hypervolume:.4f}")
        
        while self.iteration_count < self.max_iterations:
            self._dual_ucb_iteration()
            self.iteration_count += 1
            
        print(f"\nDual UCB search completed!")
        print(f"Total iterations: {self.iteration_count}")
        print(f"Total pipelines explored: {len(self.all_pipelines)}")
        print(f"Final hypervolume: {self.current_hypervolume:.4f}")
        print(f"Pareto frontier size: {len(self.pareto_frontier.frontier_plans)}")
        
        # Plot final Pareto frontier once at the end
        self.pareto_frontier.plot_plans()
        
        return self.pareto_frontier.frontier_plans
    
    def _dual_ucb_iteration(self):
        """Perform one dual UCB iteration"""
        print(f"\n--- Iteration {self.iteration_count + 1} ---")
        
        # 1. Outer bandit: select pipeline to rewrite
        selected_pipeline_id = self.outer_bandit.select_arm()
        selected_pipeline = self.pipeline_id_to_node[selected_pipeline_id]
        print(f"Outer bandit selected pipeline: {selected_pipeline_id}")
        
        # 2. Inner bandit: keep sampling until we find a directive that works
        selected_directive = None
        target_ops = None
        new_pipeline = None
        max_directive_retries = 10  # Prevent infinite loops
        directive_retry_count = 0
        
        # Track failed directives to exclude from sampling in this iteration
        failed_directives = set()
        failed_attempts = []
        
        while new_pipeline is None and directive_retry_count < max_directive_retries:
            # Sample directive from inner bandit, excluding failed ones
            available_directives = [d for d in self.inner_bandit.arms if d not in failed_directives]
            if not available_directives:
                print("No more directives to try")
                break
                
            # Create temporary bandit with only available directives for sampling
            temp_bandit = UCBBandit(available_directives, self.inner_bandit.c)
            temp_bandit.arm_counts = {d: self.inner_bandit.arm_counts[d] for d in available_directives}
            temp_bandit.arm_rewards = {d: self.inner_bandit.arm_rewards[d] for d in available_directives}
            temp_bandit.total_pulls = self.inner_bandit.total_pulls
            
            candidate_directive = temp_bandit.select_arm()
            print(f"Inner bandit trying directive: {candidate_directive.name}")
            
            # Use agent to check if directive can apply and get target ops
            can_apply, ops, messages = self._check_directive_applicability(selected_pipeline, candidate_directive, failed_attempts)
            
            if can_apply and ops:
                selected_directive = candidate_directive
                target_ops = ops
                print(f"Inner bandit found applicable directive: {selected_directive.name} for ops: {target_ops}")
                
                # 3. Try to apply directive
                try:
                    new_pipeline = self._apply_directive_with_target_ops(selected_pipeline, selected_directive, target_ops, messages)
                    if new_pipeline is not None:
                        # Success! Break out of loop
                        break
                    else:
                        print("Failed to apply directive - instantiation returned None")
                        failed_attempts.append(f"Directive '{selected_directive.name}' with ops {target_ops} failed during instantiation (returned None)")
                        failed_directives.add(selected_directive)
                        directive_retry_count += 1
                        
                except Exception as e:
                    print(f"Error applying directive {selected_directive.name}: {e}")
                    error_msg = str(e)
                    failed_attempts.append(f"Directive '{selected_directive.name}' with ops {target_ops} failed: {error_msg}")
                    failed_directives.add(selected_directive)
                    directive_retry_count += 1
                    
            else:
                print(f"Directive {candidate_directive.name} cannot apply, trying another...")
                failed_attempts.append(f"Directive '{candidate_directive.name}' is not applicable to this pipeline")
                failed_directives.add(candidate_directive)
                directive_retry_count += 1
        
        if new_pipeline is None:
            print(f"Failed to find working directive after {max_directive_retries} attempts")
            print(f"Failed attempts: {failed_attempts}")
            return
                
        # Execute the new pipeline
        try:
            new_pipeline.execute_plan()
        except Exception as e:
            print(f"Error executing pipeline: {e}")
            return
            
        if new_pipeline.cost <= 0:
            print("Pipeline execution failed - no rewards given")
            return
            
        # Add to our pipeline collection
        self.all_pipelines.append(new_pipeline)
        self.pipeline_id_to_node[new_pipeline.get_id()] = new_pipeline
        
        # Add new pipeline as arm to outer bandit
        self.outer_bandit.arms.append(new_pipeline.get_id())
        self.outer_bandit.arm_counts[new_pipeline.get_id()] = 0
        self.outer_bandit.arm_rewards[new_pipeline.get_id()] = 0.0
        
        print(f"New pipeline created: {new_pipeline.get_id()}")
        print(f"Cost: ${new_pipeline.cost:.2f}")
        
        # 4. Update Pareto frontier and compute rewards
        old_hypervolume = self.current_hypervolume
        
        # Add new pipeline to frontier
        affected_nodes, is_frontier_updated = self.pareto_frontier.add_plan_f1(new_pipeline)
        
        # Update normalization bounds with new pipeline
        self._update_normalization_bounds()
        
        # Compute new hypervolume (using updated normalization)
        self.current_hypervolume = self._compute_hypervolume()
        self.hypervolume_history.append(self.current_hypervolume)
        
        # 5. Update bandit visits (required for UCB calculation)
        self.outer_bandit.arm_counts[selected_pipeline_id] += 1
        self.outer_bandit.total_pulls += 1
        self.inner_bandit.arm_counts[selected_directive] += 1
        self.inner_bandit.total_pulls += 1
        
        # Track the directive application for future reward recomputation
        if selected_directive not in self.directive_applications:
            self.directive_applications[selected_directive] = []
        self.directive_applications[selected_directive].append((selected_pipeline, new_pipeline))
        
        # Recompute ALL bandit rewards based on new frontier
        self._recompute_all_rewards()
        
        hypervolume_improvement = self.current_hypervolume - old_hypervolume
        print(f"Hypervolume: {old_hypervolume:.4f} -> {self.current_hypervolume:.4f} (Δ: {hypervolume_improvement:.4f})")
        print(f"Frontier updated: {is_frontier_updated}")
        
        # Print bandit statistics
        print(f"Outer bandit stats: {len(self.outer_bandit.arms)} arms, {self.outer_bandit.total_pulls} pulls")
        print(f"Inner bandit stats: {len(self.inner_bandit.arms)} arms, {self.inner_bandit.total_pulls} pulls")
        

    
    def _check_directive_applicability(self, pipeline: Node, directive, failed_attempts: List[str] = None) -> tuple[bool, List[str]]:
        """
        Use the agent to check if a directive can apply to the pipeline.
        Returns (can_apply, target_ops) where target_ops is empty if can_apply is False.
        """
        import yaml
        
        # Load the pipeline YAML config
        with open(pipeline.yaml_file_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Get pipeline operations
        operations = config.get('operations', [])
        if not operations:
            return False, []
            
        # Input schema (like MCTS)
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

        # Add context about previous failures if any
        failure_context = ""
        if failed_attempts:
            failure_context = f"""
        
        Previous failed attempts in this iteration:
        {chr(10).join(failed_attempts)}
        
        Please avoid suggesting combinations that have already failed.
        """

        # Create applicability check prompt with full context like MCTS
        user_message = f"""
        Check if the rewrite directive '{directive.name}' can be applied to this pipeline.
        If it is applicable, select the operator(s) that should be targeted by the rewrite. 
        If the directive requires exactly one target operator, select the single operator that is most amenable to this rewrite (i.e., the one that would benefit most or is best suited for the directive).
        Some operators (e.g., fusion) require 2 operators.
        If the directive allows multiple target operators, list all that are appropriate.
        {failure_context}
        
        Pipeline:
        Pipelines in DocETL are the core structures that define the flow of data processing. A pipeline consists of five main components:
        - Default Model: The language model to use for the pipeline. Limit your choice of model to gpt-4.1-nano, gpt-4o-mini, gpt-4o, gpt-4.1
        - System Prompts: A description of your dataset and the "persona" you'd like the LLM to adopt when analyzing your data.
        - Datasets: The input data sources for your pipeline.
        - Operators: The processing steps that transform your data.
        - Pipeline Specification: The sequence of steps and the output configuration.

        Operators:
        Operators form the building blocks of data processing pipelines. Below is the list of operators:
        {op_map.to_string()}
        {op_extract.to_string()}
        {op_parallel_map.to_string()}
        {op_filter.to_string()}
        {op_reduce.to_string()}
        {op_split.to_string()}
        {op_gather.to_string()}
        {op_unnest.to_string()}
        {op_sample.to_string()}
        {op_resolve.to_string()}

        Rewrite directives:
        {get_all_directive_strings()}

        Directive to check: {directive.name}
        
        Input document schema with token statistics: {input_schema}
        Input data sample: {json.dumps(self.sample_input, indent=2)[:500000]}
        
        Pipeline operations:
        {json.dumps(operations, indent=2)}
        
        Please respond with a JSON object:
        {{
            "can_apply": true/false,
            "directive": "{directive.name}",
            "operators": ["list", "of", "target", "operator", "names"]
        }}
        
        If can_apply is false, leave operators as an empty list.
        If can_apply is true, list the specific operator names from the pipeline where this directive should be applied.
        Only include operator names that actually exist in the pipeline operations above.
        """
        
        messages = [
            {
                "role": "system", 
                "content": "You are an expert query optimization agent. Check if a rewrite directive can be applied to a pipeline and identify target operators. Respond with structured JSON."
            },
            {"role": "user", "content": user_message}
        ]
        
        try:
            response = litellm.completion(
                model=self.model,
                messages=messages,
                response_format=DirectiveApplicabilityResponse,
            )
            reply = response.choices[0].message.content
            
            # Parse response
            parsed = json.loads(reply)
            can_apply = parsed.get("can_apply", False)
            target_ops = parsed.get("operators", [])
            
            print(f"Applicability check for {directive.name}: can_apply={can_apply}, target_ops={target_ops}")
            
            # Add the agent response to messages for continuity
            messages.append({"role": "assistant", "content": reply})
            
            return can_apply, target_ops, messages
            
        except Exception as e:
            print(f"Error checking directive applicability: {e}")
            return False, [], []
    
    def _apply_directive_with_target_ops(self, pipeline: Node, directive, target_ops: List[str], message_history: List[dict]) -> Optional[Node]:
        """Apply a rewrite directive to a pipeline using pre-determined target ops"""
        import yaml
        import json
        
        # Load the pipeline YAML config
        with open(pipeline.yaml_file_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Get pipeline operations
        operations = config.get('operations', [])
        if not operations:
            print("No operations found in pipeline")
            return None
            
        try:
            # Apply the directive using the proper instantiate function with the message history
            new_ops_list, updated_message_history = instantiate_directive(
                directive_name=directive.name,
                operators=operations,
                target_ops=target_ops,
                agent_llm=self.model,
                message_history=message_history,  # Use the message history from applicability check
                global_default_model=self.model,
            )
            
            if not new_ops_list:
                print(f"Failed to instantiate directive {directive.name}")
                return None
                
            # Create new config following MCTS pattern
            from copy import deepcopy
            new_config = deepcopy(config)
            new_config["operations"] = new_ops_list
            new_config["bypass_cache"] = True
            
            # Update pipeline steps if they exist (like MCTS update_pipeline method)
            if new_ops_list is not None:
                op_names = [op.get("name") for op in new_ops_list if "name" in op]

                # Update the pipeline steps to use the new operation names
                if "pipeline" in new_config and "steps" in new_config["pipeline"]:
                    for step in new_config["pipeline"]["steps"]:
                        if "operations" in step:
                            step["operations"] = op_names
            
            # Generate unique filename in experiment output directory
            timestamp = int(time.time() * 1000)
            pipeline_base_name = Path(pipeline.yaml_file_path).stem
            new_filename = f"{pipeline_base_name}_{timestamp}_{directive.name}.yaml"
            
            # Save to experiment output directory
            if self.output_dir:
                new_path = Path(self.output_dir) / new_filename
                output_json_path = f"{pipeline_base_name}_{timestamp}_{directive.name}.json"
            else:
                # Fallback to same directory as original pipeline
                base_path = pipeline.yaml_file_path.removesuffix(".yaml")
                new_path = Path(f"{base_path}_{timestamp}_{directive.name}.yaml")
                output_json_path = f"{Path(base_path).name}_{timestamp}_{directive.name}.json"
            
            # Set unique output path (like MCTS)
            if "pipeline" in new_config and "output" in new_config["pipeline"]:
                new_config["pipeline"]["output"]["path"] = output_json_path
            
            # Save new config
            with open(new_path, 'w') as f:
                yaml.dump(new_config, f, default_flow_style=False)
                
            # Create new node
            new_node = Node(str(new_path), c=self.exploration_constant)
            new_node.parent = pipeline
            new_node.action_taken = directive.name
            
            print(f"Successfully applied directive {directive.name} to operations {target_ops}")
            return new_node
            
        except Exception as e:
            print(f"Error applying directive {directive.name}: {e}")
            import traceback
            traceback.print_exc()
            return None


def run_dual_ucb_experiment(
    yaml_path: str,
    data_dir: str = None,
    output_dir: str = None,
    experiment_name: str = "dual_ucb_experiment", 
    max_iterations: int = 100,
    exploration_weight: float = 1.414,
    model: str = DEFAULT_MODEL,
    dataset: str = "cuad",
    ground_truth_path: str | None = None,
):
    """Run dual UCB optimization experiment"""
    
    # Set up environment
    if data_dir:
        os.environ['EXPERIMENT_DATA_DIR'] = data_dir
    
    # Determine output directory
    if output_dir is None:
        output_dir = os.environ.get('EXPERIMENT_OUTPUT_DIR', DEFAULT_OUTPUT_DIR)
    
    # Create output directory
    output_path = Path(output_dir) / experiment_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"🎯 Running Dual UCB Optimization Experiment")
    print(f"=" * 50)
    print(f"Input Pipeline: {yaml_path}")
    print(f"Data Directory: {data_dir or 'default'}")
    print(f"Output Directory: {output_path}")
    print(f"Max Iterations: {max_iterations}")
    print(f"Exploration Weight (c): {exploration_weight}")
    print(f"Model: {model}")
    print(f"Experiment: {experiment_name}")
    print(f"Dataset for evaluation: {dataset}")
    print()
    
    # Initialize accuracy comparator
    sample_input_data = {"document": "Sample contract document for comparison purposes..."}
    accuracy_comparator = AccuracyComparator(input_data=sample_input_data, model=model)
    
    # Use all registered rewrite directives
    available_actions = set(ALL_DIRECTIVES)
    
    # Initialize Dual UCB
    dual_ucb = DualUCB(
        root_yaml_path=yaml_path,
        accuracy_comparator=accuracy_comparator,
        available_actions=available_actions,
        sample_input=sample_input_data,
        exploration_constant=exploration_weight,
        max_iterations=max_iterations,
        model=model,
        output_dir=str(output_path),
    )
    
    print(f"✅ Dual UCB initialized with root pipeline: {yaml_path}")
    
    # Run optimization
    print(f"\n🔍 Running Dual UCB optimization for {max_iterations} iterations...")
    start_time = datetime.now()
    
    best_nodes = dual_ucb.search()
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print(f"✅ Dual UCB optimization completed in {duration:.2f} seconds")
    
    # Evaluation (same as MCTS)
    eval_results = []
    pareto_auc = None
    
    if dataset.lower() == "cuad":
        if ground_truth_path is None:
            default_gt = Path("experiments/reasoning/data/CUAD-master_clauses.csv")
            ground_truth_path = str(default_gt)

        print(f"\n🧪 Evaluating extraction JSONs against CUAD ground truth ...")

        for n in dual_ucb.pareto_frontier.plans:
            jf = n.result_path
            if jf is None or not Path(jf).exists():
                continue
            try:
                metrics = cuad_evaluate("dual_ucb", jf, ground_truth_path)
                jp = Path(jf).resolve()
                op_root = output_path.resolve()
                if hasattr(jp, "is_relative_to") and jp.is_relative_to(op_root):
                    display_path = str(jp.relative_to(op_root))
                else:
                    display_path = jp.name

                eval_results.append({
                    "file": display_path,
                    "node_id": n.get_id(),
                    "precision": metrics["avg_precision"],
                    "recall": metrics["avg_recall"],
                    "f1": metrics["avg_f1"],
                    "cost": n.cost,
                    "accuracy": dual_ucb.pareto_frontier.plans_accuracy.get(n),
                    "on_frontier": n in dual_ucb.pareto_frontier.frontier_plans,
                 })
            except Exception as e:
                print(f"   ⚠️  Evaluation failed for {jf}: {e}")

        if eval_results:
            eval_out_file = output_path / "evaluation_metrics.json"
            with open(eval_out_file, "w") as f:
                json.dump(eval_results, f, indent=2)
            print(f"📊 Evaluation results written to {eval_out_file}")

            # Compute Pareto AUC
            try:
                frontier_points = [row for row in eval_results if row["on_frontier"]]
                if len(frontier_points) >= 2:
                    frontier_points.sort(key=lambda r: r["cost"])
                    pareto_auc = 0.0
                    prev_point = frontier_points[0]
                    for curr_point in frontier_points[1:]:
                        width = curr_point["cost"] - prev_point["cost"]
                        if width > 0:
                            pareto_auc += 0.5 * width * (prev_point["f1"] + curr_point["f1"])
                        prev_point = curr_point
                elif frontier_points:
                    pareto_auc = 0.0

                if pareto_auc is not None:
                    print(f"📐 Area under Pareto frontier (Cost vs F1): {pareto_auc:.4f}")
            except Exception as e:
                print(f"⚠️  Failed to compute Pareto AUC: {e}")
    
    # Save results
    results = {
        "experiment_name": experiment_name,
        "algorithm": "dual_ucb",
        "input_pipeline": yaml_path,
        "model": model,
        "max_iterations": max_iterations,
        "exploration_weight": exploration_weight,
        "data_dir": data_dir,
        "output_dir": str(output_path),
        "dataset": dataset,
        "ground_truth": ground_truth_path,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "duration_seconds": duration,
        "num_best_nodes": len(best_nodes) if best_nodes else 0,
        "total_pipelines_explored": len(dual_ucb.all_pipelines),
        "final_hypervolume": dual_ucb.current_hypervolume,
        "hypervolume_history": dual_ucb.hypervolume_history,
    }
    
    if pareto_auc is not None:
        results["pareto_auc"] = pareto_auc
    if eval_results:
        results["evaluation_file"] = str(eval_out_file)
    
    # Save Pareto frontier
    if dual_ucb.pareto_frontier.frontier_plans:
        pareto_file = output_path / "pareto_frontier.json"
        pareto_data = []
        
        for solution in dual_ucb.pareto_frontier.frontier_plans:
            pareto_data.append({
                "accuracy": dual_ucb.pareto_frontier.plans_accuracy.get(solution, None),
                "cost": getattr(solution, 'cost', None),
                "config_path": getattr(solution, 'yaml_file_path', None)
            })
        
        with open(pareto_file, 'w') as f:
            json.dump(pareto_data, f, indent=2)
        
        results["pareto_frontier_file"] = str(pareto_file)
        print(f"📈 Pareto frontier saved to: {pareto_file}")
    
    # Save bandit statistics
    bandit_stats = {
        "outer_bandit": {
            "arms": dual_ucb.outer_bandit.arms,
            "arm_counts": dual_ucb.outer_bandit.arm_counts,
            "arm_rewards": dual_ucb.outer_bandit.arm_rewards,
            "total_pulls": dual_ucb.outer_bandit.total_pulls,
        },
        "inner_bandit": {
            "arms": [action.name for action in dual_ucb.inner_bandit.arms],
            "arm_counts": {action.name: dual_ucb.inner_bandit.arm_counts[action] for action in dual_ucb.inner_bandit.arms},
            "arm_rewards": {action.name: dual_ucb.inner_bandit.arm_rewards[action] for action in dual_ucb.inner_bandit.arms},
            "total_pulls": dual_ucb.inner_bandit.total_pulls,
        }
    }
    
    bandit_file = output_path / "bandit_statistics.json"
    with open(bandit_file, 'w') as f:
        json.dump(bandit_stats, f, indent=2)
    results["bandit_statistics_file"] = str(bandit_file)
    
    # Save experiment summary
    summary_file = output_path / "experiment_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📋 Experiment Summary:")
    print(f"   Duration: {duration:.2f} seconds")
    print(f"   Pipelines Explored: {results['total_pipelines_explored']}")
    print(f"   Best Configs Found: {results['num_best_nodes']}")
    print(f"   Final Hypervolume: {dual_ucb.current_hypervolume:.4f}")
    print(f"   Summary saved to: {summary_file}")
    print(f"   All outputs in: {output_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run Dual UCB reasoning optimization experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic Dual UCB run
  python run_dual_ucb.py --yaml_path ./pipeline.yaml --experiment_name dual_ucb_test
  
  # With custom parameters
  python run_dual_ucb.py --yaml_path ./pipeline.yaml --data_dir ./data --output_dir ./results --max_iterations 200 --experiment_name dual_ucb_deep
  
  # High exploration
  python run_dual_ucb.py --yaml_path ./pipeline.yaml --exploration_weight 2.0 --experiment_name dual_ucb_explore
        """
    )
    
    parser.add_argument("--yaml_path", type=str, required=True,
                       help="Path to the input YAML pipeline file")
    parser.add_argument("--data_dir", type=str,
                       help="Directory containing input data files (sets EXPERIMENT_DATA_DIR)")
    parser.add_argument("--output_dir", type=str,
                       help=f"Directory to save experiment outputs (default: EXPERIMENT_OUTPUT_DIR env var or {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--experiment_name", type=str, required=True,
                       help="Name for this experiment run")
    parser.add_argument("--max_iterations", type=int, default=100,
                       help="Maximum iterations (default: 100)")
    parser.add_argument("--exploration_weight", type=float, default=1.414,
                       help="UCB exploration parameter c (default: 1.414)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                       help=f"LLM model to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--dataset", type=str, default="cuad", help="Dataset name for evaluation (default: cuad)")
    parser.add_argument("--ground_truth", type=str, help="Path to ground-truth file (if not default)")
    
    args = parser.parse_args()
    
    result = run_dual_ucb_experiment(
        yaml_path=args.yaml_path,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
        max_iterations=args.max_iterations,
        exploration_weight=args.exploration_weight,
        model=args.model,
        dataset=args.dataset,
        ground_truth_path=args.ground_truth,
    )
    
    print("\n🎉 Dual UCB experiment completed successfully!")


if __name__ == "__main__":
    main()
