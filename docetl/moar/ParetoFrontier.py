import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt

from .Node import Node


class ParetoFrontier:
    """
    Pareto Frontier class for managing cost-accuracy optimization.

    This class maintains a collection of plans, estimates their accuracy through
    pairwise comparisons, constructs and updates the Pareto frontier, and provides
    value calculations for MCTS integration.
    """

    def __init__(
        self,
        action_rewards: Dict[str, float],
        action_cost_changes: Dict[str, float],
        action_accuracy_changes: Dict[str, float],
        dataset_name: str,
        evaluate_func: Callable[[str], Dict[str, Any]],
        console=None,
    ):
        """
        Initialize the Pareto Frontier.

        Args:
            action_rewards: Reference to MCTS action_rewards dictionary
            action_cost_changes: Reference to MCTS action_cost_changes dictionary
            action_accuracy_changes: Reference to MCTS action_accuracy_changes dictionary
            dataset_name: Name of the dataset being optimized (for evaluation and metric selection)
            evaluate_func: Evaluation function (results_file_path: str) -> dict
            console: Console instance for logging (default: None, uses DOCETL_CONSOLE)
        """
        from docetl.console import DOCETL_CONSOLE

        self.console = console if console is not None else DOCETL_CONSOLE
        self.dataset_name = dataset_name
        self.evaluate_func = evaluate_func

        # Dataset-to-primary-metric mapping
        self.dataset_metrics = {
            "cuad": "avg_f1",
            "blackvault": "avg_distinct_locations",
            "game_reviews": "weighted_score",
            "medec": "combined_score",
            "sustainability": "combined_score",
            "biodex": "avg_rp_at_5",  # Optimize for RP@5 as specified
            "facility": "combined_score",
        }

        # Internal state
        self.plans: List[Node] = []
        self.plans_accuracy: Dict[Node, float] = {}
        self.plans_cost: Dict[Node, float] = {}  # Real costs
        self.frontier_plans: List[Node] = []  # List of nodes on frontier
        self.frontier_data: List[List[int]] = (
            []
        )  # List of [acc, real_cost] of nodes on frontier
        self.action_rewards = action_rewards
        self.action_cost_changes = action_cost_changes
        self.action_accuracy_changes = action_accuracy_changes

        # Distance to current Pareto frontier: positive for on-frontier, negative for off-frontier
        self.node_distances: Dict[Node, float] = {}

        # Root plan reference point
        self.root_accuracy: Optional[float] = None
        self.root_cost: Optional[float] = None

    def add_plan(self, node: Node) -> Dict[Node, int]:
        """
        Add a new plan (Node) to the frontier and estimate its accuracy.

        Args:
            node: Node object representing the plan

        Returns:
            Dict containing estimated accuracy, pareto_value, and other metrics
        """
        if node.cost == -1:  # Handle error case
            return {}

        # Store plan information
        self.plans.append(node)
        self.plans_cost[node] = node.cost

        # Estimate accuracy through pairwise comparisons
        if len(self.plans_accuracy) == 0:
            # First plan gets baseline accuracy
            estimated_accuracy = 0.5
        else:
            estimated_accuracy = self.estimate_accuracy_via_comparisons(node)

        self.plans_accuracy[node] = estimated_accuracy

        # Update Pareto frontier
        affected_nodes = self.update_pareto_frontier()
        if node not in affected_nodes:
            affected_nodes[node] = 0
        return affected_nodes

    def add_plan_f1(self, node: Node, accuracy: float) -> Tuple[Dict[Node, int], bool]:
        """
        Add a new plan (Node) to the frontier with pre-evaluated accuracy.

        Args:
            node: Node object representing the plan
            accuracy: Pre-evaluated accuracy score for the node

        Returns:
            Dict containing affected nodes, bool indicating wether the frontier is updated
        """
        if node.cost == -1:  # Handle error case
            self.plans_accuracy[node] = float("-inf")
            return {}, False

        # Store plan information
        self.plans.append(node)
        self.plans_cost[node] = node.cost  # Store real cost
        # Scaled cost will be calculated in update_pareto_frontier_HV

        # Store the pre-evaluated accuracy
        self.plans_accuracy[node] = accuracy

        # Update Pareto frontier
        affected_nodes, is_frontier_updated = self.update_pareto_frontier_HV(node)
        return affected_nodes, is_frontier_updated

    def get_all_plans_summary(self) -> List[Dict[str, Any]]:
        """
        Get summary of all plans with their metrics.

        Returns:
            List of dictionaries containing plan information and metrics
        """
        summaries = []
        for node in self.plans:
            summary = {
                "node": node.get_id(),
                "path": node.yaml_file_path,
                "cost": node.cost,
                "accuracy": self.plans_accuracy[node],
                "value": node.value,
                "is_frontier": node in self.frontier_plans,
            }
            summaries.append(summary)

        return summaries

    # Helper function to project point onto step function frontier
    def project_to_frontier(self, node_acc, node_cost, frontier_data):
        """
        Project point onto the step function formed by frontier.
        For a step function interpretation, the reward is simply the vertical distance
        to the step function (accuracy distance only).
        """
        if not frontier_data:
            return node_acc

        # Sort frontier by cost (ascending)
        frontier_sorted = sorted(frontier_data, key=lambda x: x[1])  # Sort by cost

        # Find the step function accuracy for this cost
        step_function_accuracy = (
            0.0  # Default if cost is lower than all frontier points
        )

        for fp_acc, fp_cost in frontier_sorted:
            if node_cost >= fp_cost:
                # Cost is >= this frontier point's cost, so step function is at this accuracy
                step_function_accuracy = fp_acc
            else:
                # Cost is < this frontier point's cost, so we use the previous step
                break

        # Return the vertical (accuracy) distance to the step function
        vertical_distance = abs(node_acc - step_function_accuracy)
        return vertical_distance

    def _update_action_rewards(self, node: Node, reward: float) -> None:
        """
        Update action rewards and track cost/accuracy changes based on the reward received by a node.
        Updates the cumulative sum for the latest action that led to this node.

        Args:
            node: The node that received the reward
            reward: The reward value to incorporate
        """
        if not node.latest_action or not self.action_rewards:
            return
        action = node.latest_action
        if action in self.action_rewards:
            # Update cumulative reward sum
            self.action_rewards[action] += reward

            # Track cost and accuracy changes
            if (
                node.parent
                and node.parent in self.plans_cost
                and node in self.plans_cost
            ):
                cost_change = self.plans_cost[node] - self.plans_cost[node.parent]
                self.action_cost_changes[action] += cost_change

            if (
                node.parent
                and node.parent in self.plans_accuracy
                and node in self.plans_accuracy
            ):
                accuracy_change = (
                    self.plans_accuracy[node] - self.plans_accuracy[node.parent]
                )
                self.action_accuracy_changes[action] += accuracy_change

    def update_pareto_frontier_HV(self, new_node) -> Tuple[Dict[Node, int], bool]:
        """
        Update the Pareto frontier based on current plans and calculate hyper-volume indicator.
        """

        valid_nodes = [node for node in self.plans if node.cost != -1]
        affected_nodes = {}

        if not valid_nodes:
            self.frontier_plans = []
            self.frontier_data = []
            return affected_nodes, False

        # Save old frontier nodes before updating
        old_frontier_nodes = self.frontier_plans

        # Sort by real cost for frontier calculation
        valid_nodes.sort(key=lambda node: self.plans_cost[node])

        # Reconstruct old frontier data using real costs
        archive_frontier_data = []
        for node in old_frontier_nodes:
            if node in valid_nodes:  # Only include valid nodes
                acc = self.plans_accuracy.get(node, float("-inf"))
                real_cost = self.plans_cost[node]
                archive_frontier_data.append([acc, real_cost])
            else:
                self.console.log(
                    f"[yellow]INVALID NODE:[/yellow] {node.id}, [dim]cost:[/dim] {node.cost}, [dim]in_valid_nodes:[/dim] {node in valid_nodes}"
                )

        frontier = []
        max_accuracy_so_far = float("-inf")

        for node in valid_nodes:
            accuracy = self.plans_accuracy.get(node, 0.0)

            # Plan is on frontier if it has higher accuracy than all lower-cost plans
            if accuracy > max_accuracy_so_far:
                frontier.append(node)
                max_accuracy_so_far = accuracy

        new_frontier_data = []
        for node in frontier:
            acc = self.plans_accuracy.get(node)
            real_cost = self.plans_cost[node]  # Use real cost
            new_frontier_data.append([acc, real_cost])

        # Check if frontier actually changed
        old_frontier_set = set(old_frontier_nodes)
        new_frontier_set = set(frontier)
        frontier_updated = old_frontier_set != new_frontier_set

        # Update affected nodes based on frontier changes
        for node in valid_nodes:
            node_real_cost = self.plans_cost[node]
            node_acc = self.plans_accuracy[node]

            if node in new_frontier_set and node not in old_frontier_set:
                # Newly on frontier - reward based on vertical distance to OLD frontier step function
                node.on_frontier = True
                vertical_distance_to_old = self.project_to_frontier(
                    node_acc, node_real_cost, archive_frontier_data
                )
                affected_nodes[node] = vertical_distance_to_old
                # Update node distances - positive for on frontier
                self.node_distances[node] = vertical_distance_to_old
                # Update action rewards
                self._update_action_rewards(node, vertical_distance_to_old)

            elif (node not in new_frontier_set and node in old_frontier_set) or (
                node.id == new_node.id
            ):
                # Newly off frontier - give negative reward based on vertical distance to NEW frontier step function
                node.on_frontier = False
                vertical_distance = self.project_to_frontier(
                    node_acc, node_real_cost, new_frontier_data
                )
                affected_nodes[node] = -vertical_distance
                # Update node distances - negative for off frontier
                self.node_distances[node] = -vertical_distance
                # Update action rewards
                if node.id == new_node.id:
                    self._update_action_rewards(node, -vertical_distance)
            elif node not in new_frontier_set:
                # stay off frontier nodes - update the reward to be negative vertical distance to the NEW frontier step function
                node.on_frontier = False
                vertical_distance = self.project_to_frontier(
                    node_acc, node_real_cost, new_frontier_data
                )
                old_distance = self.node_distances.get(node, 0)
                distance_diff = -vertical_distance - old_distance
                affected_nodes[node] = distance_diff
                # Update node distances - negative for off frontier
                self.node_distances[node] = -vertical_distance

        self.frontier_plans = frontier
        self.frontier_data = new_frontier_data
        if new_node.id > 0:
            graph_dir = str(new_node.yaml_file_path).rsplit("/", 1)[0] + "/graph/"
            os.makedirs(graph_dir, exist_ok=True)
            save_path = graph_dir + f"plan_{new_node.id}.png"
            self.plot_plans(save_path, new_node.id, str(new_node.yaml_file_path))
        return affected_nodes, frontier_updated

    def plot_plans(self, save_path=None, plan_num=None, yaml_file=None):
        """
        Plot all current plans as dots on a cost vs. accuracy graph, annotating each with its id.
        Frontier plans are blue, non-frontier plans are grey.

        Args:
            save_path: If provided, save the plot to this path instead of showing it
            iteration_num: If provided, include iteration number in the title
        """
        if plt is None:
            raise ImportError(
                "matplotlib is required for plotting. Please install it with 'pip install matplotlib'."
            )

        plt.figure(figsize=(10, 8))

        # Separate frontier and non-frontier plans
        frontier_nodes = [node for node in self.plans if node in self.frontier_plans]
        non_frontier_nodes = [
            node for node in self.plans if node not in self.frontier_plans
        ]

        # Plot non-frontier plans (grey)
        if non_frontier_nodes:
            costs = [self.plans_cost[node] for node in non_frontier_nodes]
            accuracies = [self.plans_accuracy[node] for node in non_frontier_nodes]
            ids = [node.get_id() for node in non_frontier_nodes]
            plt.scatter(costs, accuracies, color="grey", label="Off Frontier")
            for x, y, label in zip(costs, accuracies, ids):
                plt.annotate(
                    str(label),
                    (x, y),
                    textcoords="offset points",
                    xytext=(5, 5),
                    ha="left",
                    fontsize=9,
                    color="grey",
                )

        # Plot frontier plans (blue)
        if frontier_nodes:
            costs = [self.plans_cost[node] for node in frontier_nodes]
            accuracies = [self.plans_accuracy[node] for node in frontier_nodes]
            ids = [node.get_id() for node in frontier_nodes]
            plt.scatter(costs, accuracies, color="blue", label="Frontier")
            for x, y, label in zip(costs, accuracies, ids):
                plt.annotate(
                    str(label),
                    (x, y),
                    textcoords="offset points",
                    xytext=(5, 5),
                    ha="left",
                    fontsize=9,
                    color="blue",
                )

        plt.xlabel("Cost")
        plt.ylabel("Accuracy")

        if plan_num is not None:
            plt.title(f"Plan {plan_num}: {yaml_file}")
        else:
            plt.title("Plans: Cost vs. Accuracy")

        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()

    def __len__(self) -> int:
        """Return number of plans in the frontier."""
        return len(self.plans)

    def __contains__(self, node: Node) -> bool:
        """Check if plan is managed by this frontier."""
        return node in self.plans

    def delete_plan(self, node: Node) -> None:
        """
        Completely delete a node from all ParetoFrontier data structures.
        """
        if node in self.plans:
            self.plans.remove(node)

        accuracy = self.plans_accuracy.pop(node, None)
        cost = self.plans_cost.pop(node, None)

        if node in self.frontier_plans:
            self.frontier_plans.remove(node)
            if accuracy is not None and cost is not None:
                try:
                    self.frontier_data.remove([accuracy, cost])
                except ValueError:
                    pass

        self.node_distances.pop(node, None)
