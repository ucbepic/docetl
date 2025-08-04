# Import evaluation function lookup
import sys
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt

from .acc_comparator import AccuracyComparator
from .Node import Node

sys.path.append("../../experiments/reasoning")
try:
    from experiments.reasoning.evaluation.utils import get_evaluate_func
except ImportError:
    # Fallback import path
    import os

    sys.path.append(
        os.path.join(os.path.dirname(__file__), "../../experiments/reasoning")
    )
    from evaluation.utils import get_evaluate_func


class ParetoFrontier:
    """
    Pareto Frontier class for managing cost-accuracy optimization.

    This class maintains a collection of plans, estimates their accuracy through
    pairwise comparisons, constructs and updates the Pareto frontier, and provides
    value calculations for MCTS integration.
    """

    def __init__(
        self,
        accuracy_comparator: AccuracyComparator,
        action_rewards: Dict[str, float],
        dataset_name: str,
    ):
        """
        Initialize the Pareto Frontier.

        Args:
            accuracy_comparator: Comparator for evaluating plan accuracy
            action_rewards: Reference to MCTS action_rewards dictionary
            dataset_name: Name of the dataset being optimized (for evaluation and metric selection)
        """
        self.accuracy_comparator = accuracy_comparator
        self.dataset_name = dataset_name

        # Get evaluation function for this dataset
        self.evaluate_func = get_evaluate_func(dataset_name)

        # Dataset-to-primary-metric mapping
        self.dataset_metrics = {
            "cuad": "avg_f1",
            "blackvault": "avg_distinct_locations",
            "game_reviews": "weighted_score",
            "medec": "combined_score",
        }

        # Internal state
        self.plans: List[Node] = []
        self.plans_accuracy: Dict[Node, float] = {}
        self.plans_cost: Dict[Node, float] = {}  # Real costs for display
        self.plans_scaled_cost: Dict[Node, float] = (
            {}
        )  # Scaled costs [0,1] for calculations
        self.frontier_plans: List[Node] = []  # List of nodes on frontier
        self.frontier_data: List[List[int]] = (
            []
        )  # List of [acc, scaled_cost] of nodes on frontier
        self.action_rewards = action_rewards

        # Root plan reference point for hypervolume calculation
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

    def add_plan_f1(self, node: Node) -> Tuple[Dict[Node, int], bool]:
        """
        Add a new plan (Node) to the frontier and estimate its accuracy.

        Args:
            node: Node object representing the plan

        Returns:
            Dict containing affected nodes, bool indicating wether the frontier is updated
        """
        if node.cost == -1:  # Handle error case
            return {}, False

        # Store plan information
        self.plans.append(node)
        self.plans_cost[node] = node.cost  # Store real cost
        # Scaled cost will be calculated in update_pareto_frontier_HV

        result_file_path = node.parsed_yaml["pipeline"]["output"]["path"]

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

        self.plans_accuracy[node] = true_accuracy

        # Set root reference point if this is the first plan (root)
        if len(self.plans) == 1:
            self.root_accuracy = true_accuracy
            self.root_cost = node.cost
            print(
                f"Root reference point set: accuracy={true_accuracy}, cost={node.cost}"
            )

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

    def estimate_accuracy_via_comparisons(self, new_node: Node) -> float:
        """
        Estimate accuracy of new plan through pairwise comparisons.
        """
        # Compare the new node with all nodes on the frontier
        comparison_nodes = self.frontier_plans

        if not comparison_nodes:
            return 0.5

        accuracy_estimates = []

        for existing_node in comparison_nodes:
            try:
                # Get comparison score (-3, -1, 0, 1, 3)
                comparison_score = self.accuracy_comparator.compare(
                    new_node, existing_node
                )

                existing_accuracy = self.plans_accuracy[existing_node]

                # Convert score to accuracy adjustment
                score_to_adjustment = {
                    -3: -0.15,  # Much worse
                    -1: -0.05,  # Slightly worse
                    0: 0.0,  # About the same
                    1: 0.05,  # Slightly better
                    3: 0.15,  # Much better
                }

                adjustment = score_to_adjustment.get(int(comparison_score), 0.0)
                estimated_accuracy = existing_accuracy + adjustment
                accuracy_estimates.append(estimated_accuracy)

            except Exception as e:
                print(
                    f"Comparison failed between {new_node.yaml_file_path} and {existing_node.yaml_file_path}: {e}"
                )
                continue

        if not accuracy_estimates:
            return 0.5

        # Use average of all accuracy estimates
        final_accuracy = sum(accuracy_estimates) / len(accuracy_estimates)

        # Constrain to reasonable range
        return max(0.1, min(0.95, final_accuracy))

    # Helper function to project point onto piecewise linear surface P
    def project_to_frontier(self, node_acc, node_cost, frontier_data):
        """
        Project point onto the piecewise linear surface formed by frontier.
        Returns the projected point coordinates. The projection is whichever point (frontier point or segment projection) gives the smallest distance.
        """
        if not frontier_data:
            return [node_acc, node_cost]

        # Find the closest point on the frontier envelope
        min_distance = float("inf")
        projected_point = [node_acc, node_cost]

        # Check projection onto each frontier segment
        frontier_sorted = sorted(frontier_data, key=lambda x: x[1])  # Sort by cost

        for i in range(len(frontier_sorted)):
            # Project onto the point itself
            fp_acc, fp_cost = frontier_sorted[i]
            distance = ((node_acc - fp_acc) ** 2 + (node_cost - fp_cost) ** 2) ** 0.5
            if distance < min_distance:
                min_distance = distance
                projected_point = [fp_acc, fp_cost]

            # Project onto line segment between consecutive frontier points
            if i < len(frontier_sorted) - 1:
                p1_acc, p1_cost = frontier_sorted[i]
                p2_acc, p2_cost = frontier_sorted[i + 1]

                # Project point onto line segment
                # Vector from p1 to p2
                v_acc = p2_acc - p1_acc
                v_cost = p2_cost - p1_cost

                # Vector from p1 to node
                w_acc = node_acc - p1_acc
                w_cost = node_cost - p1_cost

                # Project w onto v
                if v_acc**2 + v_cost**2 > 0:  # Avoid division by zero
                    t = (w_acc * v_acc + w_cost * v_cost) / (v_acc**2 + v_cost**2)
                    t = max(0, min(1, t))  # Clamp t to [0,1] for line segment

                    proj_acc = p1_acc + t * v_acc
                    proj_cost = p1_cost + t * v_cost

                    distance = (
                        (node_acc - proj_acc) ** 2 + (node_cost - proj_cost) ** 2
                    ) ** 0.5
                    if distance < min_distance:
                        min_distance = distance
                        projected_point = [proj_acc, proj_cost]

        return projected_point

    def _update_action_rewards(self, node: Node, reward: float) -> None:
        """
        Update action rewards based on the reward received by a node.
        Updates the cumulative sum for the latest action that led to this node.

        Args:
            node: The node that received the reward
            reward: The reward value to incorporate
        """
        if not node.latest_action or not self.action_rewards:
            return
        action = node.latest_action
        if action in self.action_rewards:
            # Update cumulative sum
            self.action_rewards[action] += reward

    def _update_scaled_costs(self, valid_nodes: List[Node]) -> None:
        """
        Calculate and update scaled costs for all valid nodes to [0,1] range.
        Also updates the scaled_cost attribute in each node.
        """
        if len(valid_nodes) <= 1:
            # Single node or empty, set scaled cost to 0.5
            for node in valid_nodes:
                scaled_cost = 0.5
                self.plans_scaled_cost[node] = scaled_cost
                node.scaled_cost = scaled_cost
            return

        # Get min and max costs from real costs
        min_cost = min(self.plans_cost[node] for node in valid_nodes)
        max_cost = max(self.plans_cost[node] for node in valid_nodes)
        cost_range = max_cost - min_cost

        if cost_range > 0:
            # Scale all costs to [0,1]
            for node in valid_nodes:
                real_cost = self.plans_cost[node]
                scaled_cost = (real_cost - min_cost) / cost_range
                self.plans_scaled_cost[node] = scaled_cost
                node.scaled_cost = scaled_cost
        else:
            # All costs are the same, set to 0.5
            for node in valid_nodes:
                scaled_cost = 0.5
                self.plans_scaled_cost[node] = scaled_cost
                node.scaled_cost = scaled_cost

    def update_pareto_frontier_HV(self, new_node) -> Tuple[Dict[Node, int], bool]:
        """
        Update the Pareto frontier based on current plans and calculate hyper-volume indicator.
        """

        print("UPDATING Pareto Frontier")

        valid_nodes = [node for node in self.plans if node.cost != -1]
        affected_nodes = {}

        if not valid_nodes:
            self.frontier_plans = []
            self.frontier_data = []
            return affected_nodes, False

        # Save old frontier nodes before updating
        old_frontier_nodes = self.frontier_plans

        # Calculate scaled costs for all valid nodes
        self._update_scaled_costs(valid_nodes)

        # Sort by scaled cost for frontier calculation
        valid_nodes.sort(key=lambda node: self.plans_scaled_cost[node])

        # Reconstruct old frontier data using NEW scaled costs
        archive_frontier_data = []
        for node in old_frontier_nodes:
            if (
                node in valid_nodes and node in self.plans_scaled_cost
            ):  # Only include valid nodes
                acc = self.plans_accuracy.get(node, 0.0)
                scaled_cost = self.plans_scaled_cost[node]
                archive_frontier_data.append([acc, scaled_cost])
            else:
                print(
                    f"INVALID NODE: {node.id}, cost: {node.cost}, in_valid_nodes: {node in valid_nodes}, in_scaled_cost: {node in self.plans_scaled_cost}"
                )

        frontier = []
        max_accuracy_so_far = -1

        for node in valid_nodes:
            accuracy = self.plans_accuracy.get(node, 0.0)

            # Plan is on frontier if it has higher accuracy than all lower-cost plans
            if accuracy > max_accuracy_so_far:
                frontier.append(node)
                max_accuracy_so_far = accuracy

        new_frontier_data = []
        for node in frontier:
            acc = self.plans_accuracy.get(node)
            scaled_cost = self.plans_scaled_cost[
                node
            ]  # Use scaled cost for calculations
            new_frontier_data.append([acc, scaled_cost])

        # Check if frontier actually changed
        old_frontier_set = set(old_frontier_nodes)
        new_frontier_set = set(frontier)
        frontier_updated = old_frontier_set != new_frontier_set

        # Update affected nodes based on frontier changes
        for node in valid_nodes:
            node_scaled_cost = self.plans_scaled_cost[node]
            node_acc = self.plans_accuracy[node]

            if node in new_frontier_set and node not in old_frontier_set:
                # Newly on frontier - reward based on distance to OLD frontier
                node.on_frontier = True
                projected_point_old = self.project_to_frontier(
                    node_acc, node_scaled_cost, archive_frontier_data
                )
                # Weight accuracy 2x more important than cost
                weighted_distance_to_old = (
                    (2 * (node_acc - projected_point_old[0])) ** 2
                    + (node_scaled_cost - projected_point_old[1]) ** 2
                ) ** 0.5
                affected_nodes[node] = weighted_distance_to_old
                # Update action rewards
                self._update_action_rewards(node, weighted_distance_to_old)
            elif node not in new_frontier_set:
                # Not on frontier - give negative reward based on distance to NEW frontier
                node.on_frontier = False
                projected_point = self.project_to_frontier(
                    node_acc, node_scaled_cost, new_frontier_data
                )
                # Weight accuracy 2x more important than cost
                weighted_distance = (
                    (2 * (node_acc - projected_point[0])) ** 2
                    + (node_scaled_cost - projected_point[1]) ** 2
                ) ** 0.5
                affected_nodes[node] = -weighted_distance
                # Update action rewards
                self._update_action_rewards(node, -weighted_distance)
            # Nodes that stayed on frontier don't get updated (maintain their current reward)

        self.frontier_plans = frontier
        self.frontier_data = new_frontier_data
        self.plot_plans()
        return affected_nodes, frontier_updated

    def update_pareto_frontier(self) -> Dict[Node, int]:
        """
        Update the Pareto frontier based on current plans.
        """

        print("UPDATING Pareto Frontier")
        valid_nodes = [node for node in self.plans if node.cost != -1]
        affected_nodes = {}

        if not valid_nodes:
            self.frontier_plans = []
            return affected_nodes

        # Sort by cost
        valid_nodes.sort(key=lambda node: node.cost)

        frontier = []
        max_accuracy_so_far = -1

        for node in valid_nodes:
            accuracy = self.plans_accuracy.get(node, 0.0)

            # Plan is on frontier if it has higher accuracy than all lower-cost plans
            if accuracy > max_accuracy_so_far:
                frontier.append(node)
                max_accuracy_so_far = accuracy

        for node in valid_nodes:
            if node in frontier and node not in self.frontier_plans:  # reward 0 -> 1
                node.on_frontier = True
                affected_nodes[node] = 1
            elif node in self.frontier_plans and node not in frontier:  # reward 1 -> 0
                affected_nodes[node] = -1

        self.frontier_plans = frontier

        self.plot_plans()
        return affected_nodes

    def plot_plans(self):
        """
        Plot all current plans as dots on a cost vs. accuracy graph, annotating each with its id.
        Frontier plans are blue, non-frontier plans are grey.
        """
        if plt is None:
            raise ImportError(
                "matplotlib is required for plotting. Please install it with 'pip install matplotlib'."
            )

        # Separate frontier and non-frontier plans
        frontier_nodes = [node for node in self.plans if node in self.frontier_plans]
        non_frontier_nodes = [
            node for node in self.plans if node not in self.frontier_plans
        ]

        # Plot non-frontier plans (grey)
        if non_frontier_nodes:
            costs = [node.scaled_cost for node in non_frontier_nodes]
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
            costs = [node.scaled_cost for node in frontier_nodes]
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
        plt.title("Plans: Cost vs. Accuracy")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def __len__(self) -> int:
        """Return number of plans in the frontier."""
        return len(self.plans)

    def __contains__(self, node: Node) -> bool:
        """Check if plan is managed by this frontier."""
        return node in self.plans
