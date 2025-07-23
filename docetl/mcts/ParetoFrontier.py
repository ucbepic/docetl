from typing import Any, Dict, List

import matplotlib.pyplot as plt

from .acc_comparator import AccuracyComparator
from .Node import Node


class ParetoFrontier:
    """
    Pareto Frontier class for managing cost-accuracy optimization.

    This class maintains a collection of plans, estimates their accuracy through
    pairwise comparisons, constructs and updates the Pareto frontier, and provides
    value calculations for MCTS integration.
    """

    def __init__(self, accuracy_comparator: AccuracyComparator):
        """
        Initialize the Pareto Frontier.

        Args:
            accuracy_comparator: Comparator for evaluating plan accuracy
        """
        self.accuracy_comparator = accuracy_comparator

        # Internal state
        self.plans: List[Node] = []
        self.plans_accuracy: Dict[Node, float] = {}
        self.plans_cost: Dict[Node, float] = {}
        self.frontier_plans: List[Node] = []  # List of nodes on frontier

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
            costs = [node.cost for node in non_frontier_nodes]
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
            costs = [node.cost for node in frontier_nodes]
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

    # def calculate_plan_metrics(self, node: Node) -> Dict[str, float]:
    #     """
    #     Calculate comprehensive metrics for a plan.
    #     """
    #     if node not in self.plans:
    #         return {'accuracy': 0.0, 'pareto_value': 0.0, 'frontier_distance': float('inf')}

    #     accuracy = self.plans_accuracy[node]
    #     cost = node.cost

    #     # Normalized values
    #     normalized_cost = self.normalize_cost(cost)
    #     normalized_accuracy = self.normalize_accuracy(accuracy)

    #     # Check if on frontier
    #     is_frontier = node in self.frontier_plans

    #     # Calculate Pareto value
    #     pareto_value = self.calculate_pareto_value(node, is_frontier)

    #     return {
    #         'accuracy': accuracy,
    #         'pareto_value': pareto_value,
    #         'frontier_distance': self.calculate_frontier_distance(node),
    #         'normalized_cost': normalized_cost,
    #         'normalized_accuracy': normalized_accuracy,
    #         'is_frontier': is_frontier
    #     }

    # def calculate_frontier_distance(self, node: Node) -> float:
    #     """
    #     Calculate distance from plan to Pareto frontier.
    #     """
    #     if not self.frontier_plans or node in self.frontier_plans:
    #         return 0.0

    #     plan_accuracy = self.plans_accuracy[node]
    #     plan_cost = node.cost

    #     min_distance = float('inf')

    #     for frontier_node in self.frontier_plans:
    #         frontier_accuracy = self.plans_accuracy[frontier_node]
    #         frontier_cost = frontier_node.cost

    #         # Euclidean distance in normalized space
    #         cost_diff = self.normalize_cost(plan_cost) - self.normalize_cost(frontier_cost)
    #         accuracy_diff = self.normalize_accuracy(plan_accuracy) - self.normalize_accuracy(frontier_accuracy)

    #         distance = math.sqrt(cost_diff**2 + accuracy_diff**2)
    #         min_distance = min(min_distance, distance)

    #     return min_distance

    # def calculate_pareto_value(self, node: Node) -> float:
    #     """
    #     Calculate the Pareto value for a given node.

    #     Args:
    #         node: The node to calculate value for
    #         normalized_cost: Cost normalized to [0, 1] range
    #         normalized_accuracy: Accuracy normalized to [0, 1] range
    #         is_frontier: Whether the node is on the Pareto frontier

    #     Returns:
    #         float: Pareto value between 0 and 1
    #     """
    #     is_frontier = node in self.frontier_plans

    #     if is_frontier:
    #         # High base value for frontier plans
    #         pareto_value = 0.8
    #     else:
    #         # Distance-based value for non-frontier plans
    #         frontier_distance = self.calculate_frontier_distance(node)
    #         pareto_value = max(0.1, 0.6 - frontier_distance)

    #     return pareto_value

    def __len__(self) -> int:
        """Return number of plans in the frontier."""
        return len(self.plans)

    def __contains__(self, node: Node) -> bool:
        """Check if plan is managed by this frontier."""
        return node in self.plans


if __name__ == "__main__":
    # Example usage
    # Note: You'll need to implement your own AccuracyComparator
    # from your_accuracy_comparator import AccuracyComparator

    pass

    # Create example nodes (you'll need to create actual Node objects)
    # from Node import Node
    # node_a = Node("plan_a.yaml")
    # node_b = Node("plan_b.yaml")
    # node_c = Node("plan_c.yaml")

    # Add some example plans (commented out since we need actual Node objects)
    # frontier.add_plan(node_a)
    # frontier.add_plan(node_b)
    # frontier.add_plan(node_c)
