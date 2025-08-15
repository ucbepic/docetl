import json
import numpy as np
import matplotlib.pyplot as plt
import os

dataset_metrics = {
    "cuad": "avg_f1",
    "blackvault": "avg_distinct_locations",
    "game_reviews": "weighted_score",
    "medec": "combined_score",
    "sustainability": "economic_activity_accuracy",
    "biodex": "avg_rp_at_10",  
}


def find_pareto_frontier(file, exp_name):
    """
    Find Pareto frontier points from a JSON file.
    
    Args:
        file: Path to the JSON file containing evaluation results
        exp_name: Name of the experiment (e.g., "medec", "cuad", etc.)
    
    Returns:
        List of tuples: (iteration_number, accuracy, cost)
    """
    # Get the metric for this experiment
    if exp_name not in dataset_metrics:
        print(f"Unknown experiment: {exp_name}")
        return []
    
    accuracy_metric = dataset_metrics[exp_name]
    print(f"Using metric: {accuracy_metric}")
    
    # Load data from the file
    try:
        with open(file, 'r') as f:
            data = json.load(f)
        print(f"Loaded {len(data)} data points from {file}")
    except FileNotFoundError:
        print(f"File not found: {file}")
        return []
    except json.JSONDecodeError:
        print(f"Invalid JSON in file: {file}")
        return []
    
    # Filter out entries that don't have the required metrics
    valid_data = [item for item in data if accuracy_metric in item and "cost" in item]
    
    if not valid_data:
        return []
    
    print("valid_data length: ", len(valid_data))
    
    # Sort by cost (ascending) and accuracy (descending)
    sorted_data = sorted(valid_data, key=lambda x: (x["cost"], -x[accuracy_metric]))
    
    pareto_points = []
    best_accuracy = float('-inf')

    if "on_frontier" in sorted_data[0]:
        for item in sorted_data:
            if item["on_frontier"]:
                pareto_points.append((item["node_id"], item[accuracy_metric], item["cost"]))
    else:
        for item in sorted_data:
            current_accuracy = item[accuracy_metric]
            current_cost = item["cost"]
            
            # Check if this point dominates previous points
            if current_accuracy > best_accuracy:
                best_accuracy = current_accuracy
                
                # Extract iteration number from filename
                iteration = item.get("node_id", "")
                if iteration.endswith("_results"):
                    iteration = iteration[:-8]
                if iteration.startswith("iteration_"):
                    iteration = iteration[10:]
                
                pareto_points.append((iteration, current_accuracy, current_cost))
        
    return pareto_points


def plot_pareto_frontier(file, exp_name, pareto_points, output_path=None):
    """
    Plot Pareto frontier points from evaluation results.
    
    Args:
        file: Path to the JSON file containing evaluation results
        exp_name: Name of the experiment
        pareto_points: List of Pareto frontier points (iteration, accuracy, cost)
        output_path: Directory to save the plot (optional)
    
    Returns:
        None
    """
    # Get the metric for this experiment
    if exp_name not in dataset_metrics:
        print(f"Unknown experiment: {exp_name}")
        return
    
    accuracy_metric = dataset_metrics[exp_name]
    
    # Load data from the file
    try:
        with open(file, 'r') as f:
            data = json.load(f)
        print(f"Loaded {len(data)} data points from {file}")
    except FileNotFoundError:
        print(f"File not found: {file}")
        return
    except json.JSONDecodeError:
        print(f"Invalid JSON in file: {file}")
        return
    
    # Filter out entries that don't have the required metrics
    valid_data = [item for item in data if accuracy_metric in item and "cost" in item]
    
    if not valid_data:
        return
    
    if not pareto_points:
        print("No Pareto frontier points provided.")
        return
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    
    # Plot only Pareto frontier points with black markers
    pareto_costs = [point[2] for point in pareto_points]
    pareto_accuracies = [point[1] for point in pareto_points]
    plt.scatter(pareto_costs, pareto_accuracies, c="black", s=100, alpha=0.8)
    
    # Add annotations for Pareto frontier points with full iteration labels
    for iteration, accuracy, cost in pareto_points:
        if iteration == "original":
            label = "original"
        else:
            label = f"iteration_{iteration}"
        plt.annotate(label, (cost, accuracy), 
                    textcoords="offset points", xytext=(5, 5), 
                    fontsize=10, fontweight="bold", color="black")
    
    # Add labels and title
    plt.xlabel("Cost ($)", fontsize=12)
    plt.ylabel(f"{accuracy_metric.replace('_', ' ').title()}", fontsize=12)
    plt.title(f"Cost vs {accuracy_metric.replace('_', ' ').title()} - {exp_name.upper()} Dataset\nPareto Frontier Points", fontsize=14)
    
    # # Set log scale for cost axis
    # plt.xscale('log')
    
    # Add grid
    plt.grid(True, linestyle="--", alpha=0.3)
    
    # Save the plot if output path is provided
    if output_path:
        os.makedirs(output_path, exist_ok=True)
        plot_filename = f"pareto_frontier_{exp_name}.png"
        plot_path = os.path.join(output_path, plot_filename)
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"📈 Pareto frontier plot saved to: {plot_path}")
    
    plt.show()


def calculate_hypervolume(pareto_points, reference_point, accuracy_metric):
    """
    Calculate hypervolume for Pareto frontier points.
    
    Args:
        pareto_points: List of Pareto frontier points (iteration, accuracy, cost)
        reference_point: Reference point data with accuracy and cost
        accuracy_metric: Name of the accuracy metric to use
    
    Returns:
        float: Hypervolume value
    """
    if not pareto_points:
        print("No Pareto frontier points provided, returning 0.0")
        return 0.0
    
    # Use provided reference point
    ref_accuracy = reference_point["accuracy"]
    ref_cost = reference_point["cost"]
    
    print(f"Using reference point: accuracy={ref_accuracy:.4f}, cost={ref_cost:.6f}")
    
    # Sort frontier points by cost (ascending)
    sorted_points = sorted(pareto_points, key=lambda x: x[2])  # x[2] is cost
    
    hypervolume = 0.0
    
    # Calculate trapezoid areas between consecutive frontier points
    for i in range(len(sorted_points) - 1):
        curr_point = sorted_points[i]
        next_point = sorted_points[i + 1]
        
        curr_cost = curr_point[2]  # cost
        curr_accuracy = curr_point[1]  # accuracy
        next_cost = next_point[2]  # cost
        next_accuracy = next_point[1]  # accuracy
        
        if (curr_cost <= ref_cost and curr_accuracy > ref_accuracy and
            next_cost <= ref_cost and next_accuracy > ref_accuracy):
            
            # Scale costs from [0, ref_cost] to [0, 1] for hypervolume calculation
            scaled_curr_cost = curr_cost / ref_cost
            scaled_next_cost = next_cost / ref_cost
            
            # Trapezoid area: (height1 + height2) * width / 2
            width = scaled_next_cost - scaled_curr_cost
            height1 = curr_accuracy - ref_accuracy
            height2 = next_accuracy - ref_accuracy
            
            if width > 0 and height1 > 0 and height2 > 0:
                trapezoid_area = (height1 + height2) * width / 2
                hypervolume += trapezoid_area
    
    # Add final rectangle from last point to reference cost
    if sorted_points:
        last_point = sorted_points[-1]
        last_cost = last_point[2]  # cost
        last_accuracy = last_point[1]  # accuracy
        
        if last_cost < ref_cost and last_accuracy > ref_accuracy:
            # Scale cost for hypervolume calculation
            scaled_last_cost = last_cost / ref_cost
            final_width = 1.0 - scaled_last_cost  # Scaled reference cost is 1.0
            final_height = last_accuracy - ref_accuracy
            if final_width > 0 and final_height > 0:
                final_rectangle = final_width * final_height
                hypervolume += final_rectangle
    
    return hypervolume


def find_highest_cost_across_all_files(file_baseline, file_mcts, file_simple, accuracy_metric):
    """
    Find the highest cost across all data points in the three JSON files.
    
    Args:
        file_baseline: Path to the baseline evaluation file
        file_mcts: Path to the MCTS evaluation file
        file_simple: Path to the simple baseline evaluation file
        accuracy_metric: Name of the accuracy metric to use
    
    Returns:
        float: Highest cost found across all files
    """
    highest_cost = 0.0
    
    # Check baseline file
    try:
        with open(file_baseline, 'r') as f:
            data_baseline = json.load(f)
        for item in data_baseline:
            if accuracy_metric in item and "cost" in item:
                highest_cost = max(highest_cost, item["cost"])
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"Warning: Could not read baseline file {file_baseline}")
    
    # Check MCTS file
    try:
        with open(file_mcts, 'r') as f:
            data_mcts = json.load(f)
        for item in data_mcts:
            if accuracy_metric in item and "cost" in item:
                highest_cost = max(highest_cost, item["cost"])
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"Warning: Could not read MCTS file {file_mcts}")
    
    # Check simple baseline file
    try:
        with open(file_simple, 'r') as f:
            data_simple = json.load(f)
        for item in data_simple:
            if accuracy_metric in item and "cost" in item:
                highest_cost = max(highest_cost, item["cost"])
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"Warning: Could not read simple baseline file {file_simple}")
    
    print(f"Highest cost found across all files: {highest_cost:.6f}")
    return highest_cost


def calculate_hypervolume_comparison(file_baseline, file_mcts, file_simple, exp_name, pareto_points_baseline, pareto_points_mcts, pareto_points_simple):
    """
    Calculate hypervolume for baseline, MCTS, and simple baseline Pareto frontiers.
    
    Args:
        file_baseline: Path to the baseline evaluation file
        file_mcts: Path to the MCTS evaluation file
        file_simple: Path to the simple baseline evaluation file
        exp_name: Name of the experiment
        pareto_points_baseline: List of baseline Pareto frontier points
        pareto_points_mcts: List of MCTS Pareto frontier points
        pareto_points_simple: List of simple baseline Pareto frontier points
    
    Returns:
        tuple: (baseline_hypervolume, mcts_hypervolume, simple_hypervolume, reference_point)
    """
    # Get the metric for this experiment
    if exp_name not in dataset_metrics:
        print(f"Unknown experiment: {exp_name}")
        return 0.0, 0.0, 0.0, None
    
    accuracy_metric = dataset_metrics[exp_name]
    
    # Load baseline data to find original point
    try:
        with open(file_baseline, 'r') as f:
            data_baseline = json.load(f)
    except FileNotFoundError:
        print(f"Baseline file not found: {file_baseline}")
        return 0.0, 0.0, 0.0, None
    except json.JSONDecodeError:
        print(f"Invalid JSON in baseline file: {file_baseline}")
        return 0.0, 0.0, 0.0, None
    
    # Find original point in baseline data
    original_point = None
    for item in data_baseline:
        if item.get("file") == "original_output.json":
            original_point = item
            break
    
    if not original_point:
        raise Exception("Original point not found in baseline data")
    
    print(f"Original point: {accuracy_metric}={original_point[accuracy_metric]:.4f}, cost={original_point['cost']:.6f}")
    
    # Find highest cost across all files and use it for reference point
    highest_cost = find_highest_cost_across_all_files(file_baseline, file_mcts, file_simple, accuracy_metric)
    
    # Create reference point: accuracy = 0, cost = highest cost found
    reference_point = {
        "accuracy": 0.0,
        "cost": highest_cost
    }
    
    print(f"Reference point set to: accuracy={reference_point['accuracy']:.4f}, cost={reference_point['cost']:.6f}")
    
    # Calculate hypervolume for baseline
    baseline_hypervolume = calculate_hypervolume(pareto_points_baseline, reference_point, accuracy_metric)

    # Calculate hypervolume for MCTS
    mcts_hypervolume = calculate_hypervolume(pareto_points_mcts, reference_point, accuracy_metric)
    
    # Calculate hypervolume for simple baseline
    simple_hypervolume = calculate_hypervolume(pareto_points_simple, reference_point, accuracy_metric)
    
    
    return baseline_hypervolume, mcts_hypervolume, simple_hypervolume, reference_point


def plot_hypervolume_trapezoids_and_rectangle(plt, pareto_points, reference_point, color, alpha=0.3, label_suffix=""):
    """
    Plot hypervolume areas as continuous shaded regions following the Pareto frontier curve,
    extending smoothly to the reference point without any vertical lines.
    
    Args:
        plt: Matplotlib plot object
        pareto_points: List of Pareto frontier points (iteration, accuracy, cost)
        reference_point: Reference point data with accuracy and cost
        color: Color for the shaded areas
        alpha: Transparency of the shaded areas
        label_suffix: Suffix to add to the label
    """
    if not pareto_points:
        print(f"No Pareto points for {label_suffix}")
        return
    
    # Sort frontier points by cost (ascending)
    sorted_points = sorted(pareto_points, key=lambda x: x[2])  # x[2] is cost
    
    ref_accuracy = reference_point["accuracy"]
    ref_cost = reference_point["cost"]
    
    print(f"Plotting {label_suffix}: {len(sorted_points)} points, ref_cost={ref_cost:.6f}, ref_accuracy={ref_accuracy:.4f}")
    
    # Create continuous shaded area following the Pareto frontier and extending to reference point
    if len(sorted_points) >= 1:
        # Extract costs and accuracies for points within reference bounds
        costs = [point[2] for point in sorted_points if point[2] <= ref_cost and point[1] > ref_accuracy]
        accuracies = [point[1] for point in sorted_points if point[2] <= ref_cost and point[1] > ref_accuracy]
        
        print(f"  {label_suffix}: {len(costs)} points within bounds")
        
        if len(costs) >= 1:
            # Create polygon for the continuous shaded area
            polygon_points = []
            
            # Start from reference point
            polygon_points.append([ref_cost, ref_accuracy])
            
            # Go to first point on curve
            polygon_points.append([costs[0], ref_accuracy])
            polygon_points.append([costs[0], accuracies[0]])
            
            # If we have multiple points, follow the Pareto frontier curve
            if len(costs) > 1:
                for i in range(len(costs)):
                    polygon_points.append([costs[i], accuracies[i]])
                
                # Extend horizontally to the reference point (no vertical line)
                polygon_points.append([ref_cost, accuracies[-1]])
            else:
                # For single point, just extend horizontally to reference point
                polygon_points.append([ref_cost, accuracies[0]])
            
            # Close back to reference point
            polygon_points.append([ref_cost, ref_accuracy])
            
            # Plot the continuous area
            if len(polygon_points) > 2:
                polygon_points = np.array(polygon_points)
                plt.fill(polygon_points[:, 0], polygon_points[:, 1], 
                        color=color, alpha=alpha)
                print(f"  {label_suffix}: Plotted area with {len(polygon_points)} polygon points")
            else:
                print(f"  {label_suffix}: Not enough polygon points ({len(polygon_points)})")
        else:
            print(f"  {label_suffix}: Not enough points within bounds")
    else:
        print(f"  {label_suffix}: Not enough sorted points ({len(sorted_points)})")


def plot_pareto_frontier_comparison(file_baseline, file_mcts, file_simple, exp_name, pareto_points_baseline, pareto_points_mcts, pareto_points_simple, output_path=None, reference_point=None):
    """
    Plot Pareto frontier points from baseline, MCTS, and simple baseline on a single graph.
    
    Args:
        file_baseline: Path to the baseline evaluation file
        file_mcts: Path to the MCTS evaluation file
        file_simple: Path to the simple baseline evaluation file
        exp_name: Name of the experiment
        pareto_points_baseline: List of baseline Pareto frontier points
        pareto_points_mcts: List of MCTS Pareto frontier points
        pareto_points_simple: List of simple baseline Pareto frontier points
        output_path: Directory to save the plot (optional)
    
    Returns:
        None
    """
    # Get the metric for this experiment
    if exp_name not in dataset_metrics:
        print(f"Unknown experiment: {exp_name}")
        return
    
    accuracy_metric = dataset_metrics[exp_name]
    
    # Load baseline data
    try:
        with open(file_baseline, 'r') as f:
            data_baseline = json.load(f)
        print(f"Loaded {len(data_baseline)} baseline data points")
    except FileNotFoundError:
        print(f"Baseline file not found: {file_baseline}")
        return
    except json.JSONDecodeError:
        print(f"Invalid JSON in baseline file: {file_baseline}")
        return
    
    # Load MCTS data
    try:
        with open(file_mcts, 'r') as f:
            data_mcts = json.load(f)
        print(f"Loaded {len(data_mcts)} MCTS data points")
    except FileNotFoundError:
        print(f"MCTS file not found: {file_mcts}")
        return
    except json.JSONDecodeError:
        print(f"Invalid JSON in MCTS file: {file_mcts}")
        return
    
    # Load simple baseline data
    try:
        with open(file_simple, 'r') as f:
            data_simple = json.load(f)
        print(f"Loaded {len(data_simple)} simple baseline data points")
    except FileNotFoundError:
        print(f"Simple baseline file not found: {file_simple}")
        return
    except json.JSONDecodeError:
        print(f"Invalid JSON in simple baseline file: {file_simple}")
        return
    
    # Filter data for required metrics
    valid_baseline = [item for item in data_baseline if accuracy_metric in item and "cost" in item]
    valid_mcts = [item for item in data_mcts if accuracy_metric in item and "cost" in item]
    valid_simple = [item for item in data_simple if accuracy_metric in item and "cost" in item]
    
    if not valid_baseline or not valid_mcts or not valid_simple:
        print("No valid data found in baseline, MCTS, or simple baseline files.")
        return
    
    # Find original point in baseline data
    original_point = None
    for item in data_baseline:
        if item.get("file") == "original_output.json":
            original_point = item
            break
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Log what's being plotted for each approach
    print(f"Plotting Baseline: {len(pareto_points_baseline) if pareto_points_baseline else 0} Pareto frontier points")
    print(f"Plotting MCTS: {len(pareto_points_mcts) if pareto_points_mcts else 0} Pareto frontier points")
    print(f"Plotting Simple Baseline: {len(pareto_points_simple) if pareto_points_simple else 0} Pareto frontier points")
    
    # Plot simple baseline Pareto frontier points (green) if available
    if pareto_points_simple:
        simple_costs = [point[2] for point in pareto_points_simple]
        simple_accuracies = [point[1] for point in pareto_points_simple]
        plt.scatter(simple_costs, simple_accuracies, c="green", s=100, alpha=0.8, label="Simple Baseline (with iteration #)")
    else:
        print("Note: No Simple Baseline Pareto frontier points to plot")
    
    # Plot baseline Pareto frontier points (black) if available
    if pareto_points_baseline:
        baseline_costs = [point[2] for point in pareto_points_baseline]
        baseline_accuracies = [point[1] for point in pareto_points_baseline]
        plt.scatter(baseline_costs, baseline_accuracies, c="black", s=100, alpha=0.8, label="Baseline (with iteration #)")
    else:
        print("Note: No Baseline Pareto frontier points to plot")
    
    # Plot MCTS Pareto frontier points (blue) if available
    if pareto_points_mcts:
        mcts_costs = [point[2] for point in pareto_points_mcts]
        mcts_accuracies = [point[1] for point in pareto_points_mcts]
        plt.scatter(mcts_costs, mcts_accuracies, c="blue", s=100, alpha=0.8, label="MCTS (with iteration #)")
    else:
        print("Note: No MCTS Pareto frontier points to plot")
    
    # Plot all non-frontier nodes with lighter colors
    # Simple baseline non-frontier nodes
    if valid_simple:
        simple_frontier_costs = set(point[2] for point in pareto_points_simple) if pareto_points_simple else set()
        simple_frontier_accuracies = set(point[1] for point in pareto_points_simple) if pareto_points_simple else set()
        
        non_frontier_simple = [item for item in valid_simple 
                             if not (item["cost"] in simple_frontier_costs and item[accuracy_metric] in simple_frontier_accuracies)]
        
        if non_frontier_simple:
            non_frontier_costs = [item["cost"] for item in non_frontier_simple]
            non_frontier_accuracies = [item[accuracy_metric] for item in non_frontier_simple]
            plt.scatter(non_frontier_costs, non_frontier_accuracies, c="green", s=50, alpha=0.3, label="Simple Baseline (Non-Frontier)")
    
    # Baseline non-frontier nodes
    if valid_baseline:
        baseline_frontier_costs = set(point[2] for point in pareto_points_baseline) if pareto_points_baseline else set()
        baseline_frontier_accuracies = set(point[1] for point in pareto_points_baseline) if pareto_points_baseline else set()
        
        non_frontier_baseline = [item for item in valid_baseline 
                               if not (item["cost"] in baseline_frontier_costs and item[accuracy_metric] in baseline_frontier_accuracies)]
        
        if non_frontier_baseline:
            non_frontier_costs = [item["cost"] for item in non_frontier_baseline]
            non_frontier_accuracies = [item[accuracy_metric] for item in non_frontier_baseline]
            plt.scatter(non_frontier_costs, non_frontier_accuracies, c="black", s=50, alpha=0.3, label="Baseline (Non-Frontier)")
    
    # MCTS non-frontier nodes
    if valid_mcts:
        mcts_frontier_costs = set(point[2] for point in pareto_points_mcts) if pareto_points_mcts else set()
        mcts_frontier_accuracies = set(point[1] for point in pareto_points_mcts) if pareto_points_mcts else set()
        
        non_frontier_mcts = [item for item in valid_mcts 
                           if not (item["cost"] in mcts_frontier_costs and item[accuracy_metric] in mcts_frontier_accuracies)]
        
        if non_frontier_mcts:
            non_frontier_costs = [item["cost"] for item in non_frontier_mcts]
            non_frontier_accuracies = [item[accuracy_metric] for item in non_frontier_mcts]
            plt.scatter(non_frontier_costs, non_frontier_accuracies, c="blue", s=50, alpha=0.3, label="MCTS (Non-Frontier)")
    
    # Plot original point in red
    if original_point:
        plt.scatter(original_point["cost"], original_point[accuracy_metric], 
                   c="red", s=100, marker="o",
                   label="Original", zorder=5)
        # Add annotation for original point
        plt.annotate("original", (original_point["cost"], original_point[accuracy_metric]), 
                    textcoords="offset points", xytext=(5, 5), 
                    fontsize=10, fontweight="bold", color="red")
    
    # Plot hypervolume areas as trapezoids and rectangle for each Pareto frontier
    if reference_point:
        # Debug information
        print(f"Plotting hypervolume areas:")
        print(f"  Baseline: {len(pareto_points_baseline) if pareto_points_baseline else 0} points")
        print(f"  MCTS: {len(pareto_points_mcts) if pareto_points_mcts else 0} points")
        print(f"  Simple Baseline: {len(pareto_points_simple) if pareto_points_simple else 0} points")
        
        # Plot hypervolume areas using the provided reference point
        plot_hypervolume_trapezoids_and_rectangle(plt, pareto_points_simple, reference_point, "green", 0.1, "Simple Baseline")
        plot_hypervolume_trapezoids_and_rectangle(plt, pareto_points_baseline, reference_point, "black", 0.1, "Baseline")
        plot_hypervolume_trapezoids_and_rectangle(plt, pareto_points_mcts, reference_point, "blue", 0.1, "MCTS")
        
        # Plot reference point as a large marker
        plt.scatter(reference_point["cost"], reference_point["accuracy"], 
                   c="gray", s=100, marker="o", edgecolors="black", linewidth=1,
                   label="Reference Point", zorder=4)
        # Add annotation for reference point
        plt.annotate("Reference", (reference_point["cost"], reference_point["accuracy"]), 
                    textcoords="offset points", xytext=(5, 5), 
                    fontsize=10, fontweight="bold", color="gray")
    
    # Add annotations for simple baseline Pareto frontier points
    if pareto_points_simple:
        for i, (iteration, accuracy, cost) in enumerate(pareto_points_simple):
            if iteration == "0":
                label = "original"
            else:
                label = str(iteration)  # Use the actual iteration value
            plt.annotate(label, (cost, accuracy), 
                        textcoords="offset points", xytext=(5, 5), 
                        fontsize=10, fontweight="bold", color="green")
    
    # Add annotations for baseline Pareto frontier points
    if pareto_points_baseline:
        for i, (iteration, accuracy, cost) in enumerate(pareto_points_baseline):
            if iteration == "original_output":
                continue
            else:
                label = str(iteration)  # Use the actual iteration value
            plt.annotate(label, (cost, accuracy), 
                        textcoords="offset points", xytext=(5, 5), 
                        fontsize=10, fontweight="bold", color="black")
    
    # Add annotations for MCTS Pareto frontier points
    if pareto_points_mcts:
        for i, (iteration, accuracy, cost) in enumerate(pareto_points_mcts):
            if iteration == "0":
                continue
            else:
                label = str(iteration)  # Use the actual iteration value
            plt.annotate(label, (cost, accuracy), 
                        textcoords="offset points", xytext=(5, 5), 
                        fontsize=10, fontweight="bold", color="blue")
    
    # Add labels and title
    plt.xlabel("Cost ($)", fontsize=12)
    plt.ylabel(f"{accuracy_metric.replace('_', ' ').title()}", fontsize=12)
    
    # Always show all three approaches in the title
    title = f"Cost vs {accuracy_metric.replace('_', ' ').title()} - {exp_name.upper()} Dataset"
    plt.title(title, fontsize=14)
    
    # Set log scale for cost axis
    plt.xscale('log')
    
    # Log the cost range for debugging
    if reference_point:
        print(f"Cost axis range: 0.001 to {reference_point['cost']:.6f} (log scale)")
    
    # Add grid and legend
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(loc='lower left')
    
    # Save the plot if output path is provided
    if output_path:
        os.makedirs(output_path, exist_ok=True)
        plot_filename = f"pareto_frontier_comparison_{exp_name}.png"
        plot_path = os.path.join(output_path, plot_filename)
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"📈 Pareto frontier comparison plot saved to: {plot_path}")
    
    plt.show()


def main():
    
    evaluation_file_baseline = "/Users/lindseywei/Documents/DocETL-optimizer/reasoning-optimizer/outputs/sustainability_baseline/evaluation_metrics.json"
    evaluation_file_mcts = "/Users/lindseywei/Documents/DocETL-optimizer/reasoning-optimizer/outputs/sustainability_mcts/evaluation_metrics.json"
    evaluation_file_simple = "/Users/lindseywei/Documents/DocETL-optimizer/reasoning-optimizer/outputs/sustainability_simple_baseline/evaluation_metrics.json"
    exp_name = "sustainability"
    
    # Find Pareto frontier for all three approaches
    pareto_points_baseline = find_pareto_frontier(evaluation_file_baseline, exp_name)
    pareto_points_mcts = find_pareto_frontier(evaluation_file_mcts, exp_name)
    pareto_points_simple = find_pareto_frontier(evaluation_file_simple, exp_name)
    
    # Check if we have at least baseline and MCTS (required for comparison)
    if pareto_points_baseline and pareto_points_mcts:
        # Calculate hypervolume for all three approaches
        print("=" * 60)
        print("HYPERVOLUME CALCULATION")
        print("=" * 60)
        baseline_hv, mcts_hv, simple_hv, reference_point = calculate_hypervolume_comparison(
            evaluation_file_baseline, evaluation_file_mcts, evaluation_file_simple, exp_name, 
            pareto_points_baseline, pareto_points_mcts, pareto_points_simple
        )
        print(f"Baseline hypervolume: {baseline_hv:.4f}")
        print(f"MCTS hypervolume: {mcts_hv:.4f}")
        print(f"Simple Baseline hypervolume: {simple_hv:.4f}")
        print(f"Reference point: accuracy={reference_point['accuracy']:.4f}, cost={reference_point['cost']:.6f}")
        
        # Plot all three approaches together
        print("\n" + "=" * 60)
        print("PLOTTING ALL THREE APPROACHES TOGETHER")
        print("=" * 60)
        output_dir = os.path.dirname(evaluation_file_baseline)
        plot_pareto_frontier_comparison(evaluation_file_baseline, evaluation_file_mcts, evaluation_file_simple, exp_name, pareto_points_baseline, pareto_points_mcts, pareto_points_simple, output_dir, reference_point)
    else:
        print("No Pareto frontier points found for baseline or MCTS approaches.")
    
    return 0


if __name__ == "__main__":
    exit(main())