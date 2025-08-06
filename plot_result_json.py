import json
import matplotlib.pyplot as plt
import numpy as np

def plot_accuracy_cost_graph(json_file_path):
    """
    Read JSON file and create an accuracy-cost graph with logarithmic cost scale.
    Blue points for on-frontier entries, gray points for off-frontier entries.
    Each point is annotated with its node_id.
    
    Args:
        json_file_path (str): Path to the JSON file
    """
    # Read the JSON file
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    
    # Extract data for on-frontier and off-frontier points
    on_frontier_data = []
    off_frontier_data = []
    
    # Handle both single object and array of objects
    if isinstance(data, dict):
        data = [data]
    
    for entry in data:
        if 'cost' in entry and 'mcts_accuracy' in entry and 'on_frontier' in entry and 'node_id' in entry:
            point_data = {
                'cost': entry['cost'],
                'accuracy': entry['mcts_accuracy'],
                'node_id': entry['node_id']
            }
            
            if entry['on_frontier']:
                on_frontier_data.append(point_data)
            else:
                off_frontier_data.append(point_data)
    
    if not (on_frontier_data or off_frontier_data):
        print("No valid cost/accuracy data found in the JSON file.")
        return
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    
    # Plot off-frontier points first (grey)
    if off_frontier_data:
        costs = [point['cost'] for point in off_frontier_data]
        accuracies = [point['accuracy'] for point in off_frontier_data]
        node_ids = [point['node_id'] for point in off_frontier_data]
        
        plt.scatter(costs, accuracies, color='grey', label='Off Frontier')
        
        # Add annotations for off-frontier points
        for x, y, label in zip(costs, accuracies, node_ids):
            plt.annotate(
                str(label),
                (x, y),
                textcoords="offset points",
                xytext=(5, 5),
                ha="left",
                fontsize=9,
                color="grey"
            )
    
    # Plot on-frontier points (blue)
    if on_frontier_data:
        costs = [point['cost'] for point in on_frontier_data]
        accuracies = [point['accuracy'] for point in on_frontier_data]
        node_ids = [point['node_id'] for point in on_frontier_data]
        
        plt.scatter(costs, accuracies, color='blue', label='Frontier')
        
        # Add annotations for frontier points
        for x, y, label in zip(costs, accuracies, node_ids):
            plt.annotate(
                str(label),
                (x, y),
                textcoords="offset points",
                xytext=(5, 5),
                ha="left",
                fontsize=9,
                color="blue"
            )
    
    # Set logarithmic scale for x-axis (cost)
    plt.xscale('log')
    
    # Add labels and title (matching the reference style)
    plt.xlabel('Cost')
    plt.ylabel('Avg_distinct_locations')
    plt.title('Plans: Cost vs. Avg_distinct_locations')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    
    # Show the plot
    plt.show()
    
    # Print statistics
    total_points = len(on_frontier_data) + len(off_frontier_data)
    all_costs = [point['cost'] for point in on_frontier_data + off_frontier_data]
    all_accuracies = [point['accuracy'] for point in on_frontier_data + off_frontier_data]
    
    print(f"Statistics:")
    print(f"Total data points: {total_points}")
    print(f"On-frontier points: {len(on_frontier_data)}")
    print(f"Off-frontier points: {len(off_frontier_data)}")
    
    if all_costs and all_accuracies:
        print(f"Cost range: {min(all_costs):.6f} to {max(all_costs):.6f}")
        print(f"Accuracy range: {min(all_accuracies):.4f} to {max(all_accuracies):.4f}")

def plot_linear_scale_graph(json_file_path):
    """
    Create the same plot but with linear scale (like the reference image).
    """
    # Read the JSON file
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    
    # Extract data
    on_frontier_data = []
    off_frontier_data = []
    
    if isinstance(data, dict):
        data = [data]
    
    for entry in data:
        if 'cost' in entry and 'mcts_accuracy' in entry and 'on_frontier' in entry and 'node_id' in entry:
            point_data = {
                'cost': entry['cost'],
                'accuracy': entry['mcts_accuracy'],
                'node_id': entry['node_id']
            }
            
            if entry['on_frontier']:
                on_frontier_data.append(point_data)
            else:
                off_frontier_data.append(point_data)
    
    if not (on_frontier_data or off_frontier_data):
        print("No valid cost/accuracy data found in the JSON file.")
        return
    
    # Create the plot with linear scale (like reference image)
    plt.figure(figsize=(10, 8))
    
    # Plot off-frontier points first (grey)
    if off_frontier_data:
        costs = [point['cost'] for point in off_frontier_data]
        accuracies = [point['accuracy'] for point in off_frontier_data]
        node_ids = [point['node_id'] for point in off_frontier_data]
        
        plt.scatter(costs, accuracies, color='grey', label='Off Frontier')
        
        # Add annotations for off-frontier points
        for x, y, label in zip(costs, accuracies, node_ids):
            plt.annotate(
                str(label),
                (x, y),
                textcoords="offset points",
                xytext=(5, 5),
                ha="left",
                fontsize=9,
                color="grey"
            )
    
    # Plot on-frontier points (blue)
    if on_frontier_data:
        costs = [point['cost'] for point in on_frontier_data]
        accuracies = [point['accuracy'] for point in on_frontier_data]
        node_ids = [point['node_id'] for point in on_frontier_data]
        
        plt.scatter(costs, accuracies, color='blue', label='Frontier')
        
        # Add annotations for frontier points
        for x, y, label in zip(costs, accuracies, node_ids):
            plt.annotate(
                str(label),
                (x, y),
                textcoords="offset points",
                xytext=(5, 5),
                ha="left",
                fontsize=9,
                color="blue"
            )
    
    # Use linear scale (like the reference image)
    plt.xlabel('Cost')
    plt.ylabel('Accuracy')
    plt.title('Plans: Cost vs. Accuracy')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    
    # Show the plot
    plt.show()

def plot_scaled_accuracy_graph(json_file_path):
    """
    Create a plot with accuracy scaled to [0,1] and cost on log scale.
    """
    # Read the JSON file
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    
    # Extract data
    on_frontier_data = []
    off_frontier_data = []
    
    if isinstance(data, dict):
        data = [data]
    
    for entry in data:
        if 'cost' in entry and 'mcts_accuracy' in entry and 'on_frontier' in entry and 'node_id' in entry:
            point_data = {
                'cost': entry['cost'],
                'accuracy': entry['mcts_accuracy'],
                'node_id': entry['node_id']
            }
            
            if entry['on_frontier']:
                on_frontier_data.append(point_data)
            else:
                off_frontier_data.append(point_data)
    
    if not (on_frontier_data or off_frontier_data):
        print("No valid cost/accuracy data found in the JSON file.")
        return
    
    # Get all accuracy values to find min/max for scaling
    all_accuracies = [point['accuracy'] for point in on_frontier_data + off_frontier_data]
    min_acc = min(all_accuracies)
    max_acc = max(all_accuracies)
    
    print(f"Original accuracy range: {min_acc:.4f} to {max_acc:.4f}")
    
    # Scale accuracy to [0,1]
    def scale_accuracy(acc):
        if max_acc == min_acc:
            return 0.5  # If all values are the same, put them in the middle
        return (acc - min_acc) / (max_acc - min_acc)
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    
    # Plot off-frontier points
    if off_frontier_data:
        costs = [point['cost'] for point in off_frontier_data]
        accuracies = [scale_accuracy(point['accuracy']) for point in off_frontier_data]
        node_ids = [point['node_id'] for point in off_frontier_data]
        
        plt.scatter(costs, accuracies, color='grey', label='Off Frontier')
        
        for x, y, label in zip(costs, accuracies, node_ids):
            plt.annotate(
                str(label),
                (x, y),
                textcoords="offset points",
                xytext=(5, 5),
                ha="left",
                fontsize=9,
                color="grey"
            )
    
    # Plot on-frontier points
    if on_frontier_data:
        costs = [point['cost'] for point in on_frontier_data]
        accuracies = [scale_accuracy(point['accuracy']) for point in on_frontier_data]
        node_ids = [point['node_id'] for point in on_frontier_data]
        
        plt.scatter(costs, accuracies, color='blue', label='Frontier')
        
        for x, y, label in zip(costs, accuracies, node_ids):
            plt.annotate(
                str(label),
                (x, y),
                textcoords="offset points",
                xytext=(5, 5),
                ha="left",
                fontsize=9,
                color="blue"
            )
    
    # Set logarithmic scale for x-axis (cost)
    plt.xscale('log')
    
    # Set y-axis limits to [0,1] with some padding
    plt.ylim(-0.05, 1.05)
    
    plt.xlabel('Cost (log scale)')
    plt.ylabel('Accuracy (scaled to [0,1])')
    plt.title('Plans: Cost vs. Scaled Accuracy')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    
    # Show the plot
    plt.show()
    
    print(f"Accuracy has been scaled from [{min_acc:.4f}, {max_acc:.4f}] to [0, 1]")

def plot_comparison_graphs(json_file_path):
    """
    Create side-by-side comparison of original and scaled accuracy.
    """
    # Read the JSON file
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    
    # Extract data
    on_frontier_data = []
    off_frontier_data = []
    
    if isinstance(data, dict):
        data = [data]
    
    for entry in data:
        if 'cost' in entry and 'mcts_accuracy' in entry and 'on_frontier' in entry and 'node_id' in entry:
            point_data = {
                'cost': entry['cost'],
                'accuracy': entry['mcts_accuracy'],
                'node_id': entry['node_id']
            }
            
            if entry['on_frontier']:
                on_frontier_data.append(point_data)
            else:
                off_frontier_data.append(point_data)
    
    if not (on_frontier_data or off_frontier_data):
        print("No valid cost/accuracy data found in the JSON file.")
        return
    
    # Get scaling parameters
    all_accuracies = [point['accuracy'] for point in on_frontier_data + off_frontier_data]
    min_acc = min(all_accuracies)
    max_acc = max(all_accuracies)
    
    def scale_accuracy(acc):
        if max_acc == min_acc:
            return 0.5
        return (acc - min_acc) / (max_acc - min_acc)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # ========== GRAPH 1: ORIGINAL ACCURACY ==========
    ax1.set_title('Plans: Cost vs. Original Accuracy')
    
    # Plot off-frontier points
    if off_frontier_data:
        costs = [point['cost'] for point in off_frontier_data]
        accuracies = [point['accuracy'] for point in off_frontier_data]
        node_ids = [point['node_id'] for point in off_frontier_data]
        
        ax1.scatter(costs, accuracies, color='grey', label='Off Frontier')
        
        for x, y, label in zip(costs, accuracies, node_ids):
            ax1.annotate(
                str(label),
                (x, y),
                textcoords="offset points",
                xytext=(5, 5),
                ha="left",
                fontsize=9,
                color="grey"
            )
    
    # Plot on-frontier points
    if on_frontier_data:
        costs = [point['cost'] for point in on_frontier_data]
        accuracies = [point['accuracy'] for point in on_frontier_data]
        node_ids = [point['node_id'] for point in on_frontier_data]
        
        ax1.scatter(costs, accuracies, color='blue', label='Frontier')
        
        for x, y, label in zip(costs, accuracies, node_ids):
            ax1.annotate(
                str(label),
                (x, y),
                textcoords="offset points",
                xytext=(5, 5),
                ha="left",
                fontsize=9,
                color="blue"
            )
    
    ax1.set_xscale('log')
    ax1.set_xlabel('Cost (log scale)')
    ax1.set_ylabel('Accuracy (original)')
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend()
    
    # ========== GRAPH 2: SCALED ACCURACY ==========
    ax2.set_title('Plans: Cost vs. Scaled Accuracy')
    
    # Plot off-frontier points
    if off_frontier_data:
        costs = [point['cost'] for point in off_frontier_data]
        accuracies = [scale_accuracy(point['accuracy']) for point in off_frontier_data]
        node_ids = [point['node_id'] for point in off_frontier_data]
        
        ax2.scatter(costs, accuracies, color='grey', label='Off Frontier')
        
        for x, y, label in zip(costs, accuracies, node_ids):
            ax2.annotate(
                str(label),
                (x, y),
                textcoords="offset points",
                xytext=(5, 5),
                ha="left",
                fontsize=9,
                color="grey"
            )
    
    # Plot on-frontier points
    if on_frontier_data:
        costs = [point['cost'] for point in on_frontier_data]
        accuracies = [scale_accuracy(point['accuracy']) for point in on_frontier_data]
        node_ids = [point['node_id'] for point in on_frontier_data]
        
        ax2.scatter(costs, accuracies, color='blue', label='Frontier')
        
        for x, y, label in zip(costs, accuracies, node_ids):
            ax2.annotate(
                str(label),
                (x, y),
                textcoords="offset points",
                xytext=(5, 5),
                ha="left",
                fontsize=9,
                color="blue"
            )
    
    ax2.set_xscale('log')
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_xlabel('Cost (log scale)')
    ax2.set_ylabel('Accuracy (scaled to [0,1])')
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend()
    
    # Adjust layout
    plt.tight_layout()
    plt.show()
    
    print(f"Scaling info: Original range [{min_acc:.4f}, {max_acc:.4f}] → [0, 1]")

# Example usage
if __name__ == "__main__":
    json_file_path = '/Users/lindseywei/Documents/DocETL-optimizer/reasoning-optimizer/outputs/blackvault_mcts/evaluation_metrics.json'
    
    try:
        # Single plot with log scale and annotations
        print("Generating accuracy-cost graph (log scale) with node ID annotations...")
        plot_accuracy_cost_graph(json_file_path)
        
        # Single plot with linear scale (like reference image)
        print("\nGenerating accuracy-cost graph (linear scale) with node ID annotations...")
        plot_linear_scale_graph(json_file_path)
        
        # Side-by-side comparison
        print("\nGenerating comparison graphs...")
        plot_comparison_graphs(json_file_path)
        
    except FileNotFoundError:
        print(f"Error: Could not find file '{json_file_path}'")
        print("Please make sure the file exists and the path is correct.")
    except json.JSONDecodeError:
        print(f"Error: Could not parse JSON from file '{json_file_path}'")
        print("Please make sure the file contains valid JSON data.")
    except Exception as e:
        print(f"An error occurred: {e}")