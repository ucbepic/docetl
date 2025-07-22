import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_plans():
    """
    Plot all current plans as dots on a cost vs. F1 score graph, annotating each with its id.
    Frontier plans are blue, non-frontier plans are grey.
    """
    # Data from the table
    data = {
        'id': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        'cost': [0.12, 0.14, 0.07, 0.54, 1.77, 0.22, 3.79, 0.63, 0.12, 0.51, 0.42, 0.13, 6.82],
        'f1': [0.424281, 0.511, 0.335, 0.34, 0.759, 0.324, 0.69, 0.493, 0.362, 0.397, 0.344, 0.449, 0.776],
        'on_frontier': [True, False, True, False, True, False, False, False, True, False, False, False, True]
    }
    
    df = pd.DataFrame(data)
    
    # Separate frontier and non-frontier plans
    frontier_points = df[df['on_frontier'] == True]
    non_frontier_points = df[df['on_frontier'] == False]
    
    # Plot non-frontier plans (grey)
    if not non_frontier_points.empty:
        costs = non_frontier_points['cost'].tolist()
        f1_scores = non_frontier_points['f1'].tolist()
        ids = non_frontier_points['id'].tolist()
        plt.scatter(costs, f1_scores, color='grey', label='Off Frontier')
        for x, y, label in zip(costs, f1_scores, ids):
            plt.annotate(str(label), (x, y), textcoords="offset points", xytext=(5,5), ha='left', fontsize=9, color='grey')
    
    # Plot frontier plans (blue)
    if not frontier_points.empty:
        costs = frontier_points['cost'].tolist()
        f1_scores = frontier_points['f1'].tolist()
        ids = frontier_points['id'].tolist()
        plt.scatter(costs, f1_scores, color='blue', label='Frontier')
        for x, y, label in zip(costs, f1_scores, ids):
            plt.annotate(str(label), (x, y), textcoords="offset points", xytext=(5,5), ha='left', fontsize=9, color='blue')
    
    # Set axis limits to match the iteration plot
    plt.xlim(0, 12)  # Adjust x-axis to accommodate the higher cost values from iteration data
    plt.ylim(0, 0.8)  # Adjust y-axis to accommodate the F1 range from iteration data
    
    plt.xlabel('Cost')
    plt.ylabel('F1 Score')
    plt.title('Plans: Cost vs. F1 Score')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Call the function to generate the plot
plot_plans()