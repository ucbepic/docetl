import matplotlib.pyplot as plt
import numpy as np

# Data from the table (organized by rows for line plotting)
data_by_row = {
    'row1': {'accuracy': [0.424281, 0.510667, 0.526197, 0.526197, 0.526197, 0.526197], 
             'cost': [0.12, 0.56, 0.24, 0.24, 0.24, 0.24]},
    'row2': {'accuracy': [None, 0.474173, 0.548675, 0.585296, 0.585296, 0.585296], 
             'cost': [None, 0.57, 0.16, 0.14, 0.14, 0.14]},
    'row3': {'accuracy': [None, 0.364597, 0.364597, 0.364597, 0.364597, 0.364597], 
             'cost': [None, 0.14, 0.14, 0.14, 0.14, 0.14]},
    'row4': {'accuracy': [None, 0.553681, 0.553681, 0.553681, 0.553681, 0.553681], 
             'cost': [None, 0.15, 0.15, 0.15, 0.15, 0.15]},
    'row5': {'accuracy': [None, 0.05106, 0.449565, 0.449565, 0.449565, 0.449565], 
             'cost': [None, 0.24, 0.2, 0.2, 0.2, 0.2]}
}

# Define colors for each iteration (lighter overall with bigger depth differences)
iteration_colors = ['#87CEEB', '#4682B4', '#1E90FF', '#0066CC', '#003d7a', '#001122']

# Define lighter, more distinct colors for lines
line_colors = ['#FFB6C1', '#98FB98', '#87CEEB', '#DDA0DD', '#F0E68C']

# Create figure with subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# ========== GRAPH 1: WITHOUT LINES ==========
ax1.set_title('Cost vs Accuracy by Iteration (Without Lines)', fontsize=14, fontweight='bold')

# Plot scatter points for each iteration
for iteration in range(6):
    accuracy_points = []
    cost_points = []
    
    for row_data in data_by_row.values():
        if row_data['accuracy'][iteration] is not None:
            accuracy_points.append(row_data['accuracy'][iteration])
            cost_points.append(row_data['cost'][iteration])
    
    ax1.scatter(accuracy_points, cost_points, 
               color=iteration_colors[iteration], 
               s=120, 
               alpha=0.9,
               edgecolors='white',
               linewidth=1.5,
               label=f'Iteration {iteration}')

ax1.set_xlabel('Accuracy (F1 Score)', fontsize=12)
ax1.set_ylabel('Cost', fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax1.set_xlim(0, 0.7)
ax1.set_ylim(0, 0.7)

# ========== GRAPH 2: WITH LINES ==========
ax2.set_title('Cost vs Accuracy by Iteration (With Lines)', fontsize=14, fontweight='bold')

# Plot lines for each row (from iteration 1 to 5, or 0 to 5 for row1)
for i, (row_name, row_data) in enumerate(data_by_row.items()):
    # Get data from iteration 1 to 5 (skip iteration 0 except for row1)
    if row_name == 'row1':
        # For row1, start from iteration 0
        x_line = row_data['accuracy']
        y_line = row_data['cost']
        iterations = list(range(6))
    else:
        # For other rows, start from iteration 1
        x_line = row_data['accuracy'][1:]  # Skip None value
        y_line = row_data['cost'][1:]      # Skip None value
        iterations = list(range(1, 6))
    
    # Plot the line
    ax2.plot(x_line, y_line, 
             color=line_colors[i], 
             linewidth=1, 
             alpha=0.7,
             linestyle='-',
             label=f'Row {i+1}')

# Plot scatter points for each iteration
for iteration in range(6):
    accuracy_points = []
    cost_points = []
    
    for row_data in data_by_row.values():
        if row_data['accuracy'][iteration] is not None:
            accuracy_points.append(row_data['accuracy'][iteration])
            cost_points.append(row_data['cost'][iteration])
    
    ax2.scatter(accuracy_points, cost_points, 
               color=iteration_colors[iteration], 
               s=120, 
               alpha=0.9,
               edgecolors='white',
               linewidth=1.5,
               label=f'Iteration {iteration}',
               zorder=5)  # Ensure points are on top of lines

ax2.set_xlabel('Accuracy (F1 Score)', fontsize=12)
ax2.set_ylabel('Cost', fontsize=12)
ax2.grid(True, alpha=0.3)

# Create custom legend for the second plot
iteration_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=iteration_colors[i], 
                               markersize=8, markeredgecolor='white', markeredgewidth=1.5, 
                               label=f'Iteration {i}') for i in range(6)]
line_handles = [plt.Line2D([0], [0], color=line_colors[i], linewidth=2, 
                          label=f'Row {i+1}') for i in range(5)]

ax2.legend(handles=iteration_handles + line_handles, bbox_to_anchor=(1.05, 1), loc='upper left')
ax2.set_xlim(0, 0.7)
ax2.set_ylim(0, 0.7)

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plots
plt.show()

# Optional: Save the plots
# plt.savefig('cost_accuracy_comparison.png', dpi=300, bbox_inches='tight')

print("Two graphs generated successfully!")
print("\nLeft graph: Scatter plot without lines")
print("Right graph: Scatter plot with lines connecting each row across iterations")
print("\nLine colors are lighter and more distinct:")
print("- Row 1: Light pink")
print("- Row 2: Light green") 
print("- Row 3: Light blue")
print("- Row 4: Light purple")
print("- Row 5: Light yellow")