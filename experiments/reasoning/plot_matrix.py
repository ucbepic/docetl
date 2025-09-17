import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Create the cost savings matrix data - EXACTLY as in original
methods = ['simple_baseline', 'baseline', 'mcts', 'lotus', 'pz_direct', 'pz_retrieval']

# Raw data matrix - exactly matching the original format
raw_data = [
    ['--', -22.03, 'Unable', 'None', 'None', -22.23],
    ['Unable', '--', 'Unable', 'None', 'None', 'Unable'],
    [18.32, -0.23, '--', 'None', 'None', -0.44],
    ['Unable', 'Unable', 'Unable', '--', 'None', 'Unable'],
    ['Unable', 'Unable', 'Unable', 'None', '--', 'Unable'],
    ['Unable', 0.20, 'Unable', 'None', 'None', '--']
]

# Create DataFrame
df = pd.DataFrame(raw_data, index=methods, columns=methods)

print("Original Matrix:")
print("=" * 80)
print(df.to_string())
print("=" * 80)

# Find ALL numeric values and calculate color intensity
numeric_positions = []
numeric_values = []
for i in range(len(methods)):
    for j in range(len(methods)):
        val = df.iloc[i, j]
        if isinstance(val, (int, float)) and val != '--':
            numeric_positions.append((i, j, val))
            numeric_values.append(abs(val))
            print(f"NUMERIC: Row {i} ({methods[i]}) -> Col {j} ({methods[j]}) = {val}")

print(f"\nTotal numeric values found: {len(numeric_positions)}")

# Calculate max absolute value for normalization
max_abs_value = max(numeric_values) if numeric_values else 1
print(f"Max absolute value: {max_abs_value}")

def get_color_intensity(value, max_val):
    """Calculate color intensity with more vibrant gradient"""
    intensity = abs(value) / max_val
    # Use a moderate curve for balanced colors
    intensity = intensity ** 0.6  # Moderate curve
    # Allow more color intensity while keeping it elegant
    return max(0.2, min(0.8, intensity))

# Create visualization
fig, ax = plt.subplots(figsize=(12, 10))

# More vibrant but still elegant color scheme
base_colors = {
    'positive': np.array([74, 222, 128]) / 255,      # Fresh green
    'negative': np.array([248, 113, 113]) / 255,    # Warm red
    'diagonal': np.array([249, 250, 251]) / 255,    # Very light gray
    'unable': np.array([243, 244, 246]) / 255,      # Light gray
    'none': np.array([243, 244, 246]) / 255         # Light gray
}

# Draw each cell individually
for i in range(len(methods)):
    for j in range(len(methods)):
        val = df.iloc[i, j]
        
        # Determine cell color and intensity
        if isinstance(val, (int, float)):
            intensity = get_color_intensity(val, max_abs_value)
            
            if val > 0:
                # Vibrant green gradient
                base_color = base_colors['positive']
                # Create more vibrant gradient
                color = base_color * (0.4 + 0.6 * intensity)
                # Add less white tint for more color
                color = color + (1 - color) * (1 - intensity) * 0.3
                text_color = 'black'  # Always black for better readability
            else:
                # Vibrant red gradient
                base_color = base_colors['negative']
                # Create more vibrant gradient
                color = base_color * (0.4 + 0.6 * intensity)
                # Add less white tint for more color
                color = color + (1 - color) * (1 - intensity) * 0.3
                text_color = 'black'  # Always black for better readability
            text = f'{val:.2f}'
            
        elif val == '--':
            color = base_colors['diagonal']
            text_color = 'gray'
            text = '--'
        elif val == 'Unable':
            color = base_colors['unable']
            text_color = 'gray'
            text = 'Unable'
        elif val == 'None':
            color = base_colors['none']
            text_color = 'gray'
            text = 'None'
        else:
            color = 'white'
            text_color = 'black'
            text = str(val)
        
        # Draw rectangle with subtle border
        rect = plt.Rectangle((j, i), 1, 1, facecolor=color, edgecolor='white', linewidth=1.5)
        ax.add_patch(rect)
        
        # Add text with larger font
        fontweight = 'bold' if isinstance(val, (int, float)) else 'normal'
        fontsize = 14 if isinstance(val, (int, float)) else 12
        ax.text(j + 0.5, i + 0.5, text, ha='center', va='center', 
                fontsize=fontsize, fontweight=fontweight, color=text_color)

# Set up the plot
ax.set_xlim(0, len(methods))
ax.set_ylim(0, len(methods))
ax.set_aspect('equal')

# Set ticks and labels with larger font
ax.set_xticks(np.arange(len(methods)) + 0.5)
ax.set_yticks(np.arange(len(methods)) + 0.5)
ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=13)
ax.set_yticklabels(methods, fontsize=13)

# Invert y-axis to match matrix convention
ax.invert_yaxis()

# Remove outer frame
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# Title with larger font
plt.title('Cost Savings Matrix', fontsize=24, fontweight='bold', pad=30, color='#1f2937')

plt.tight_layout()
plt.show()

# Summary statistics
numeric_vals = [val for i, j, val in numeric_positions]
print(f"\nSummary Statistics:")
print(f"Maximum cost savings: {max(numeric_vals):.2f}")
print(f"Maximum cost increase: {min(numeric_vals):.2f}")
print(f"Mean cost change: {np.mean(numeric_vals):.2f}")

# Show the exact numeric comparisons with color intensity
print(f"\nNumeric comparisons with color intensity:")
for i, j, val in numeric_positions:
    intensity = get_color_intensity(val, max_abs_value)
    color_type = "Green" if val > 0 else "Pink"
    print(f"  {methods[i]} -> {methods[j]}: {val:.2f} ({color_type}, intensity: {intensity:.2f})")

# Export to CSV
df.to_csv('cost_savings_matrix.csv')
print(f"\nMatrix exported to 'cost_savings_matrix.csv'")

# Create tab-separated format for Google Docs
print("\nTab-separated format for Google Docs:")
print("-" * 80)
header = "Method\t" + "\t".join(methods)
print(header)
for i, method in enumerate(methods):
    row = method + "\t" + "\t".join([str(df.iloc[i, j]) for j in range(len(methods))])
    print(row)
print("-" * 80)