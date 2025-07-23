import matplotlib.pyplot as plt


def plot_plans():
    """
    Plot all current plans as dots on a cost vs. F1 score graph, annotating each with its group-iteration label.
    Different iterations are colored from light to dark (1 to 5 iterations).
    """
    # Data from the table - organized by groups and iterations
    data = {
        # Group 1 (first two rows)
        "group_1": {
            "iterations": [0, 1, 2, 3, 4, 5],
            "cost": [0.12, 1.78, 4.33, 5.88, 7.14, 8.59],
            "f1": [0.424281, 0.754267, 0.218676, 0.18939, 0.226566, 0.419691],
        },
        # Group 2 (second two rows)
        "group_2": {
            "iterations": [1, 2],
            "cost": [2.1, 2.31],
            "f1": [0.708584, 0.660892],
        },
        # Group 3 (third two rows)
        "group_3": {
            "iterations": [1, 2, 3, 4],
            "cost": [0.14, 0.17, 0.17, 0.42],
            "f1": [0.276697, 0, 0.001876, 0.002079],
        },
        # Group 4 (fourth two rows)
        "group_4": {
            "iterations": [1, 2, 3, 4],
            "cost": [0.14, 0.19, 0.21, 0.81],
            "f1": [0.434194, 0.46699, 0.455202, 0.54082],
        },
        # Group 5 (fifth two rows)
        "group_5": {
            "iterations": [1, 2, 3, 4, 5],
            "cost": [1.82, 3.21, 8.68, 8.85, 11.59],
            "f1": [0.752722, 0.733601, 0.632046, 0.612578, 0.637308],
        },
    }

    # Color map for iterations (light to dark)
    colors = {
        0: "#FF69B4",  # Pink for iteration 0
        1: "#B3D9FF",  # Light blue
        2: "#80BFFF",  # Medium light blue
        3: "#4DA6FF",  # Medium blue
        4: "#1A8CFF",  # Medium dark blue
        5: "#0066CC",  # Dark blue
    }

    # Plot each group
    for group_idx, (group_name, group_data) in enumerate(data.items(), 1):
        iterations = group_data["iterations"]
        costs = group_data["cost"]
        f1_scores = group_data["f1"]

        for i, (iteration, cost, f1) in enumerate(zip(iterations, costs, f1_scores)):
            # Create label
            if iteration == 0:
                label = "0"
            else:
                label = f"{group_idx}-{iteration}"

            # Plot point
            plt.scatter(
                cost,
                f1,
                color=colors[iteration],
                s=80,
                alpha=0.8,
                edgecolors="black",
                linewidth=0.5,
            )

            # Add annotation
            plt.annotate(
                label,
                (cost, f1),
                textcoords="offset points",
                xytext=(5, 5),
                ha="left",
                fontsize=9,
                color="black",
            )

    # Create legend for iterations
    legend_elements = []
    for iteration in sorted(colors.keys()):
        if iteration == 0:
            legend_elements.append(
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=colors[iteration],
                    markersize=8,
                    label=f"Iteration {iteration}",
                    markeredgecolor="black",
                    markeredgewidth=0.5,
                )
            )
        else:
            legend_elements.append(
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=colors[iteration],
                    markersize=8,
                    label=f"Iteration {iteration}",
                    markeredgecolor="black",
                    markeredgewidth=0.5,
                )
            )

    plt.xlabel("Cost")
    plt.ylabel("F1 Score")
    plt.title("Plans: Cost vs. F1 Score")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(handles=legend_elements, title="Iterations")
    plt.tight_layout()
    plt.show()


# Call the function to generate the plot
plot_plans()
