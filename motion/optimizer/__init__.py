from motion.executor import Operation
from typing import List, Tuple, Any, Dict
from motion.optimizer.mapper import glean_map, decompose_parallel_flatmap
from motion.operators import LLMMapper
from motion.executor.apply_operation import process_operation_results
from rich import print as rprint
from rich.table import Table
from copy import deepcopy

ENABLED_OPTIMIZERS = {"LLMMapper": ["decompose_parallel_flatmap", "gleaning"]}


def optimize(
    operation: Operation,
    dataset_size: int,
    sample_data: List[Tuple[Any, Any]],
    initial_results: List[Tuple[Any, Any]],
    initial_errors: List[Tuple[str, Any, int]],
    num_workers: int,
    base_cost: float,
) -> Operation:
    """Optimize the given operator."""
    initial_accuracy = 1 - (len(initial_errors) / len(initial_results))
    rprint(
        f"[bold]Initial estimated cost[/bold] for [cyan]{operation.operator.__class__.__name__}[/cyan]: "
        f"[green]${base_cost * (dataset_size / len(sample_data)):.2f}[/green]; "
        f"[bold]accuracy[/bold]: [yellow]{initial_accuracy:.2%}[/yellow]"
    )

    all_results: Dict[str, Dict[str, Any]] = {
        "original": {
            "operations": [operation],
            "results": initial_results,
            "cost": base_cost * (dataset_size / len(sample_data)),
            "accuracy": initial_accuracy,
        }
    }

    if isinstance(operation.operator, LLMMapper):
        if "gleaning" in ENABLED_OPTIMIZERS["LLMMapper"]:
            plans = glean_map(
                deepcopy(operation),
                deepcopy(sample_data),
                deepcopy(initial_results),
                deepcopy(initial_errors),
                num_workers,
                base_cost,
            )

            for i, plan in enumerate(plans):
                dataset_cost = plan["sample_cost"] * (dataset_size / len(sample_data))

                all_results[f"gleaning_round_{plan['rounds']}"] = {
                    "operations": plan["operations"],
                    "results": plan["results"],
                    "cost": dataset_cost,
                    "accuracy": plan["expected_accuracy"],
                    "rounds": plan["rounds"],
                }

                rprint(
                    f"[bold]Expected cost with gleaning (round {plan['rounds']})[/bold]: ${dataset_cost:.2f}, "
                    f"[bold]Expected accuracy with gleaning (round {plan['rounds']})[/bold]: {plan['expected_accuracy']:.2%}"
                )

            # Find the plan with the highest accuracy
            best_gleaning_plan = max(plans, key=lambda x: x["expected_accuracy"])
            rprint(
                f"[bold]Number of gleaning rounds for maximal accuracy[/bold]: [cyan]{best_gleaning_plan['rounds']}[/cyan]"
            )

        if "decompose_parallel_flatmap" in ENABLED_OPTIMIZERS["LLMMapper"]:
            (
                flatmap_operations,
                flatmap_results,
                flatmap_errors,
                flatmap_cost,
                flatmap_accuracy,
            ) = decompose_parallel_flatmap(
                deepcopy(operation),
                deepcopy(sample_data),
                deepcopy(initial_results),
                deepcopy(initial_errors),
                num_workers,
            )
            dataset_cost = flatmap_cost * (dataset_size / len(sample_data))

            all_results["decomposed_parallel_flatmap"] = {
                "operations": flatmap_operations,
                "results": flatmap_results,
                "cost": dataset_cost,
                "accuracy": flatmap_accuracy,
            }

            rprint(
                f"[bold]Expected cost with decomposed parallel flatmap[/bold]: ${dataset_cost:.2f}, "
                f"[bold]Expected accuracy with decomposed parallel flatmap[/bold]: {flatmap_accuracy:.2%}"
            )

    # Print results table
    table = Table(
        title=f"Optimization Results for {operation.operator.__class__.__name__}"
    )
    table.add_column("Strategy", style="cyan")
    table.add_column("Cost", style="green")
    table.add_column("Accuracy", style="yellow")

    for strategy, result in all_results.items():
        table.add_row(strategy, f"${result['cost']:.2f}", f"{result['accuracy']:.2%}")

    rprint(table)

    # Choose the best strategy based on accuracy
    best_strategy = max(all_results, key=lambda x: all_results[x]["accuracy"])
    best_result = all_results[best_strategy]

    rprint(
        f"[bold green]Choosing the best plan: {best_strategy} with accuracy: {best_result['accuracy']:.2%} and cost: ${best_result['cost']:.2f}[/bold green]"
    )

    if "gleaning" in best_strategy:
        for op in best_result["operations"]:
            op.operator.optimal_rounds = best_result["rounds"]

    return best_result["operations"], process_operation_results(
        best_result["results"], [], operation
    )
