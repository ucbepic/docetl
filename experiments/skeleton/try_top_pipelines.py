import argparse
import glob
import json
import os
# import subprocess # No longer needed
import tempfile
import time # Added for timing
from pathlib import Path
from typing import Any, Dict, List, Tuple # Added Tuple

import yaml
from rich.console import Console # Added for potential rich output from runner

# --- Add DSLRunner import ---
from docetl.runner import DSLRunner

# Constants
BASE_DIR: Path = Path(__file__).parent
LOG_DIR_SUFFIX: str = ".optimizer_logs"
YAML_SUFFIX: str = ".yaml"
RESULTS_PREFIX: str = "strategy_results_"
RESULTS_SUFFIX: str = ".json"
UCB_STRATEGY_KEY: str = "ucb"
TOP_N: int = 10


def find_latest_results_file(log_dir: Path) -> Path | None:
    """Finds the most recent strategy results JSON file in a directory."""
    search_pattern: str = str(log_dir / f"{RESULTS_PREFIX}*{RESULTS_SUFFIX}")
    result_files: List[str] = glob.glob(search_pattern)
    if not result_files:
        print(f"Error: No results files found matching '{search_pattern}'")
        return None
    # Sort by modification time, newest first
    latest_file: str = max(result_files, key=os.path.getmtime)
    print(f"Found latest results file: {latest_file}")
    return Path(latest_file)


def load_json_file(file_path: Path) -> Any | None:
    """Loads data from a JSON file."""
    try:
        with file_path.open("r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from file: {file_path}")
        return None
    except Exception as e:
        print(f"Error loading JSON file {file_path}: {e}")
        return None


def load_yaml_file(file_path: Path) -> Any | None:
    """Loads data from a YAML file."""
    try:
        with file_path.open("r") as f:
            # Use SafeLoader for security
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return None
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file {file_path}: {e}")
        return None
    except Exception as e:
        print(f"Error loading YAML file {file_path}: {e}")
        return None


# --- Replace the previous complex function with this simpler one ---
def run_pipeline_and_get_stats(config_path: Path, console: Console) -> Tuple[float | None, float | None]:
    """
    Runs the pipeline defined in the config file using DSLRunner.load_run_save()
    and returns the total cost and execution time.

    Args:
        config_path: Path to the temporary YAML configuration file.
        console: Rich console instance.

    Returns:
        A tuple containing (total_cost, runtime_seconds), or (None, None) if execution fails.
    """
    total_cost: float | None = None # Initialize as None
    runtime_seconds: float | None = None
    runner: DSLRunner | None = None
    start_time: float | None = None

    try:
        print(f"Instantiating DSLRunner for config: {config_path}")
        # Instantiate the runner directly
        runner = DSLRunner.from_yaml(str(config_path), max_threads=None, console=console)

        start_time = time.time()

        # Execute the entire pipeline using the intended method
        print("Executing runner.load_run_save()...")
        # Assuming load_run_save handles loading, running steps, and saving
        # We don't need the return value (final_docs) here, but we run it.
        runner.load_run_save()

        end_time = time.time()
        runtime_seconds = end_time - start_time

        # Attempt to retrieve the total cost from the runner instance
        # Common attribute names might be 'total_cost' or similar.
        # Adjust this based on the actual attribute name in DSLRunner.
        if hasattr(runner, 'total_cost'):
             total_cost = runner.total_cost
             print(f"Pipeline execution finished. Total Cost: ${total_cost:.6f}, Runtime: {runtime_seconds:.2f}s")
        else:
             print(f"Pipeline execution finished. Runtime: {runtime_seconds:.2f}s. Cost attribute not found on runner.")
             # Keep total_cost as None if attribute doesn't exist

        return total_cost, runtime_seconds

    except Exception as e:
        print(f"[bold red]Error during pipeline execution with DSLRunner: {e}[/bold red]")
        # Ensure timer stops even if error occurs mid-execution
        if start_time:
             end_time = time.time()
             runtime_seconds = end_time - start_time
             print(f"Execution aborted after {runtime_seconds:.2f}s")
        # Cost is unknown if execution failed partway through
        return None, runtime_seconds


def main() -> None:
    """Main function to load results and run top pipelines."""
    parser = argparse.ArgumentParser(
        description="Run top N pipelines from optimizer results."
    )
    parser.add_argument(
        "--config-name",
        required=True,
        choices=["map_test", "map_summary_test"],
        help="Name of the base configuration (e.g., 'map_test').",
    )
    # Add new argument for specific pipeline ranks
    parser.add_argument(
        "--ranks",
        type=int,
        nargs="+",
        help="Specific pipeline ranks to run (e.g., --ranks 1 8 9). If not specified, runs top N pipelines.",
    )
    # Add a new CLI parameter to make running the original pipeline optional
    parser.add_argument(
        "--include-original",
        action="store_true",
        help="Include the original pipeline run even when specific ranks are requested."
    )
    args = parser.parse_args()

    config_name: str = args.config_name
    specific_ranks: List[int] | None = args.ranks
    log_dir: Path = BASE_DIR / f"{config_name}{LOG_DIR_SUFFIX}"
    base_config_path: Path = BASE_DIR / f"{config_name}{YAML_SUFFIX}"

    # Initialize console for rich output
    console = Console()
    # Dictionary to store stats
    pipeline_stats: Dict[str, Dict[str, float | None]] = {}

    # 1. Find the latest results file
    latest_results_file: Path | None = find_latest_results_file(log_dir)
    if not latest_results_file:
        return

    # 2. Load the results JSON
    results_data: Any | None = load_json_file(latest_results_file)
    if not results_data:
        return

    # 3. Extract UCB strategy data
    if UCB_STRATEGY_KEY not in results_data:
        print(f"Error: Strategy '{UCB_STRATEGY_KEY}' not found in results file.")
        return
    ucb_strategy_data: Dict[str, Any] = results_data[UCB_STRATEGY_KEY]
    if "ranked_results" not in ucb_strategy_data:
        print(f"Error: 'ranked_results' key not found for strategy '{UCB_STRATEGY_KEY}'.")
        return
    ucb_ranked_results: List[Any] = ucb_strategy_data["ranked_results"]


    # 4. Select pipelines based on specific ranks or default to top N
    top_pipelines: List[List[Dict[str, Any]]] = []
    
    # Print the total number of available pipelines
    print(f"Total available pipelines: {len(ucb_ranked_results)}")
    
    # Pipeline selection logic - either specific ranks or top N
    if specific_ranks:
        print(f"Using specific pipeline ranks: {specific_ranks}")
        # Adjust ranks to 0-based indexing
        ranks_to_use = [rank - 1 for rank in specific_ranks if rank > 0 and rank <= len(ucb_ranked_results)]
        
        # Check if any requested ranks are out of bounds
        invalid_ranks = [rank for rank in specific_ranks if rank <= 0 or rank > len(ucb_ranked_results)]
        if invalid_ranks:
            print(f"Warning: Ranks {invalid_ranks} are out of bounds (valid range: 1-{len(ucb_ranked_results)})")
        
        # Extract pipelines for the requested ranks
        for idx in ranks_to_use:
            result_item = ucb_ranked_results[idx]
            if isinstance(result_item, (list, tuple)) and len(result_item) > 0 and isinstance(result_item[0], list):
                pipeline_config: List[Dict[str, Any]] = result_item[0]
                if all(isinstance(op, dict) and "name" in op for op in pipeline_config):
                    top_pipelines.append(pipeline_config)
                else:
                    print(f"Warning: Skipping rank {idx+1} due to invalid pipeline config format: {pipeline_config}")
            else:
                print(f"Warning: Skipping rank {idx+1} due to unexpected format: {result_item}")
    else:
        print(f"Using default top {TOP_N} pipelines")
        # Default behavior - select top N pipelines
        for i, result_item in enumerate(ucb_ranked_results):
            if i >= TOP_N: break
            if isinstance(result_item, (list, tuple)) and len(result_item) > 0 and isinstance(result_item[0], list):
                pipeline_config: List[Dict[str, Any]] = result_item[0]
                if all(isinstance(op, dict) and "name" in op for op in pipeline_config):
                    top_pipelines.append(pipeline_config)
                else:
                    print(f"Warning: Skipping result {i+1} due to invalid pipeline config format within the list: {pipeline_config}")
            else:
                print(f"Warning: Skipping result {i+1} due to unexpected format in ranked_results: {result_item}")
    
    if not top_pipelines:
        print(f"Error: No valid pipelines extracted from 'ranked_results' for strategy '{UCB_STRATEGY_KEY}'.")
        return
    print(f"Found {len(top_pipelines)} pipelines to run from strategy '{UCB_STRATEGY_KEY}'.")


    # 5. Load the base YAML config
    base_config: Dict[str, Any] | None = load_yaml_file(base_config_path)
    if not base_config:
        return

    # --- Determine and create the base output directory for all runs ---
    original_path_obj: Path | None = None # Define for broader scope
    output_target_dir: Path | None = None # Define for broader scope
    try:
        original_pipeline_output_path_str = base_config["pipeline"]["output"]["path"]
        original_path_obj = Path(original_pipeline_output_path_str)
        original_parent_dir = original_path_obj.parent
        original_stem = original_path_obj.stem
        output_target_dir = original_parent_dir / original_stem
        os.makedirs(output_target_dir, exist_ok=True)
        print(f"Base output directory set to: {output_target_dir}")
    except (KeyError, TypeError, AttributeError, OSError) as e:
        print(f"[bold red]Error:[/bold red] Could not determine or create base output directory from config path '{base_config.get('pipeline', {}).get('output', {}).get('path', 'N/A')}': {e}")
        return
    if not original_path_obj:
         print("[bold red]Error:[/bold red] Logic error: original_path_obj not set.")
         return


    # --- Determine whether to run original pipeline ---
    run_original = not specific_ranks or args.include_original

    # --- Run the original pipeline (conditionally) ---
    if run_original:
        print("\n--- Running Original Pipeline ---")
        temp_config_path_orig: Path | None = None # Define for finally block
        try:
            original_config_copy = yaml.safe_load(yaml.safe_dump(base_config))
            original_output_filename = f"original{original_path_obj.suffix}"
            original_output_path = output_target_dir / original_output_filename

            original_config_copy["pipeline"]["output"]["path"] = str(original_output_path)
            print(f"Original pipeline output path set to: {original_output_path}")

            # Save to temp file
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", delete=False, encoding="utf-8"
            ) as temp_config_file:
                yaml.safe_dump(original_config_copy, temp_config_file)
                temp_config_path_str: str = temp_config_file.name
            temp_config_path_orig = Path(temp_config_path_str)

            # --- Use the new function ---
            cost, runtime = run_pipeline_and_get_stats(temp_config_path_orig, console)
            pipeline_stats[original_output_filename] = {"cost": cost, "runtime": runtime}

        except Exception as e:
            print(f"[bold red]Error:[/bold red] Failed to prepare or run the original pipeline: {e}")
        finally:
            # Ensure temporary file is deleted
            if temp_config_path_orig and temp_config_path_orig.exists():
                try:
                    os.unlink(temp_config_path_orig)
                    print(f"Deleted temporary config file for original run: {temp_config_path_orig}")
                except OSError as e:
                    print(f"Warning: Could not delete temporary file {temp_config_path_orig}: {e}")
    else:
        print("\n--- Skipping Original Pipeline (specific ranks requested without --include-original) ---")


    # 6. Iterate, modify config, and run each selected pipeline
    for i, pipeline_ops in enumerate(top_pipelines):
        # Find the actual rank in the original results for this pipeline
        if specific_ranks and i < len(specific_ranks):
            # When using specific ranks, use the actual requested rank
            rank = specific_ranks[i]
        else:
            # In default mode, rank is position + 1
            rank = i + 1
            
        print(f"\n--- Running Pipeline Rank {rank} ---")
        temp_config_path: Path | None = None # Define for finally block
        modified_pipeline_output_path: Path | None = None # Define for stats key

        try:
            modified_config: Dict[str, Any] = yaml.safe_load(yaml.safe_dump(base_config))

            # Prepare new operations section as a LIST
            new_operations: List[Dict[str, Any]] = []
            operation_names: List[str] = []
            for op_def in pipeline_ops:
                if not isinstance(op_def, dict) or "name" not in op_def:
                    print(f"Warning: Invalid operation definition in rank {rank}: {op_def}. Skipping pipeline.")
                    operation_names = []
                    break
                op_name: str = op_def["name"]
                new_operations.append(op_def)
                operation_names.append(op_name)
            if not operation_names:
                continue

            modified_config["operations"] = new_operations
            print(f"Updated top-level 'operations' section with a list of {len(new_operations)} operations.")

            # Replace operation names within the specific pipeline step AND set output path
            pipeline_section = modified_config.get("pipeline")
            if not isinstance(pipeline_section, dict): raise ValueError("Missing 'pipeline' section")
            steps_section = pipeline_section.get("steps")
            if not isinstance(steps_section, list) or not steps_section: raise ValueError("Missing 'pipeline.steps'")
            first_step = steps_section[0]
            if not isinstance(first_step, dict): raise ValueError("First step is not dict")
            if "operations" not in first_step: raise ValueError("Missing 'operations' in first step")
            step_name = first_step.get("name", "Unnamed Step 0")
            first_step["operations"] = operation_names
            print(f"Updated 'operations' list within step '{step_name}'.")

            pipeline_output_section = pipeline_section.get("output")
            if not isinstance(pipeline_output_section, dict): raise ValueError("Missing 'pipeline.output'")
            original_suffix = original_path_obj.suffix
            new_output_filename = f"rank_{rank}{original_suffix}"
            modified_pipeline_output_path = output_target_dir / new_output_filename
            pipeline_output_section["path"] = str(modified_pipeline_output_path)
            print(f"Updated 'pipeline.output.path' to: {modified_pipeline_output_path}")


            # Create temp file
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", delete=False, encoding="utf-8"
            ) as temp_config_file:
                yaml.safe_dump(modified_config, temp_config_file)
                temp_config_path_str: str = temp_config_file.name
            temp_config_path = Path(temp_config_path_str)

            # --- Use the new function ---
            cost, runtime = run_pipeline_and_get_stats(temp_config_path, console)
            if modified_pipeline_output_path:
                 pipeline_stats[modified_pipeline_output_path.name] = {"cost": cost, "runtime": runtime}
            else:
                 print(f"Warning: Could not determine output filename for rank {rank} stats.")


        except (ValueError, KeyError, IndexError) as e:
            print(f"[bold red]Error:[/bold red] Failed to update pipeline structure or output path for rank {rank}: {e}. Skipping.")
            continue
        except Exception as e:
             print(f"[bold red]Error:[/bold red] Failed to prepare or run pipeline rank {rank}: {e}")
        finally:
            # Ensure temporary file is deleted
            if temp_config_path and temp_config_path.exists():
                try:
                    os.unlink(temp_config_path)
                    print(f"Deleted temporary config file: {temp_config_path}")
                except OSError as e:
                    print(f"Warning: Could not delete temporary file {temp_config_path}: {e}")

    # --- Load existing stats if available ---
    def load_existing_stats(stats_file_path: Path) -> Dict[str, Dict[str, float | None]]:
        """Load existing stats from JSON file if it exists."""
        if stats_file_path.exists():
            try:
                with open(stats_file_path, "r") as f:
                    existing_stats = json.load(f)
                    print(f"Loaded existing stats with {len(existing_stats)} entries.")
                    return existing_stats
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Failed to load existing stats file {stats_file_path}: {e}")
                print("Creating new stats file.")
        return {}

    # Before initializing the empty pipeline_stats dictionary, load existing stats
    if output_target_dir:
        stats_file_path = output_target_dir / "stats.json"
        pipeline_stats = load_existing_stats(stats_file_path)
    else:
        pipeline_stats = {}

    # Then at the end when writing stats:
    if output_target_dir:
        stats_file_path = output_target_dir / "stats.json"
        try:
            print(f"\nWriting execution stats to: {stats_file_path}")
            with open(stats_file_path, "w") as f:
                json.dump(pipeline_stats, f, indent=4)
            print(f"Stats successfully written with {len(pipeline_stats)} entries.")
        except Exception as e:
            print(f"[bold red]Error writing stats file {stats_file_path}: {e}[/bold red]")
    else:
        print("[bold yellow]Warning:[/bold yellow] Output target directory not set, cannot write stats.json.")


if __name__ == "__main__":
    main()
