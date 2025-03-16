import json
import os
import time
import argparse
from typing import Dict, Any, List, Optional, Tuple
import logging
from pathlib import Path
import concurrent.futures
from functools import partial

from dotenv import load_dotenv
import yaml
import traceback

from docetl.operations.map import ParallelMapOperation, MapOperation
from docetl.runner import DSLRunner

# Configure logging
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("agenticpreprint/testrewriteability/evaluation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables (API keys, etc.)
load_dotenv()

# Constants
PLANS_DIR = "agenticpreprint/testrewriteability/proj_synth_plans"
RESULTS_DIR = "agenticpreprint/testrewriteability/results"
CUAD_DATA_PATH = "agenticpreprint/cuad/raw.json"
CONFIG_PATH = "agenticpreprint/testrewriteability/base_cuad.yaml"
NUM_SAMPLES = 50
DEFAULT_MODEL = "gpt-4o-mini"
MAX_THREADS = 64
PARALLEL_PLANS = 5  # Number of plans to process in parallel


def load_config(config_path: str) -> Dict[str, Any]:
    """Load the YAML configuration file."""
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Error loading config from {config_path}: {str(e)}")
        raise


def load_data(data_path: str, num_samples: int) -> List[Dict[str, Any]]:
    """Load data from the specified path and sample the first n entries."""
    try:
        with open(data_path, "r") as f:
            data = json.load(f)
        
        # Ensure each data item has a 'document' field as expected by the extractor
        processed_data = []
        for i, item in enumerate(data[:num_samples]):
            if isinstance(item, dict) and "document" in item:
                processed_data.append(item)
            else:
                # Convert to expected format if needed
                processed_data.append({"document": item})
                
        return processed_data
    except Exception as e:
        logger.error(f"Error loading data from {data_path}: {str(e)}")
        raise


def get_plan_files(plans_dir: str, limit: Optional[int] = None, start_idx: int = 0) -> List[str]:
    """
    Get a list of plan files to evaluate.
    
    Args:
        plans_dir: Directory containing the plan files
        limit: Optional limit on the number of files to return PER PLAN TYPE
        start_idx: Index to start from when limiting files (applied to each type)
        
    Returns:
        List of plan file names sorted by type and then by number
    """
    try:
        # Group files by type
        parallel_files = []
        chain_files = []
        glean_files = []
        
        # Collect all matching files and sort them into types
        for f in os.listdir(plans_dir):
            if not f.endswith('.json'):
                continue
                
            if f.startswith('parallel_plan_'):
                parallel_files.append(f)
            elif f.startswith('chain_plan_'):
                chain_files.append(f)
            elif f.startswith('glean_plan_'):
                glean_files.append(f)
        
        # Helper function to sort files by their number
        def sort_by_number(filename):
            try:
                # Extract the number from the filename (e.g., "parallel_plan_5.json" -> 5)
                num = int(filename.split('_')[-1].split('.')[0])
                return num
            except ValueError:
                return 0
        
        # Sort each type by number
        parallel_files.sort(key=sort_by_number)
        chain_files.sort(key=sort_by_number)
        glean_files.sort(key=sort_by_number)
        
        # Apply limit and start index to each type
        if limit is not None:
            parallel_files = parallel_files[start_idx:start_idx + limit]
            chain_files = chain_files[start_idx:start_idx + limit]
            glean_files = glean_files[start_idx:start_idx + limit]
        
        # Combine in a consistent order: parallel first, then chain, then glean
        all_files = parallel_files + chain_files + glean_files
        
        logger.info(f"Selected {len(parallel_files)} parallel plans, {len(chain_files)} chain plans, and {len(glean_files)} gleaning plans")
        
        return all_files
        
    except Exception as e:
        logger.error(f"Error getting plan files from {plans_dir}: {str(e)}")
        raise


def load_plan(plan_path: str) -> Dict[str, Any]:
    """Load a plan configuration from a file."""
    try:
        with open(plan_path, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading plan from {plan_path}: {str(e)}")
        raise


def save_results(results: List[Dict[str, Any]], plan_name: str, cost: float, runtime: float) -> None:
    """Save the results to a JSON file."""
    try:
        output_path = os.path.join(RESULTS_DIR, f"{plan_name}.json")
        
        # Create a result object with metadata
        result_obj = {
            "results": results,
            "metadata": {
                "plan_name": plan_name,
                "cost": cost,
                "runtime": runtime,
                "timestamp": time.time(),
                "num_samples": len(results)
            }
        }
        
        with open(output_path, "w") as f:
            json.dump(result_obj, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")
    except Exception as e:
        logger.error(f"Error saving results for {plan_name}: {str(e)}")
        raise


def execute_plan(
    plan_config: Dict[str, Any], 
    input_data: List[Dict[str, Any]], 
    api_wrapper: DSLRunner
) -> Tuple[List[Dict[str, Any]], float]:
    """Execute a plan on the input data."""
    
    # Handle different plan configurations
    # If it's a two-step plan (parallel_map with transform)
    if "parallel_map_with_transform" in plan_config:
        logger.info("Executing parallel_map_with_transform plan")
        parallel_op_config = plan_config["parallel_map_with_transform"][0]
        transform_op_config = plan_config["parallel_map_with_transform"][1]
        
        # First run the parallel map
        parallel_operation = ParallelMapOperation(
            api_wrapper, parallel_op_config, DEFAULT_MODEL, MAX_THREADS
        )
        intermediate_results, cost1 = parallel_operation.execute(input_data)
        logger.info(f"Parallel stage completed with cost ${cost1:.6f}")
        
        # Then run the transform operation
        transform_operation = MapOperation(
            api_wrapper, transform_op_config, DEFAULT_MODEL, MAX_THREADS
        )
        final_results, cost2 = transform_operation.execute(intermediate_results)
        logger.info(f"Transform stage completed with cost ${cost2:.6f}")
        
        return final_results, cost1 + cost2
    
    # For chain plans
    elif any(key.startswith("chain_") for key in plan_config.keys()):
        logger.info("Executing chain plan")
        # Find the entry in the dict
        chain_key = next(iter(plan_config))
        chain_configs = plan_config[chain_key]
        
        if not isinstance(chain_configs, list):
            chain_configs = [chain_configs]
            
        current_data = input_data
        total_cost = 0
        
        # Execute each operation in the chain
        for i, op_config in enumerate(chain_configs):
            logger.info(f"Executing chain step {i+1}/{len(chain_configs)}")
            if op_config["type"] == "map":
                operation = MapOperation(
                    api_wrapper, op_config, DEFAULT_MODEL, MAX_THREADS
                )
            elif op_config["type"] == "parallel_map":
                operation = ParallelMapOperation(
                    api_wrapper, op_config, DEFAULT_MODEL, MAX_THREADS
                )
            else:
                raise ValueError(f"Unsupported operation type: {op_config['type']}")
                
            current_data, cost = operation.execute(current_data)
            total_cost += cost
            logger.info(f"Chain step {i+1} completed with cost ${cost:.6f}")
            
        return current_data, total_cost
    
    # For gleaning plans (check for both formats)
    elif any(key.startswith("gleaning_") for key in plan_config.keys()):
        logger.info("Executing gleaning plan")
        # Find the operation configuration in the gleaning plan
        gleaning_key = next(k for k in plan_config.keys() if k.startswith("gleaning_"))
        gleaning_configs = plan_config[gleaning_key]
        
        if not isinstance(gleaning_configs, list):
            gleaning_configs = [gleaning_configs]
            
        # Most gleaning plans have just one operation 
        op_config = gleaning_configs[0]
        
        operation = MapOperation(
            api_wrapper, op_config, DEFAULT_MODEL, MAX_THREADS
        )
        results, cost = operation.execute(input_data)
        logger.info(f"Gleaning operation completed with cost ${cost:.6f}")
        return results, cost
    
    # For any other single operation
    else:
        logger.info("Executing single operation plan")
        # Find the operation configuration
        op_key = next(iter(plan_config))
        op_config = plan_config[op_key]
        
        if isinstance(op_config, list):
            op_config = op_config[0]  # Take the first if it's a list
            
        if op_config["type"] == "map":
            logger.info("Executing map operation")
            operation = MapOperation(
                api_wrapper, op_config, DEFAULT_MODEL, MAX_THREADS
            )
        elif op_config["type"] == "parallel_map":
            logger.info("Executing parallel_map operation")
            operation = ParallelMapOperation(
                api_wrapper, op_config, DEFAULT_MODEL, MAX_THREADS
            )
        else:
            raise ValueError(f"Unsupported operation type: {op_config['type']}")
            
        results, cost = operation.execute(input_data)
        logger.info(f"Operation completed with cost ${cost:.6f}")
        return results, cost


def process_plan(
    plan_file: str, 
    data: List[Dict[str, Any]], 
    config: Dict[str, Any]
) -> Tuple[str, bool]:
    """
    Process a single plan file.
    
    Args:
        plan_file: Name of the plan file to process
        data: Input data to run the plan on
        config: Configuration for the DSLRunner
        
    Returns:
        Tuple of (plan_name, success_flag)
    """
    plan_name = os.path.splitext(plan_file)[0]
    plan_path = os.path.join(PLANS_DIR, plan_file)
    
    # Create a dedicated API wrapper for this plan
    api_wrapper = DSLRunner(config, max_threads=MAX_THREADS)
    
    # Check if results already exist
    result_path = os.path.join(RESULTS_DIR, f"{plan_name}.json")
    if os.path.exists(result_path):
        logger.info(f"Results for {plan_name} already exist, skipping")
        return plan_name, True
    
    logger.info(f"\n{'='*50}")
    logger.info(f"Evaluating plan: {plan_name}")
    logger.info(f"{'='*50}")
    
    try:
        # Load plan
        plan_config = load_plan(plan_path)
        
        # Track execution time
        start_time = time.time()
        
        # Execute plan
        results, cost = execute_plan(plan_config, data, api_wrapper)
        
        # Calculate runtime
        runtime = time.time() - start_time
        
        logger.info(f"Plan executed successfully in {runtime:.2f}s")
        logger.info(f"Total cost: ${cost:.6f}")
        
        # Save results
        save_results(results, plan_name, cost, runtime)
        
        return plan_name, True
    except Exception as e:
        logger.error(f"Error executing plan {plan_name}: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Save error information
        try:
            error_path = os.path.join(RESULTS_DIR, f"{plan_name}_error.json")
            with open(error_path, "w") as f:
                json.dump({
                    "error": str(e),
                    "plan_name": plan_name,
                    "traceback": traceback.format_exc()
                }, f, indent=2)
            logger.info(f"Error information saved to {error_path}")
        except Exception as save_err:
            logger.error(f"Error saving error information: {str(save_err)}")
        
        return plan_name, False


def process_plans_in_parallel(
    plan_files: List[str],
    data: List[Dict[str, Any]],
    config: Dict[str, Any],
    max_workers: int = PARALLEL_PLANS
) -> Dict[str, bool]:
    """
    Process multiple plans in parallel using a process pool.
    
    Args:
        plan_files: List of plan files to process
        data: Input data to run the plans on
        config: Configuration for the DSLRunner
        max_workers: Maximum number of parallel workers
        
    Returns:
        Dictionary mapping plan names to success flags
    """
    results = {}
    
    # Create a process pool to run plans in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Create a partial function with fixed arguments
        process_func = partial(process_plan, data=data, config=config)
        
        # Submit all tasks
        future_to_plan = {executor.submit(process_func, plan_file): plan_file for plan_file in plan_files}
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_plan):
            plan_file = future_to_plan[future]
            try:
                plan_name, success = future.result()
                results[plan_name] = success
                if success:
                    logger.info(f"Successfully processed plan: {plan_name}")
                else:
                    logger.error(f"Failed to process plan: {plan_name}")
            except Exception as e:
                logger.error(f"Exception processing plan {plan_file}: {str(e)}")
                logger.error(traceback.format_exc())
                results[os.path.splitext(plan_file)[0]] = False
    
    return results


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate projection synthesis plans")
    parser.add_argument(
        "--limit", 
        type=int, 
        default=None, 
        help="Limit the number of plans to evaluate"
    )
    parser.add_argument(
        "--start-index", 
        type=int, 
        default=0, 
        help="Start from this index when limiting plans"
    )
    parser.add_argument(
        "--plan-type", 
        type=str, 
        choices=["parallel", "chain", "glean", "all"], 
        default="all", 
        help="Type of plans to evaluate"
    )
    parser.add_argument(
        "--plan-number", 
        type=int, 
        default=None, 
        help="Specific plan number to evaluate (e.g., 5 for parallel_plan_5)"
    )
    parser.add_argument(
        "--parallel", 
        type=int, 
        default=PARALLEL_PLANS, 
        help=f"Number of plans to process in parallel (default: {PARALLEL_PLANS})"
    )
    return parser.parse_args()


def main():
    """Main function to evaluate all plans."""
    # Parse command-line arguments
    args = parse_arguments()
    
    logger.info("Starting plan evaluation")
    
    # Create results directory if it doesn't exist
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    try:
        # Load configuration
        logger.info(f"Loading configuration from {CONFIG_PATH}")
        config = load_config(CONFIG_PATH)
        
        # Load data
        logger.info(f"Loading data from {CUAD_DATA_PATH}...")
        data = load_data(CUAD_DATA_PATH, NUM_SAMPLES)
        logger.info(f"Loaded {len(data)} samples")
        
        # Get plan files
        plan_files = get_plan_files(PLANS_DIR, args.limit, args.start_index)
        
        # Filter by plan type if specified
        if args.plan_type != "all":
            prefix = f"{args.plan_type}_plan_"
            plan_files = [f for f in plan_files if f.startswith(prefix)]
            
        # Filter by plan number if specified
        if args.plan_number is not None:
            if args.plan_type == "all":
                # Try to find any plan with this number
                plan_files = [
                    f for f in plan_files if 
                    f.endswith(f"_{args.plan_number}.json") or 
                    f.endswith(f"-{args.plan_number}.json")
                ]
            else:
                # Look for a specific plan type and number
                specific_plan = f"{args.plan_type}_plan_{args.plan_number}.json"
                plan_files = [f for f in plan_files if f == specific_plan]
        
        logger.info(f"Found {len(plan_files)} plan files to evaluate")
        
        # Process plans in parallel
        num_workers = min(args.parallel, len(plan_files))
        logger.info(f"Processing plans with {num_workers} parallel workers")
        
        results = process_plans_in_parallel(
            plan_files=plan_files,
            data=data,
            config=config,
            max_workers=num_workers
        )
        
        # Calculate summary statistics
        successful_plans = sum(1 for success in results.values() if success)
        failed_plans = sum(1 for success in results.values() if not success)
        
        # Log summary
        logger.info(f"\n{'='*50}")
        logger.info(f"Evaluation complete")
        logger.info(f"Successful plans: {successful_plans}")
        logger.info(f"Failed plans: {failed_plans}")
        logger.info(f"{'='*50}")
        
    except Exception as e:
        logger.error(f"Fatal error in main execution: {str(e)}")
        logger.error(traceback.format_exc())
        

if __name__ == "__main__":
    main()
