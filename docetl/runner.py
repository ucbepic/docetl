import json
import os
from typing import Dict, List, Optional, Tuple
import time

from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.status import Status

from docetl.operations import get_operation
from docetl.utils import load_config
from docetl.operations.utils import flush_cache

load_dotenv()


class DSLRunner:
    """
    A class for executing Domain-Specific Language (DSL) configurations.

    This class is responsible for loading, validating, and executing DSL configurations
    defined in YAML files. It manages datasets, executes pipeline steps, and tracks
    the cost of operations.

    Attributes:
        config (Dict): The loaded configuration from the YAML file.
        default_model (str): The default language model to use for operations.
        max_threads (int): Maximum number of threads for parallel processing.
        console (Console): Rich console for output formatting.
        datasets (Dict): Storage for loaded datasets.
    """

    def __init__(self, yaml_file: str, max_threads: int = None):
        """
        Initialize the DSLRunner with a YAML configuration file.

        Args:
            yaml_file (str): Path to the YAML configuration file.
            max_threads (int, optional): Maximum number of threads to use. Defaults to None.
        """
        self.config = load_config(yaml_file)
        self.default_model = self.config.get("default_model", "gpt-4o-mini")
        self.max_threads = max_threads or (os.cpu_count() or 1) * 4
        self.console = Console()
        self.status = None
        self.datasets = {}

        self.intermediate_dir = self.config["pipeline"]["output"].get(
            "intermediate_dir"
        )

        # Check if output path is correctly formatted as JSON
        output_path = self.config.get("pipeline", {}).get("output", {}).get("path")
        if output_path:
            if not output_path.lower().endswith(".json"):
                raise ValueError(
                    f"Output path '{output_path}' is not a JSON file. Please provide a path ending with '.json'."
                )
        else:
            raise ValueError(
                "No output path specified in the configuration. Please provide an output path ending with '.json' in the configuration."
            )

        self.syntax_check()

    def syntax_check(self):
        """
        Perform a syntax check on all operations defined in the configuration.

        This method validates each operation by attempting to instantiate it.
        If any operation fails to instantiate, a ValueError is raised.

        Raises:
            ValueError: If any operation fails the syntax check.
        """
        self.console.log(
            "[yellow]Performing syntax check on all operations...[/yellow]"
        )

        for operation_config in self.config["operations"]:
            operation = operation_config["name"]
            operation_type = operation_config["type"]

            try:
                operation_class = get_operation(operation_type)
                operation_class(
                    operation_config,
                    self.default_model,
                    self.max_threads,
                    self.console,
                )
            except Exception as e:
                raise ValueError(
                    f"Syntax check failed for operation '{operation}': {str(e)}"
                )

        self.console.log("[green]Syntax check passed for all operations.[/green]")

    def find_operation(self, op_name: str) -> Dict:
        for operation_config in self.config["operations"]:
            if operation_config["name"] == op_name:
                return operation_config
        raise ValueError(f"Operation '{op_name}' not found in configuration.")

    def run(self) -> float:
        """
        Execute the entire pipeline defined in the configuration.

        This method loads datasets, executes each step in the pipeline, saves the output,
        and returns the total cost of execution.

        Returns:
            float: The total cost of executing the pipeline.
        """
        start_time = time.time()
        self.load_datasets()
        total_cost = 0
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            for step in self.config["pipeline"]["steps"]:
                step_name = step["name"]
                step_task = progress.add_task(
                    f"Running step [cyan]{step_name}[/cyan]...", total=1
                )
                input_data = self.datasets[step["input"]] if "input" in step else None
                output_data, step_cost = self.execute_step(step, input_data, progress)
                self.datasets[step_name] = output_data
                flush_cache(self.console)
                total_cost += step_cost
                progress.update(
                    step_task,
                    advance=1,
                    description=f"Step [cyan]{step_name}[/cyan] completed. Cost: [green]${step_cost:.2f}[/green]",
                )

        self.save_output(self.datasets[self.config["pipeline"]["steps"][-1]["name"]])
        self.console.log(f"[bold green]Total cost: [green]${total_cost:.2f}[/green]")
        self.console.log(
            f"[bold green]Total time: [green]{time.time() - start_time:.2f} seconds[/green]"
        )

        return total_cost

    def load_datasets(self):
        """
        Load all datasets defined in the configuration.

        This method reads datasets from files and stores them in the `datasets` attribute.

        Raises:
            ValueError: If an unsupported dataset type is encountered.
        """
        for name, dataset_config in self.config["datasets"].items():
            if dataset_config["type"] == "file":
                with open(dataset_config["path"], "r") as file:
                    self.datasets[name] = json.load(file)
                    self.datasets[name] = self.datasets[name]
            else:
                raise ValueError(f"Unsupported dataset type: {dataset_config['type']}")

    def save_output(self, data: List[Dict]):
        """
        Save the final output of the pipeline.

        Args:
            data (List[Dict]): The data to be saved.

        Raises:
            ValueError: If an unsupported output type is specified in the configuration.
        """
        output_config = self.config["pipeline"]["output"]
        if output_config["type"] == "file":
            with open(output_config["path"], "w") as file:
                json.dump(data, file, indent=2)
            self.console.log(
                f"[green italic]💾 Output saved to {output_config['path']}[/green italic]"
            )
        else:
            raise ValueError(f"Unsupported output type: {output_config['type']}")

    def execute_step(
        self, step: Dict, input_data: Optional[List[Dict]], progress: Progress
    ) -> Tuple[List[Dict], float]:
        """
        Execute a single step in the pipeline.

        This method runs all operations defined for a step, updating the progress
        and calculating the cost.

        Args:
            step (Dict): The step configuration.
            input_data (Optional[List[Dict]]): Input data for the step.
            progress (Progress): Progress tracker for rich output.

        Returns:
            Tuple[List[Dict], float]: A tuple containing the output data and the total cost of the step.
        """
        total_cost = 0
        for operation in step["operations"]:
            if isinstance(operation, dict):
                operation_name = list(operation.keys())[0]
                operation_config = self.find_operation(operation_name)
            else:
                operation_name = operation
                operation_config = {}

            op_object = self.find_operation(operation_name).copy()
            op_object.update(operation_config)

            # If sample is set, sample the input data
            if op_object.get("sample"):
                input_data = input_data[: op_object["sample"]]

            op_task = progress.add_task(
                f"Running operation [cyan]{operation_name}[/cyan]...", total=1
            )
            self.console.log("[bold]Running Operation:[/bold]")
            self.console.log(f"  Type: [cyan]{op_object['type']}[/cyan]")
            self.console.log(f"  Name: [cyan]{op_object.get('name', 'Unnamed')}[/cyan]")

            operation_class = get_operation(op_object["type"])
            operation_instance = operation_class(
                op_object, self.default_model, self.max_threads, self.console
            )
            if op_object["type"] == "equijoin":
                left_data = self.datasets[op_object["left"]]
                right_data = self.datasets[op_object["right"]]
                input_data, cost = operation_instance.execute(left_data, right_data)
            else:
                input_data, cost = operation_instance.execute(input_data)
            total_cost += cost
            progress.update(
                op_task,
                advance=1,
                description=f"Operation [cyan]{operation_name}[/cyan] completed. Cost: [green]${cost:.2f}[/green]",
            )

            # Checkpoint after each operation
            if self.intermediate_dir:
                self._save_checkpoint(step["name"], operation_name, input_data)

        return input_data, total_cost

    def _save_checkpoint(self, step_name: str, operation_name: str, data: List[Dict]):
        """
        Save a checkpoint of the current data after an operation.

        This method creates a JSON file containing the current state of the data
        after an operation has been executed. The checkpoint is saved in a directory
        structure that reflects the step and operation names.

        Args:
            step_name (str): The name of the current step in the pipeline.
            operation_name (str): The name of the operation that was just executed.
            data (List[Dict]): The current state of the data to be checkpointed.

        Note:
            The checkpoint is saved only if a checkpoint directory has been specified
            when initializing the DSLRunner.
        """
        checkpoint_path = os.path.join(
            self.intermediate_dir, step_name, f"{operation_name}.json"
        )
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        with open(checkpoint_path, "w") as f:
            json.dump(data, f)
        self.console.log(
            f"[green]Intermediate saved for operation '{operation_name}' in step '{step_name}' at {checkpoint_path}[/green]"
        )


if __name__ == "__main__":
    runner = DSLRunner("workloads/medical/map_opt.yaml")
    runner.run()
