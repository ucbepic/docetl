from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv
import os
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import print as rprint
from motion.operations import get_operation
from motion.utils import load_config
import json

load_dotenv()


class DSLRunner:
    def __init__(self, yaml_file: str, max_threads: int = None):
        self.config = load_config(yaml_file)
        self.default_model = self.config.get("default_model", "gpt-4o-mini")
        self.max_threads = max_threads or (os.cpu_count() or 1) * 4
        self.console = Console()
        self.datasets = {}
        self.syntax_check()

    def syntax_check(self):
        for operation in self.config["operations"]:
            operation_config = self.config["operations"][operation]
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

        self.console.print("[green]Syntax check passed for all operations.[/green]")

    def run(self) -> float:
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
                total_cost += step_cost
                progress.update(
                    step_task,
                    advance=1,
                    description=f"Step [cyan]{step_name}[/cyan] completed. Cost: [green]${step_cost:.2f}[/green]",
                )

        self.save_output(self.datasets[self.config["pipeline"]["steps"][-1]["name"]])
        rprint(f"[bold green]Total cost: [green]${total_cost:.2f}[/green]")

        return total_cost

    def load_datasets(self):
        for name, dataset_config in self.config["datasets"].items():
            if dataset_config["type"] == "file":
                with open(dataset_config["path"], "r") as file:
                    self.datasets[name] = json.load(file)
            else:
                raise ValueError(f"Unsupported dataset type: {dataset_config['type']}")

    def save_output(self, data: List[Dict]):
        output_config = self.config["pipeline"]["output"]
        if output_config["type"] == "file":
            with open(output_config["path"], "w") as file:
                json.dump(data, file, indent=2)
        else:
            raise ValueError(f"Unsupported output type: {output_config['type']}")

    def execute_step(
        self, step: Dict, input_data: Optional[List[Dict]], progress: Progress
    ) -> Tuple[List[Dict], float]:
        total_cost = 0
        for operation in step["operations"]:
            if isinstance(operation, dict):
                operation_name = list(operation.keys())[0]
                operation_config = operation[operation_name]
            else:
                operation_name = operation
                operation_config = {}

            op_object = self.config["operations"][operation_name].copy()
            op_object.update(operation_config)
            op_object["name"] = operation_name

            op_task = progress.add_task(
                f"Running operation [cyan]{operation_name}[/cyan]...", total=1
            )
            self.console.print(f"[bold]Running Operation:[/bold]")
            self.console.print(f"  Type: [cyan]{op_object['type']}[/cyan]")
            self.console.print(
                f"  Name: [cyan]{op_object.get('name', 'Unnamed')}[/cyan]"
            )

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

        return input_data, total_cost


if __name__ == "__main__":
    runner = DSLRunner("workloads/medical/equijoin.yaml")
    runner.run()
