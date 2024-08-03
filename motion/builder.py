from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
import copy
import yaml
from typing import Dict, List, Any, Optional, Tuple, Union
from motion.operations import get_operation
from motion.optimizers.map_optimizer import MapOptimizer
from motion.optimizers.reduce_optimizer import ReduceOptimizer
from motion.optimizers.resolve_optimizer import ResolveOptimizer
from motion.utils import load_config
from rich.console import Console
from rich.table import Table
import random
import json
from litellm import completion, completion_cost
import os
import jinja2
from jinja2 import Environment, meta
import re
from tqdm import tqdm


def extract_jinja_variables(template_string):
    # Create a Jinja2 environment
    env = Environment()

    # Parse the template
    ast = env.parse(template_string)

    # Find all the variables referenced in the template
    variables = meta.find_undeclared_variables(ast)

    # Use regex to find any additional variables that might be missed
    # This regex looks for {{ variable }} patterns
    regex_variables = set(re.findall(r"{{\s*(\w+)\s*}}", template_string))

    # Combine both sets of variables
    all_variables = variables.union(regex_variables)

    return list(all_variables)


SUPPORTED_OPS = ["map", "resolve", "reduce"]


class LLMClient:
    def __init__(self, model="gpt-4o"):
        self.model = model
        self.total_cost = 0

    def generate(self, messages, system_prompt, parameters):
        response = completion(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                *messages,
            ],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "write_output",
                        "description": "Write output to a database",
                        "parameters": parameters,
                    },
                }
            ],
            parallel_tool_calls=False,
            tool_choice={"type": "function", "function": {"name": "write_output"}},
        )
        cost = completion_cost(response)
        self.total_cost += cost
        return response


class Optimizer:
    def __init__(
        self,
        yaml_file: str,
        max_threads: Optional[int] = None,
        model: str = "gpt-4o",
        timeout: int = 60,
    ):
        self.yaml_file_path = yaml_file
        self.config = load_config(yaml_file)
        self.sample_size = {
            "reduce": 40,
            "map": 10,
            "resolve": 20,
        }
        self.console = Console()
        self.optimized_config = self.config.copy()
        self.llm_client = LLMClient(model)
        self.max_threads = max_threads or (os.cpu_count() or 1) * 4
        self.operations_cost = 0
        self.timeout = timeout

        self.print_optimizer_config()

    def print_optimizer_config(self):
        self.console.print("[bold cyan]Optimizer Configuration:[/bold cyan]")
        self.console.print("─" * 40)
        self.console.print(f"[yellow]YAML File:[/yellow] {self.yaml_file_path}")
        self.console.print(f"[yellow]Sample Size:[/yellow] {self.sample_size}")
        self.console.print(f"[yellow]Max Threads:[/yellow] {self.max_threads}")
        self.console.print(f"[yellow]Model:[/yellow] {self.llm_client.model}")
        self.console.print(f"[yellow]Timeout:[/yellow] {self.timeout} seconds")
        self.console.print("─" * 40)

    def optimize(self):
        optimized_steps = []
        optimized_operations = {}
        for step in self.config["pipeline"]["steps"]:
            optimized_step, optimized_operations = self._optimize_step(step)
            optimized_steps.append(optimized_step)
            optimized_operations.update(optimized_operations)

        self.optimized_config["operations"] = optimized_operations
        self.optimized_config["pipeline"]["steps"] = optimized_steps
        self._save_optimized_config()

        self.console.print(
            f"[bold]Total agent cost: ${self.llm_client.total_cost:.2f}[/bold]"
        )
        self.console.print(
            f"[bold]Total operations cost: ${self.operations_cost:.2f}[/bold]"
        )
        self.console.print(
            f"[bold]Total cost: ${self.llm_client.total_cost + self.operations_cost:.2f}[/bold]"
        )

    def _optimize_step(
        self, step: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        optimized_operations = {}
        input_data = None

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

            if input_data is None:
                input_data = self._get_sample_data(step.get("input"), op_object)

            if (
                op_object.get("optimize", True) == False
                or op_object.get("type") not in SUPPORTED_OPS
            ):
                # If optimize is False or operation type is not supported, just run the operation without optimization
                # Use rich console status to indicate running the operation
                with self.console.status(
                    f"[bold green]Running operation: {operation_name} (Type: {op_object['type']})[/bold green]"
                ):
                    # Print the number of elements in input_data
                    self.console.print(f"[yellow]Running Operation:[/yellow]")
                    self.console.print(f"[yellow]  Type: {op_object['type']}[/yellow]")
                    self.console.print(f"[yellow]  Name: {operation_name}[/yellow]")
                    self.console.print(
                        f"[yellow]  Sample size: {len(input_data)}[/yellow]"
                    )
                    input_data = self._run_operation(op_object, input_data)
                    optimized_operations[operation_name] = op_object
            else:
                # Use rich console status to indicate optimization of the operation
                with self.console.status(
                    f"[bold blue]Optimizing operation: {operation_name} (Type: {op_object['type']})[/bold blue]"
                ):
                    # Print the number of elements in input_data
                    self.console.print(f"[yellow]Optimizing Operation:[/yellow]")
                    self.console.print(f"[yellow]  Type: {op_object['type']}[/yellow]")
                    self.console.print(f"[yellow]  Name: {operation_name}[/yellow]")
                    self.console.print(
                        f"[yellow]  Sample size: {len(input_data)}[/yellow]"
                    )
                    if op_object.get("type") == "map":
                        optimized_ops, input_data = self._optimize_map(
                            op_object, input_data
                        )
                    elif op_object.get("type") == "reduce":
                        optimized_ops, input_data = self._optimize_reduce(
                            op_object, input_data
                        )
                    elif op_object.get("type") == "resolve":
                        optimized_ops, input_data = self._optimize_resolve(
                            op_object, input_data
                        )
                    else:
                        raise ValueError(
                            f"Unsupported operation type: {op_object['type']}"
                        )

                    for op in optimized_ops:
                        op_name = op.pop("name")
                        optimized_operations[op_name] = op

        optimized_step = step.copy()
        optimized_step["operations"] = list(optimized_operations.keys())
        return optimized_step, optimized_operations

    def _get_sample_data(
        self, dataset_name: str, op_config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        if dataset_name is None:
            return []

        dataset = self.config["datasets"][dataset_name]
        if dataset["type"] == "file":
            with open(dataset["path"], "r") as f:
                data = json.load(f)

            if op_config.get("type") == "reduce":
                return self._get_reduce_sample(data, op_config.get("reduce_key"))
            else:
                return random.sample(
                    data, min(self.sample_size[op_config.get("type")], len(data))
                )
        else:
            raise ValueError(f"Unsupported dataset type: {dataset['type']}")

    def _get_reduce_sample(
        self, data: List[Dict[str, Any]], reduce_key: str
    ) -> List[Dict[str, Any]]:
        # Group data by reduce key
        grouped_data = defaultdict(list)
        for item in data:
            grouped_data[item[reduce_key]].append(item)

        # Sort groups by size in descending order
        sorted_groups = sorted(
            grouped_data.items(), key=lambda x: len(x[1]), reverse=True
        )

        sample = []

        # Take the top 5 groups
        top_5_groups = sorted_groups[:5]

        # Calculate the total count of items in the top 5 groups
        total_count = sum(len(items) for _, items in top_5_groups)

        sample = []
        for _, items in top_5_groups:
            # Calculate the proportion of items to sample from this group
            group_proportion = len(items) / total_count
            group_sample_size = int(self.sample_size["reduce"] * group_proportion)

            # Sample from the group
            group_sample = random.sample(items, min(group_sample_size, len(items)))
            sample.extend(group_sample)

        # If we haven't reached the desired sample size, add more items randomly
        if len(sample) < self.sample_size["reduce"]:
            remaining_items = [
                item
                for _, items in top_5_groups
                for item in items
                if item not in sample
            ]
            additional_sample = random.sample(
                remaining_items, self.sample_size["reduce"] - len(sample)
            )
            sample.extend(additional_sample)

        # Create a histogram of group sizes
        group_sizes = [len(items) for _, items in grouped_data.items()]
        size_counts = Counter(group_sizes)

        # Sort the sizes for a more readable output
        sorted_sizes = sorted(size_counts.items())

        # Print the histogram
        self.console.print("\n[bold]Histogram of Group Sizes:[/bold]")
        max_bar_width, max_count = 2, max(size_counts.values())
        for size, count in sorted_sizes[:5]:
            normalized_count = int(count / max_count * max_bar_width)
            bar = "█" * normalized_count
            self.console.print(f"{size:3d}: {bar} ({count})")
        self.console.print("\n")

        return sample

    def _optimize_reduce(
        self, op_config: Dict[str, Any], input_data: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        reduce_optimizer = ReduceOptimizer(
            self.config,
            self.console,
            self.llm_client,
            self.max_threads,
            self._run_operation,
        )
        optimized_op, input_data = reduce_optimizer.optimize(op_config, input_data)
        return [optimized_op], input_data

    def _optimize_resolve(
        self, op_config: Dict[str, Any], input_data: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        optimized_config, cost = ResolveOptimizer(
            self.config, op_config, self.console, self.llm_client, self.max_threads
        ).optimize(input_data)
        self.operations_cost += cost

        # Update the operation config with the optimized values
        op_config.update(optimized_config)

        # Run the optimized operation
        output_data = self._run_operation(op_config, input_data)

        return [op_config], output_data

    def _optimize_map(
        self, op_config: Dict[str, Any], input_data: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        # Execute the original operation on the sample data
        output_data = self._run_operation(op_config, input_data)

        map_optimizer = MapOptimizer(
            self.config,
            self.console,
            self.llm_client,
            self.max_threads,
            self._run_operation,
        )

        # Generate custom validator prompt
        validator_prompt = map_optimizer._generate_validator_prompt(
            op_config, input_data, output_data
        )

        # Step 2: Use the validator prompt to assess the operation's performance
        assessment = map_optimizer._assess_operation(
            op_config, input_data, output_data, validator_prompt
        )

        # Print out the assessment
        self.console.print(
            f"[bold]Assessment for whether we should improve operation {op_config['name']}:[/bold]"
        )
        self.console.print(json.dumps(assessment, indent=2))
        self.console.print("\n")  # Add a newline for better readability

        # Check if improvement is needed based on the assessment
        if assessment.get("needs_improvement", True) == False:
            self.console.print(
                f"[green]No improvement needed for operation {op_config['name']}[/green]"
            )
            return [op_config], output_data

        # Generate improved prompt plan
        improved_prompt_plan = map_optimizer._get_improved_prompt(
            op_config, assessment, input_data
        )

        # Generate chunk size plans
        chunk_size_plans = map_optimizer._generate_chunk_size_plans(
            op_config, input_data
        )

        # Evaluate both plans
        plans_to_evaluate = {
            "improved_prompt_plan": improved_prompt_plan,
            **chunk_size_plans,
        }
        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(
                    self._evaluate_plan,
                    plan_name,
                    op_config,
                    plan,
                    copy.deepcopy(input_data),
                    validator_prompt,
                    num_evaluations=min(5, len(input_data)),
                ): plan_name
                for plan_name, plan in plans_to_evaluate.items()
            }
            results = {}
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Evaluating plans",
            ):
                plan_name = futures[future]
                score, output = future.result(timeout=self.timeout)
                results[plan_name] = (score, output)

        # Create a table of scores sorted in descending order
        scores = sorted(
            [(score, plan) for plan, (score, _) in results.items()], reverse=True
        )

        self.console.print(
            f"\n[bold]Score Distribution for {op_config['name']} ({op_config['type']}, {len(scores)} plans):[/bold]"
        )
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Plan", style="dim", width=30)
        table.add_column("Score", justify="right")
        for score, plan in scores:
            table.add_row(plan, f"{score:.2f}")

        self.console.print(table)
        self.console.print("\n")

        # Choose the best plan
        best_plan_name = max(results, key=lambda x: results[x][0])
        _, best_output = results[best_plan_name]
        self.console.print(
            f"[green]Choosing {best_plan_name} for operation {op_config['name']}[/green]"
        )
        return plans_to_evaluate[best_plan_name], best_output

    def _evaluate_plan(
        self,
        plan_name: str,
        op_config: Dict[str, Any],
        plan: Union[Dict[str, Any], List[Dict[str, Any]]],
        input_data: List[Dict[str, Any]],
        validator_prompt: str,
        num_evaluations: int,
    ) -> Tuple[float, List[Dict[str, Any]]]:
        if isinstance(plan, dict):
            plan = [plan]

        output_data = input_data
        for op in plan:
            output_data = self._run_operation(op, output_data)

        scores = []

        for _ in range(num_evaluations):
            # Evaluate the quality of the output using the custom validator prompt
            quality = self._assess_output_quality(
                op_config, input_data, output_data, validator_prompt
            )
            score_map = {
                "Satisfactory": 4,
                "Mostly Satisfactory": 3,
                "Partially Satisfactory": 2,
                "Unsatisfactory": 1,
            }
            scores.append(score_map.get(quality["quality_category"], 0))

        average_score = sum(scores) / num_evaluations
        # Print the quality assessment for the last evaluation
        # self.console.print(f"[bold]Quality assessment for plan {plan_name}:[/bold]")
        # self.console.print(f"\tCategory: {quality['quality_category']}")
        # self.console.print(f"\tReason: {quality['reason']}")
        # self.console.print(f"\tAverage Score: {average_score:.2f}")
        # self.console.print("\n")  # Add a newline for better readability

        return average_score, output_data

    def _assess_output_quality(
        self,
        op_config: Dict[str, Any],
        input_data: List[Dict[str, Any]],
        output_data: List[Dict[str, Any]],
        validator_prompt: str,
    ) -> str:
        system_prompt = "You are an AI assistant tasked with evaluating the quality of data processing outputs."
        output_schema_keys = op_config["output"]["schema"].keys()
        random_idx = random.randint(0, len(input_data) - 1)
        document_id = input_data[random_idx]["document_id"]
        input_elem = input_data[random_idx]
        output_elem = [
            item for item in output_data if item["document_id"] == document_id
        ][0]
        output_elem = {key: output_elem[key] for key in output_schema_keys}

        prompt = f"""
        {validator_prompt}

        Input and Output Data Sample:
        {json.dumps({"input": input_elem, "output": output_elem}, indent=2)}

        Based on the validator prompt and the input-output data samples, assess the quality of the output.
        Categorize the quality into one of these four categories:
        1. "Unsatisfactory": The output failed to meet the validator prompt requirements.
        2. "Partially Satisfactory": The output met some of the validator prompt requirements but not all.
        3. "Mostly Satisfactory": The output mostly met the validator prompt requirements but has some room for improvement.
        3. "Satisfactory": The output fully met the validator prompt requirements.

        Provide your response in the following format:
        """

        parameters = {
            "type": "object",
            "properties": {
                "quality_category": {
                    "type": "string",
                    "enum": [
                        "Unsatisfactory",
                        "Partially Satisfactory",
                        "Satisfactory",
                    ],
                },
                "reason": {"type": "string"},
            },
            "required": ["quality_category", "reason"],
        }

        response = self.llm_client.generate(
            [
                {"role": "user", "content": prompt},
            ],
            system_prompt,
            parameters,
        )
        result = json.loads(
            response.choices[0].message.tool_calls[0].function.arguments
        )

        return result

    def _run_operation(
        self, op_config: Dict[str, Any], input_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        operation_class = get_operation(op_config["type"])
        operation_instance = operation_class(
            op_config, self.config["default_model"], self.max_threads, self.console
        )
        output_data, cost = operation_instance.execute(input_data)
        self.operations_cost += cost
        return output_data

    def _save_optimized_config(self):
        # Create a copy of the optimized config to modify
        config_to_save = self.optimized_config.copy()

        # Recursively resolve all anchors and aliases
        def resolve_anchors(data):
            if isinstance(data, dict):
                return {k: resolve_anchors(v) for k, v in data.items()}
            elif isinstance(data, list):
                return [resolve_anchors(item) for item in data]
            else:
                return data

        resolved_config = resolve_anchors(config_to_save)

        # Use safe_dump to avoid creating anchors and aliases
        # Get the base filename without extension
        base_filename = os.path.splitext(self.yaml_file_path)[0]

        # Append '_opt' to the base filename
        optimized_filename = f"{base_filename}_opt.yaml"

        with open(optimized_filename, "w") as f:
            yaml.safe_dump(resolved_config, f, default_flow_style=False)

        self.console.print(
            f"[green italic]Optimized config saved to {optimized_filename}[/green italic]"
        )


if __name__ == "__main__":
    optimizer = Optimizer("workloads/medical/reduce.yaml", model="gpt-4o")
    optimizer.optimize()
