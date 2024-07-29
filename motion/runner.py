import yaml
import json
from typing import Dict, List, Any, Optional, Tuple
import tiktoken
from litellm import completion, embedding
from itertools import groupby
from operator import itemgetter
from dotenv import load_dotenv
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm
from rich import print as rprint
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from litellm import completion_cost
from jinja2 import Template
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

load_dotenv()


def convert_val(value: Any) -> Dict[str, Any]:
    value = value.lower()
    if value in ["str", "text", "string", "varchar"]:
        return {"type": "string"}
    elif value in ["int", "integer"]:
        return {"type": "integer"}
    elif value in ["float", "decimal", "number"]:
        return {"type": "number"}
    elif value in ["bool", "boolean"]:
        return {"type": "boolean"}
    elif value.startswith("list["):
        inner_type = value[5:-1].strip()
        return {"type": "array", "items": convert_val(inner_type)}
    elif value == "list":
        raise ValueError("List type must specify its elements, e.g., 'list[str]'")
    else:
        raise ValueError(f"Unsupported value type: {value}")


class DSLRunner:
    def __init__(self, yaml_file: str, max_threads: int = None):
        with open(yaml_file, "r") as file:
            self.config = yaml.safe_load(file)
        self.default_model = self.config.get("default_model", "gpt-4o-mini")
        self.max_threads = max_threads or (os.cpu_count() or 1) * 4
        self.console = Console()
        self.datasets = {}

    def run(self):
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
            input_data, cost = self.execute_operation(op_object, input_data)
            total_cost += cost
            progress.update(
                op_task,
                advance=1,
                description=f"Operation [cyan]{operation_name}[/cyan] completed. Cost: [green]${cost:.2f}[/green]",
            )

        return input_data, total_cost

    def execute_operation(
        self, operation: Dict, input_data: List[Dict]
    ) -> Tuple[List[Dict], float]:
        operation_type = operation["type"]

        self.console.print(f"[bold]Running Operation:[/bold]")
        self.console.print(f"  Type: [cyan]{operation_type}[/cyan]")
        self.console.print(f"  Name: [cyan]{operation.get('name', 'Unnamed')}[/cyan]")

        if operation_type == "map":
            return self.execute_map(operation, input_data)
        elif operation_type == "filter":
            return self.execute_filter(operation, input_data)
        elif operation_type == "explode":
            return self.execute_explode(operation, input_data)
        elif operation_type == "parallel_flatmap":
            return self.execute_parallel_flatmap(operation, input_data)
        elif operation_type == "equijoin":
            left_data = self.datasets[operation["left"]]
            right_data = self.datasets[operation["right"]]
            return self.execute_equijoin(operation, left_data, right_data)
        elif operation_type == "split":
            return self.execute_split(operation, input_data)
        elif operation_type == "reduce":
            return self.execute_reduce(operation, input_data)
        elif operation_type == "resolve":
            return self.execute_resolve(operation, input_data)
        else:
            raise ValueError(f"Unsupported operation type: {operation_type}")

    def execute_explode(
        self, operation: Dict, input_data: List[Dict]
    ) -> Tuple[List[Dict], float]:
        explode_key = operation["explode_key"]
        results = []

        for item in input_data:
            if explode_key not in item:
                raise KeyError(f"Explode key '{explode_key}' not found in item")
            if not isinstance(item[explode_key], (list, tuple, set)):
                raise TypeError(f"Value of explode key '{explode_key}' is not iterable")

            for value in item[explode_key]:
                new_item = item.copy()
                new_item[explode_key] = value
                results.append(new_item)

        return results, 0

    def execute_equijoin(
        self, operation: Dict, left_data: List[Dict], right_data: List[Dict]
    ) -> Tuple[List[Dict], float]:
        left_key = operation["join_key"]["left"]["name"]
        right_key = operation["join_key"]["right"]["name"]
        left_limit = operation["join_key"]["left"].get("limit", float("inf"))
        right_limit = operation["join_key"]["right"].get("limit", float("inf"))
        blocking_threshold = operation.get("blocking_threshold")
        blocking_conditions = operation.get("blocking_conditions", [])
        total_cost = 0

        def is_match(left_item: Dict[str, Any], right_item: Dict[str, Any]) -> bool:
            return any(
                eval(condition, {"left": left_item, "right": right_item})
                for condition in blocking_conditions
            )

        # Initial blocking
        blocked_pairs = []
        for left_item in left_data:
            for right_item in right_data:
                if is_match(left_item, right_item):
                    blocked_pairs.append((left_item, right_item))

        if blocking_threshold is not None:
            embedding_model = operation.get("embedding_model", self.default_model)

            def get_embedding(
                item: Dict[str, Any], keys: List[str]
            ) -> Tuple[List[float], float]:
                text = " ".join(str(item[key]) for key in keys if key in item)
                response = embedding(model=embedding_model, input=[text])
                return response["data"][0]["embedding"], completion_cost(response)

            with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
                left_embeddings = list(
                    executor.map(
                        lambda x: get_embedding(x, [left_key]),
                        left_data,
                    )
                )
                right_embeddings = list(
                    executor.map(
                        lambda x: get_embedding(x, [right_key]),
                        right_data,
                    )
                )

            left_embeddings, left_costs = zip(*left_embeddings)
            right_embeddings, right_costs = zip(*right_embeddings)
            total_cost += sum(left_costs) + sum(right_costs)

            # Additional blocking based on embeddings
            for i, left_item in enumerate(left_data):
                for j, right_item in enumerate(right_data):
                    if (left_item, right_item) not in blocked_pairs:
                        if (
                            cosine_similarity(
                                [left_embeddings[i]], [right_embeddings[j]]
                            )[0][0]
                            >= blocking_threshold
                        ):
                            blocked_pairs.append((left_item, right_item))

        # LLM-based comparison for blocked pairs
        def get_hashable_key(item: Dict) -> str:
            return json.dumps(item, sort_keys=True)

        left_match_counts = defaultdict(int)
        right_match_counts = defaultdict(int)
        results = []
        comparison_costs = 0

        def compare_pair(left_item: Dict, right_item: Dict) -> Tuple[bool, float]:
            prompt_template = Template(operation["comparison_prompt"])
            prompt = prompt_template.render(left=left_item, right=right_item)
            response = self.call_llm(
                operation.get("comparison_model", self.default_model),
                "compare",
                prompt,
                {"matched": "bool"},
            )
            output = self.parse_llm_response(response)[0]
            return output["matched"], completion_cost(response)

        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            future_to_pair = {
                executor.submit(compare_pair, left, right): (left, right)
                for left, right in blocked_pairs
            }

            for future in tqdm(
                as_completed(future_to_pair),
                total=len(future_to_pair),
                desc="Comparing pairs",
            ):
                pair = future_to_pair[future]
                is_match, cost = future.result()
                comparison_costs += cost

                if is_match:
                    joined_item = {}
                    left_item, right_item = pair
                    left_key_hash = get_hashable_key(left_item)
                    right_key_hash = get_hashable_key(right_item)
                    if (
                        left_match_counts[left_key_hash] >= left_limit
                        or right_match_counts[right_key_hash] >= right_limit
                    ):
                        continue

                    for key, value in left_item.items():
                        joined_item[f"{key}_left" if key in right_item else key] = value
                    for key, value in right_item.items():
                        joined_item[f"{key}_right" if key in left_item else key] = value
                    if self.validate_output(operation, joined_item):
                        results.append(joined_item)
                        left_match_counts[left_key_hash] += 1
                        right_match_counts[right_key_hash] += 1

        total_cost += comparison_costs

        # Calculate and print the join selectivity
        join_selectivity = len(results) / (len(left_data) * len(right_data))
        self.console.print(f"Equijoin selectivity: {join_selectivity:.4f}")

        return results, total_cost

    def execute_resolve(
        self, operation: Dict, input_data: List[Dict]
    ) -> Tuple[List[Dict], float]:
        blocking_keys = operation.get("blocking_keys", [])
        blocking_threshold = operation.get("blocking_threshold")
        blocking_conditions = operation.get("blocking_conditions", [])
        total_cost = 0

        def is_match(item1: Dict[str, Any], item2: Dict[str, Any]) -> bool:
            return any(
                eval(condition, {"input1": item1, "input2": item2})
                for condition in blocking_conditions
            )

        # Initial clustering
        clusters = {}
        for i, item in enumerate(input_data):
            matched = False
            for rep, cluster in clusters.items():
                if is_match(item, input_data[rep]):
                    cluster.append(i)
                    matched = True
                    break
            if not matched:
                clusters[i] = [i]

        if blocking_threshold is not None:
            embedding_model = operation.get("embedding_model", self.default_model)

            def get_embedding(item: Dict[str, Any]) -> Tuple[List[float], float]:
                text = " ".join(str(item[key]) for key in blocking_keys if key in item)
                response = embedding(model=embedding_model, input=[text])
                return response["data"][0]["embedding"], completion_cost(response)

            with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
                embeddings = list(executor.map(get_embedding, input_data))
                embeddings, costs = zip(*embeddings)
                total_cost += sum(costs)

            # Additional clustering based on embeddings
            for i, item in enumerate(input_data):
                for rep, cluster in list(clusters.items()):
                    if (
                        i not in cluster
                        and cosine_similarity([embeddings[i]], [embeddings[rep]])[0][0]
                        >= blocking_threshold
                    ):
                        clusters[rep].append(i)

        # Pairwise comparisons within clusters
        true_matches = {}
        pair_costs = 0

        def compare_pair(item1: Dict, item2: Dict) -> Tuple[bool, float]:
            prompt_template = Template(operation["comparison_prompt"])
            prompt = prompt_template.render(input1=item1, input2=item2)
            response = self.call_llm(
                operation.get("comparison_model", self.default_model),
                "compare",
                prompt,
                {"is_match": "bool"},
            )
            output = self.parse_llm_response(response)[0]
            return output["is_match"], completion_cost(response)

        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            future_to_pair = {}
            for cluster in clusters.values():
                cluster_items = [input_data[i] for i in cluster]
                for i, item1 in enumerate(cluster_items):
                    for j, item2 in enumerate(cluster_items):
                        if i < j:
                            future = executor.submit(compare_pair, item1, item2)
                            future_to_pair[future] = (cluster[i], cluster[j])

            total_pairs = len(future_to_pair)
            for future in tqdm(
                as_completed(future_to_pair), total=total_pairs, desc="Comparing pairs"
            ):
                pair = future_to_pair[future]
                is_match, cost = future.result()
                pair_costs += cost
                if is_match:
                    if pair[0] not in true_matches:
                        true_matches[pair[0]] = set()
                    if pair[1] not in true_matches:
                        true_matches[pair[1]] = set()
                    true_matches[pair[0]].add(pair[1])
                    true_matches[pair[1]].add(pair[0])

        # Calculate and print the true match selectivity
        n = len(input_data)
        total_possible_pairs = n * (n - 1) // 2
        true_match_count = sum(len(matches) for matches in true_matches.values()) // 2
        true_match_selectivity = true_match_count / total_possible_pairs
        self.console.print(f"Self-join selectivity: {true_match_selectivity:.4f}")
        total_cost += pair_costs

        # Group true matches into sub-clusters
        sub_clusters = []
        processed = set()
        for i, matches in true_matches.items():
            if i not in processed:
                sub_cluster = {i} | matches
                for j in matches:
                    sub_cluster |= true_matches.get(j, set())
                sub_clusters.append(list(sub_cluster))
                processed |= sub_cluster

        # Add items that didn't match anything as their own clusters
        for i in range(len(input_data)):
            if i not in processed:
                sub_clusters.append([i])

        # Process each sub-cluster
        results = []

        def process_sub_cluster(sub_cluster):
            if len(sub_cluster) > 1:
                true_match_items = [input_data[i] for i in sub_cluster]
                reduction_template = Template(operation["reduction_prompt"])
                reduction_prompt = reduction_template.render(
                    matched_entries=true_match_items
                )
                reduction_response = self.call_llm(
                    operation.get("reduction_model", self.default_model),
                    "reduce",
                    reduction_prompt,
                    operation["output"]["schema"],
                )
                reduction_output = self.parse_llm_response(reduction_response)[0]
                reduction_cost = completion_cost(reduction_response)

                if self.validate_output(operation, reduction_output):
                    return (
                        [
                            {
                                **item,
                                **{
                                    k: reduction_output[k]
                                    for k in operation["output"]["schema"]
                                },
                            }
                            for item in true_match_items
                        ],
                        reduction_cost,
                    )
                return [], reduction_cost
            else:
                return [input_data[sub_cluster[0]]], 0

        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            futures = [
                executor.submit(process_sub_cluster, sub_cluster)
                for sub_cluster in sub_clusters
            ]
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Processing sub-clusters",
                leave=True,
            ):
                sub_results, sub_cost = future.result()
                results.extend(sub_results)
                total_cost += sub_cost

        return results, total_cost

    def execute_map(
        self, operation: Dict, input_data: List[Dict]
    ) -> Tuple[List[Dict], float]:
        def _process_map_item(item: Dict) -> Tuple[Optional[Dict], float]:
            prompt_template = Template(operation["prompt"])
            prompt = prompt_template.render(input=item)
            response = self.call_llm(
                operation.get("model", self.default_model),
                "map",
                prompt,
                operation["output"]["schema"],
            )
            item_cost = completion_cost(response)
            output = self.parse_llm_response(response)[0]
            # Add key-value pairs from item that are not in output_schema
            for key, value in item.items():
                if key not in operation["output"]["schema"]:
                    output[key] = value
            if self.validate_output(operation, output):
                return output, item_cost
            return None, item_cost

        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            futures = [executor.submit(_process_map_item, item) for item in input_data]
            results = []
            total_cost = 0
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Processing map items",
                leave=True,
            ):
                result, item_cost = future.result()
                if result is not None:
                    results.append(result)
                total_cost += item_cost

        return results, total_cost

    def execute_filter(
        self, operation: Dict, input_data: List[Dict]
    ) -> Tuple[List[Dict], float]:
        results = []
        total_cost = 0

        def _process_filter_item(item: Dict) -> Tuple[Optional[Dict], float]:
            prompt_template = Template(operation["prompt"])
            prompt = prompt_template.render(input=item)
            response = self.call_llm(
                operation.get("model", self.default_model),
                "filter",
                prompt,
                operation["output"]["schema"],
            )
            item_cost = completion_cost(response)
            output = self.parse_llm_response(response)[0]
            # Add key-value pairs from item that are not in output_schema
            for key, value in item.items():
                if key not in operation["output"]["schema"]:
                    output[key] = value

            if self.validate_output(operation, output):
                return output, item_cost
            return None, item_cost

        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            futures = [
                executor.submit(_process_filter_item, item) for item in input_data
            ]
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Processing filter items",
                leave=True,
            ):
                result, item_cost = future.result()
                total_cost += item_cost
                if result is not None:
                    results.append(result)

        return results, total_cost

    def execute_parallel_flatmap(
        self, operation: Dict, input_data: List[Dict]
    ) -> Tuple[List[Dict], float]:
        results = []
        total_cost = 0

        def process_prompt(item, prompt_template, model, output_schema):
            prompt = Template(prompt_template).render(input=item)
            response = self.call_llm(
                model,
                "map",
                prompt,
                output_schema,
            )
            output = self.parse_llm_response(response)[0]
            # Add key-value pairs from item that are not in output_schema
            for key, value in item.items():
                if key not in output_schema:
                    output[key] = value
            return output, completion_cost(response)

        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            futures = []
            for item in input_data:
                item_futures = []
                for i, prompt_template in enumerate(operation["prompts"]):
                    future = executor.submit(
                        process_prompt,
                        item,
                        prompt_template,
                        operation["models"][i],
                        operation["output"]["schema"],
                    )
                    item_futures.append(future)
                futures.append(item_futures)

            for item_futures in tqdm(
                futures, desc="Processing parallel flatmap items", leave=True
            ):
                item_results = []
                for future in as_completed(item_futures):
                    output, item_cost = future.result()
                    total_cost += item_cost
                    if self.validate_output(operation, output):
                        item_results.append(output)
                results.extend(item_results)

        return results, total_cost

    def execute_reduce(
        self, operation: Dict, input_data: List[Dict]
    ) -> Tuple[List[Dict], float]:
        reduce_key = operation["reduce_key"]
        sorted_data = sorted(input_data, key=itemgetter(reduce_key))
        grouped_data = groupby(sorted_data, key=itemgetter(reduce_key))

        results = []
        total_cost = 0

        def process_group(key, group):
            group_list = list(group)
            prompt_template = Template(operation["prompt"])
            prompt = prompt_template.render(reduce_key=key, values=group_list)
            response = self.call_llm(
                operation.get("model", self.default_model),
                "reduce",
                prompt,
                operation["output"]["schema"],
            )
            output = self.parse_llm_response(response)[0]
            output[reduce_key] = key
            return output, completion_cost(response)

        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            futures = [
                executor.submit(process_group, key, group)
                for key, group in grouped_data
            ]

            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Processing reduce items",
                leave=True,
            ):
                output, item_cost = future.result()
                total_cost += item_cost
                if self.validate_output(operation, output):
                    results.append(output)

        return results, total_cost

    def execute_split(
        self, operation: Dict, input_data: List[Dict]
    ) -> Tuple[List[Dict], float]:
        """
        EXAMPLE

        split_content:
            type: split
            chunk_size: 1000
            overlap_size: 100
            model: gpt-4o-mini
            split_key: content
        """
        results = []
        encoder = tiktoken.encoding_for_model(
            operation.get("model", self.default_model)
        )
        for item in input_data:
            content = item[operation["split_key"]]
            tokens = encoder.encode(content)
            chunks = []
            start = 0
            while start < len(tokens):
                end = start + operation["chunk_size"]
                chunk_tokens = tokens[start:end]
                chunk = encoder.decode(chunk_tokens)
                chunk_id = f"chunk_{start}_{end}"
                chunk_data = {"chunk_id": chunk_id, "chunk_content": chunk, **item}
                if self.validate_output(operation, chunk_data):
                    chunks.append(chunk_data)
                start = end - operation["overlap_size"]
            results.extend(chunks)
        return results, 0

    def call_llm(
        self,
        model: str,
        op_type: str,
        prompt: str,
        output_schema: Dict[str, str],
    ) -> str:
        props = {key: convert_val(value) for key, value in output_schema.items()}

        parameters = {"type": "object", "properties": props}
        parameters["required"] = list(props.keys())

        system_prompt = f"You are a helpful assistant to intelligently process data, writing outputs to a database. This is a {op_type} operation."

        response = completion(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
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
        return response

    def parse_llm_response(self, response: Any) -> List[Dict[str, Any]]:
        # This is a simplified parser
        tool_calls = response.choices[0].message.tool_calls
        tools = []
        for tool_call in tool_calls:
            if tool_call.function.name == "write_output":
                args = json.loads(tool_call.function.arguments)
                for arg in args["output"]:
                    tools.append(arg)
        return tools

    def validate_output(self, operation: Dict, output: Dict) -> bool:
        if "validate" not in operation:
            return True
        for validation in operation["validate"]:
            if not eval(validation, {"output": output}):
                self.console.print(
                    f"[bold red]Validation failed:[/bold red] {validation}"
                )
                self.console.print(f"[yellow]Output:[/yellow] {output}")
                return False
        return True

    def load_input_for_join(self, input_config: Dict) -> List[Dict]:
        if input_config["type"] == "file":
            with open(input_config["path"], "r") as file:
                return json.load(file)
        else:
            raise ValueError(f"Unsupported input type for join: {input_config['type']}")
