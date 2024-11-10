from typing import List, Dict, Tuple, Any, Optional
from concurrent.futures import ThreadPoolExecutor
from rich.progress import Progress
from .base import BaseOperation
from docetl_resolver import FastResolver
from rich.console import Console
from rich.status import Status
from jinja2 import Template
import jinja2
from docetl.operations.utils import RichLoopBar, rich_as_completed

class FastResolveOperation(BaseOperation):
    class schema(BaseOperation.schema):
        type: str = "fast_resolve"
        comparison_prompt: str
        resolution_prompt: str
        output: Optional[Dict[str, Any]] = None
        embedding_model: Optional[str] = None
        resolution_model: Optional[str] = None
        comparison_model: Optional[str] = None
        blocking_threshold: Optional[float] = None
        blocking_keys: Optional[List[str]] = None
        embedding_batch_size: Optional[int] = None
        compare_batch_size: Optional[int] = None

    def syntax_check(self):
        """Check if the config is valid."""
        required_keys = ["comparison_prompt", "output"]
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required key '{key}' in FastResolveOperation configuration")

        if "schema" not in self.config["output"]:
            raise ValueError("Missing 'schema' in 'output' configuration")

        if not isinstance(self.config["output"]["schema"], dict):
            raise TypeError("'schema' in 'output' configuration must be a dictionary")

        if not self.config["output"]["schema"]:
            raise ValueError("'schema' in 'output' configuration cannot be empty")

        # Check if the comparison_prompt is a valid Jinja2 template
        try:
            comparison_template = Template(self.config["comparison_prompt"])
            comparison_vars = comparison_template.environment.parse(
                self.config["comparison_prompt"]
            ).find_all(jinja2.nodes.Name)
            comparison_var_names = {var.name for var in comparison_vars}
            if "input1" not in comparison_var_names or "input2" not in comparison_var_names:
                raise ValueError(
                    "'comparison_prompt' must contain both 'input1' and 'input2' variables"
                )

            if "resolution_prompt" in self.config:
                reduction_template = Template(self.config["resolution_prompt"])
                reduction_vars = reduction_template.environment.parse(
                    self.config["resolution_prompt"]
                ).find_all(jinja2.nodes.Name)
                reduction_var_names = {var.name for var in reduction_vars}
                if "inputs" not in reduction_var_names:
                    raise ValueError("'resolution_prompt' must contain 'inputs' variable")
        except Exception as e:
            raise ValueError(f"Invalid Jinja2 template: {str(e)}")

    def __init__(
        self,
        runner: "ConfigWrapper",
        config: Dict,
        default_model: str,
        max_threads: int,
        console: Optional[Console] = None,
        status: Optional[Status] = None,
        is_build: bool = False,
        **kwargs,
    ):
        super().__init__(runner, config, default_model, max_threads, console, status, is_build, **kwargs)
        self.resolver = FastResolver(
            blocking_threshold=config.get("blocking_threshold", 0.8)
        )

    def batch_embeddings(self, items: List[Dict], batch_size: int = 1000) -> Tuple[List[List[float]], float]:
        """Get embeddings for all items in parallel batches."""
        all_embeddings = []
        total_cost = 0
        blocking_keys = self.config.get("blocking_keys", list(items[0].keys()))
        
        def process_batch(batch):
            texts = [
                " ".join(str(item[key]) for key in blocking_keys if key in item)
                for item in batch
            ]
            response = self.runner.api.gen_embedding(
                model=self.config.get("embedding_model", "text-embedding-3-small"),
                input=texts
            )
            return [data["embedding"] for data in response["data"]], response.get("usage", {}).get("total_tokens", 0) * 0.0001
            
        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            futures = []
            for i in range(0, len(items), batch_size):
                batch = items[i:i + batch_size]
                futures.append(executor.submit(process_batch, batch))
                
            for future in rich_as_completed(
                futures,
                total=len(futures),
                desc="Generating embeddings",
                console=self.console
            ):
                embeddings, cost = future.result()
                all_embeddings.extend(embeddings)
                total_cost += cost
                
        return all_embeddings, total_cost

    def compare_pair(self, item1: Dict, item2: Dict) -> Tuple[bool, float]:
        """Compare two items using the LLM."""
        prompt_template = Template(self.config["comparison_prompt"])
        prompt = prompt_template.render(input1=item1, input2=item2)
        
        response = self.runner.api.call_llm(
            self.config.get("comparison_model", self.default_model),
            "compare",
            [{"role": "user", "content": prompt}],
            {"is_match": "bool"},
            timeout_seconds=self.config.get("timeout", 120),
            max_retries_per_timeout=self.config.get("max_retries_per_timeout", 2),
            bypass_cache=self.config.get("bypass_cache", False),
        )
        output = self.runner.api.parse_llm_response(
            response.response,
            {"is_match": "bool"},
        )[0]
        return output["is_match"], response.total_cost

    def process_cluster(self, cluster: List[int], items: List[Dict]) -> Tuple[List[Dict], float]:
        """Process a cluster of items to generate a resolved output."""
        if len(cluster) == 1:
            return [items[cluster[0]]], 0
            
        cluster_items = [items[i] for i in cluster]
        reduction_template = Template(self.config["resolution_prompt"])
        resolution_prompt = reduction_template.render(inputs=cluster_items)
        
        response = self.runner.api.call_llm(
            self.config.get("resolution_model", self.default_model),
            "resolve",
            [{"role": "user", "content": resolution_prompt}],
            self.config["output"]["schema"],
            timeout_seconds=self.config.get("timeout", 120),
            max_retries_per_timeout=self.config.get("max_retries_per_timeout", 2),
            bypass_cache=self.config.get("bypass_cache", False),
            validation_config=(
                {
                    "val_rule": self.config.get("validate", []),
                    "validation_fn": self.validation_fn,
                }
                if self.config.get("validate", None)
                else None
            ),
        )
        
        if response.validated:
            resolved = self.runner.api.parse_llm_response(
                response.response,
                self.config["output"]["schema"],
                manually_fix_errors=self.manually_fix_errors,
            )[0]
            
            results = []
            for idx in cluster:
                item = items[idx].copy()
                # Save original values before overwriting
                keys_in_output = [k for k in resolved.keys() if k in item.keys()]
                item[f"_kv_pairs_preresolve_{self.config['name']}"] = {
                    k: item[k] for k in keys_in_output
                }
                item.update(resolved)
                results.append(item)
                
            return results, response.total_cost
            
        return [], response.total_cost

    def validation_fn(self, response: Dict[str, Any]):
        output = self.runner.api.parse_llm_response(
            response,
            schema=self.config["output"]["schema"],
        )[0]
        if self.runner.api.validate_output(self.config, output, self.console):
            return output, True
        return output, False

    def auto_batch(self, num_pairs: int) -> int:
        """Calculate optimal batch size based on number of comparisons."""
        # Maximum batch size limit for 4o-mini model
        M = 500
        
        n = len(self.input_data)
        m = num_pairs
        
        # https://www.wolframalpha.com/input?i=k%28k-1%29%2F2+%2B+%28n-k%29%28k-1%29+%3D+m%2C+solve+for+k
        # Two possible solutions for k:
        # k = -1/2 sqrt((1 - 2n)^2 - 8m) + n + 1/2
        # k = 1/2 (sqrt((1 - 2n)^2 - 8m) + 2n + 1)
        
        discriminant = (1 - 2*n)**2 - 8*m
        sqrt_discriminant = discriminant ** 0.5
        
        k1 = -0.5 * sqrt_discriminant + n + 0.5
        k2 = 0.5 * (sqrt_discriminant + 2*n + 1)
        
        # Take the maximum viable solution
        k = max(k1, k2)
        return M if k < 0 else min(int(k), M)

    def execute(self, input_data: List[Dict]) -> Tuple[List[Dict], float]:
        """Execute the fast resolve operation."""
        if not input_data:
            return [], 0
            
        self.input_data = input_data
        total_cost = 0
        
        # Set up blocking rules
        blocking_conditions = self.config.get("blocking_conditions", [])
        for condition in blocking_conditions:
            # Parse the condition string to extract keys and operation
            if "in" in condition:
                parts = condition.split("in")
                if parts[0].strip().endswith(".lower()") and parts[1].strip().endswith(".lower()"):
                    key1 = parts[0].split("[")[1].split("]")[0].strip('"\'')
                    key2 = parts[1].split("[")[1].split("]")[0].strip('"\'')
                    
                    if parts[0].strip().startswith("input1"):
                        self.resolver.add_contains_rule(key1, key2)
                    else:
                        self.resolver.add_contained_in_rule(key1, key2)
            elif "==" in condition:
                parts = condition.split("==")
                if parts[0].strip().endswith(".lower()") and parts[1].strip().endswith(".lower()"):
                    key1 = parts[0].split("[")[1].split("]")[0].strip('"\'')
                    key2 = parts[1].split("[")[1].split("]")[0].strip('"\'')
                    self.resolver.add_equals_rule(key1, key2)

        # Get embeddings with configurable batch size
        embedding_batch_size = self.config.get("embedding_batch_size", 1000)
        embeddings, embedding_cost = self.batch_embeddings(input_data, batch_size=embedding_batch_size)
        total_cost += embedding_cost
        
        # Get comparison pairs from Rust, including blocking rules
        comparison_pairs = self.resolver.process_embeddings(embeddings, input_data)
        
        # Calculate and log statistics
        total_possible_comparisons = len(input_data) * (len(input_data) - 1) // 2
        comparisons_made = len(comparison_pairs)
        comparisons_saved = total_possible_comparisons - comparisons_made
        
        self.console.log(
            f"[green]Comparisons saved by blocking: {comparisons_saved} "
            f"({(comparisons_saved / total_possible_comparisons) * 100:.2f}%)[/green]"
        )
        self.console.log(
            f"[blue]Number of pairs to compare: {comparisons_made}[/blue]"
        )
        
        # Calculate batch size for comparisons
        batch_size = self.config.get("compare_batch_size", self.auto_batch(len(comparison_pairs)))
        self.console.log(f"Using compare batch size: {batch_size}")
        
        # Process comparisons in batches with progress bar
        pbar = RichLoopBar(
            range(0, len(comparison_pairs), batch_size),
            desc=f"Processing batches of {batch_size} LLM comparisons",
            console=self.console,
        )
        
        for i in pbar:
            batch = comparison_pairs[i:i + batch_size]
            
            with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
                futures = []
                valid_pairs = []
                
                # Pre-filter pairs that might already be in same cluster or processed
                for i, j in batch:
                    if (self.resolver.find_cluster(i) != self.resolver.find_cluster(j) and 
                        not self.resolver.is_processed(i, j)):
                        futures.append(
                            executor.submit(
                                self.compare_pair,
                                input_data[i],
                                input_data[j]
                            )
                        )
                        valid_pairs.append((i, j))
                
                # Process results and merge clusters
                for future, (i, j) in zip(futures, valid_pairs):
                    is_match, cost = future.result()
                    total_cost += cost
                    # Mark pair as processed regardless of match result
                    self.resolver.mark_processed(i, j)
                    if is_match:
                        self.resolver.merge_clusters(i, j)
            
            pbar.update(i//batch_size)
        
        # Get final clusters
        clusters = self.resolver.get_clusters()
        
        # Calculate and log cluster statistics
        num_records_before = len(input_data)
        num_clusters_after = len(clusters)
        self.console.log(f"Number of records before resolution: {num_records_before}")
        self.console.log(f"Number of distinct records after resolution: {num_clusters_after}")
        
        # Calculate and log self-join selectivity
        true_match_count = sum(
            len(cluster) * (len(cluster) - 1) // 2
            for cluster in clusters
            if len(cluster) > 1
        )
        true_match_selectivity = true_match_count / total_possible_comparisons if total_possible_comparisons > 0 else 0
        self.console.log(f"Self-join selectivity: {true_match_selectivity:.4f}")
        
        # Process each cluster in parallel with progress
        results = []
        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            futures = [
                executor.submit(self.process_cluster, cluster, input_data)
                for cluster in clusters
            ]
            
            for future in rich_as_completed(
                futures,
                total=len(futures),
                desc="Resolving clusters",
                console=self.console
            ):
                cluster_results, cost = future.result()
                results.extend(cluster_results)
                total_cost += cost
        
        return results, total_cost