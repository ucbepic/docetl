import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple

import jinja2
from jinja2 import Template
from rich.prompt import Confirm

from docetl.operations.base import BaseOperation
from docetl.operations.utils import RichLoopBar, rich_as_completed
from docetl.utils import completion_cost, extract_jinja_variables
from .clustering_utils import get_embeddings_for_clustering
from sklearn.metrics.pairwise import cosine_similarity

class LinkResolveOperation(BaseOperation):
    def syntax_check(self) -> None:
        pass
    
    def execute(self, input_data: List[Dict]) -> Tuple[List[Dict], float]:
        """
        Executes the resolve links operation on the provided dataset.

        Args:
            input_data (List[Dict]): The dataset to resolve.

        Returns:
            Tuple[List[Dict], float]: A tuple containing the resolved results and the total cost of the operation.

        """
        if len(input_data) == 0:
            return [], 0

        self.prompt_template = Template(self.config["comparison_prompt"])
        
        id_key = self.config.get("id_key", "title")
        link_key = self.config.get("link_key", "related_to")
        blocking_threshold = self.config.get("blocking_threshold")
        blocking_conditions = self.config.get("blocking_conditions", [])

        embedding_model = self.config.get("embedding_model", "text-embedding-3-small")

        item_by_id = {item[id_key]: item
                      for item in input_data}
        
        id_values = set([item[id_key] for item in input_data])

        link_values = set()
        for item in input_data:
            link_values.update(item[link_key])

        to_resolve = list(link_values - id_values)
        id_values = list(id_values)
        
        if not blocking_threshold and not blocking_conditions:
            # Prompt the user for confirmation
            if not Confirm.ask(
                f"[yellow]Warning: No blocking keys or conditions specified. "
                f"This may result in a large number of comparisons. "
                f"We recommend specifying at least one blocking key or condition, or using the optimizer to automatically come up with these. "
                f"Do you want to continue without blocking?[/yellow]",
            ):
                raise ValueError("Operation cancelled by user.")


        id_embeddings, id_embedding_cost = get_embeddings_for_clustering(
            [{"key": value} for value in id_values],
            {
                "embedding_model": embedding_model,
                "embedding_keys": ["key"]
            },
            self.runner.api
        )
        link_embeddings, link_embedding_cost = get_embeddings_for_clustering(
            [{"key": value} for value in to_resolve],
            {
                "embedding_model": embedding_model,
                "embedding_keys": ["key"]
            },
            self.runner.api
        )

        similarity_matrix = cosine_similarity(link_embeddings, id_embeddings)

        closest = np.argmin(similarity_matrix, axis=1)
        acceptable = similarity_matrix.min(axis=1) < blocking_threshold

        acceptable_idxs = np.nonzero(acceptable)
        close_enough = np.zeros(len(acceptable_idxs))


        total_possible_comparisons = len(acceptable)
        comparisons_saved = total_possible_comparisons - acceptable.sum()
        
        self.console.log(
            f"[green]Comparisons saved by blocking: {comparisons_saved} "
            f"({(comparisons_saved / total_possible_comparisons) * 100:.2f}%)[/green]"
        )
        
        self.replacements = {}
        
        with ThreadPoolExecutor(max_workers=self.max_batch_size) as executor:

            futures = []
            for link_idx in np.nonzero(acceptable):
                id_idx = closest[link_idx]
                link_value = to_resolve[id_idx]
                id_value = id_values[id_idx]
                item = item_by_id[id_value]

                futures.append(
                    executor.submit(
                        self.compare,
                        link_idx = link_idx,
                        id_idx = id_idx,
                        link_value = link_value,
                        id_value = id_value,
                        item = item))

            total_cost = 0
            pbar = RichLoopBar(
                range(len(futures)),
                desc=f"Processing {self.config['name']} (map) on all documents",
                console=self.console,
            )
            for i in pbar:
                total_cost += futures[i].result()
                pbar.update(i)

        for item in input_data:
            item[link_key] = [self.replacements.get(value, value)
                              for value in item[link_key]]

        return input_data, total_cost
                    
    def compare(self, link_idx, id_idx, link_value, id_value, item):
        prompt = self.prompt_template.render(
            link_value = link_value,
            id_value = id_value,
            item = item
        )

        schema = {"is_same": "bool"}

        def validation_fn(response: Dict[str, Any]):
            output = self.runner.api.parse_llm_response(
                response,
                schema=schema,
                manually_fix_errors=self.manually_fix_errors,
            )[0]
            if self.runner.api.validate_output(self.config, output, self.console):
                return output, True
            return output, False

        response = self.runner.api.call_llm(
            model=self.config.get("model", self.default_model),
            op_type="cluster",
            messages=[{"role": "user", "content": prompt}],
            output_schema=schema,
            timeout_seconds=self.config.get("timeout", 120),
            bypass_cache=self.config.get("bypass_cache", False),
            max_retries_per_timeout=self.config.get("max_retries_per_timeout", 2),
            validation_config=(
                {
                    "num_retries": self.num_retries_on_validate_failure,
                    "val_rule": self.config.get("validate", []),
                    "validation_fn": validation_fn,
                }
                if self.config.get("validate", None)
                else None
            ),
            verbose=self.config.get("verbose", False),
        )
        total_cost += response.total_cost
        if response.validated:
            output = self.runner.api.parse_llm_response(
                response.response,
                schema=schema,
                manually_fix_errors=self.manually_fix_errors,
            )[0]
            if output["is_same"]:
                self.replacements[link_value] = id_value

        return total_cost

        
        

    