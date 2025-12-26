from concurrent.futures import ThreadPoolExecutor
from typing import Any

from jinja2 import Template
from rich.prompt import Confirm
from sklearn.metrics.pairwise import cosine_similarity

from docetl.operations.base import BaseOperation
from docetl.operations.utils import RichLoopBar, strict_render
from docetl.utils import has_jinja_syntax, prompt_user_for_non_jinja_confirmation

from .clustering_utils import get_embeddings_for_clustering


class LinkResolveOperation(BaseOperation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Check for non-Jinja prompts and prompt user for confirmation
        if "comparison_prompt" in self.config and not has_jinja_syntax(
            self.config["comparison_prompt"]
        ):
            if not prompt_user_for_non_jinja_confirmation(
                self.config["comparison_prompt"],
                self.config["name"],
                "comparison_prompt",
            ):
                raise ValueError(
                    f"Operation '{self.config['name']}' cancelled by user. Please add Jinja2 template syntax to your comparison_prompt."
                )
            # Mark that we need to append document statement
            # Note: link_resolve uses link_value, id_value, and item, so strict_render will handle it
            self.config["_append_document_to_comparison_prompt"] = True
    def execute(self, input_data: list[dict]) -> tuple[list[dict], float]:
        """
        Executes the resolve links operation on the provided dataset.

        Args:
            input_data (list[dict]): The dataset to resolve.

        Returns:
            tuple[list[dict], float]: A tuple containing the resolved results and the total cost of the operation.

        """
        if len(input_data) == 0:
            return [], 0

        self.prompt_template = Template(self.config["comparison_prompt"])

        id_key = self.config.get("id_key", "title")
        link_key = self.config.get("link_key", "related_to")
        blocking_threshold = self.config.get("blocking_threshold")
        blocking_conditions = self.config.get("blocking_conditions", [])

        # Note: We don't want to use text-embedding-3-small as it has bad performance on short texts...
        embedding_model = self.config.get("embedding_model", "text-embedding-ada-002")

        item_by_id = {item[id_key]: item for item in input_data}

        id_values = set([item[id_key] for item in input_data])

        link_values = set()
        for item in input_data:
            link_values.update(item[link_key])

        to_resolve = list(link_values - id_values)
        id_values = list(id_values)

        if not blocking_threshold and not blocking_conditions:
            # Prompt the user for confirmation
            if not Confirm.ask(
                "[yellow]Warning: No blocking keys or conditions specified. "
                "This may result in a large number of comparisons. "
                "We recommend specifying at least one blocking key or condition, or using the optimizer to automatically come up with these. "
                "Do you want to continue without blocking?[/yellow]",
            ):
                raise ValueError("Operation cancelled by user.")

        id_embeddings, id_embedding_cost = get_embeddings_for_clustering(
            [{"key": value} for value in id_values],
            {"embedding_model": embedding_model, "embedding_keys": ["key"]},
            self.runner.api,
        )
        link_embeddings, link_embedding_cost = get_embeddings_for_clustering(
            [{"key": value} for value in to_resolve],
            {"embedding_model": embedding_model, "embedding_keys": ["key"]},
            self.runner.api,
        )

        similarity_matrix = cosine_similarity(link_embeddings, id_embeddings)

        acceptable = similarity_matrix >= blocking_threshold

        total_possible_comparisons = acceptable.shape[0] * acceptable.shape[1]
        comparisons_saved = total_possible_comparisons - acceptable.sum().sum()

        self.console.log(
            f"[green]Comparisons saved by blocking: {comparisons_saved} "
            f"({(comparisons_saved / total_possible_comparisons) * 100:.2f}%)[/green]"
        )

        self.replacements = {}

        batch_size = self.config.get("compare_batch_size", 100)
        with ThreadPoolExecutor(max_workers=batch_size) as executor:

            futures = []
            for link_idx in range(acceptable.shape[0]):
                for id_idx in range(acceptable.shape[1]):
                    if not acceptable[link_idx, id_idx]:
                        continue

                    id_value = id_values[id_idx]
                    link_value = to_resolve[link_idx]
                    item = item_by_id[id_value]

                    futures.append(
                        executor.submit(
                            self.compare,
                            link_idx=link_idx,
                            id_idx=id_idx,
                            link_value=link_value,
                            id_value=id_value,
                            item=item,
                        )
                    )

            total_cost = 0
            pbar = RichLoopBar(
                range(len(futures)),
                desc=f"Processing {self.config['name']} (map) on all documents",
                console=self.console,
            )
            for i in pbar:
                total_cost += futures[i].result()
                pbar.update(i)

        self.console.log(
            f"[green]Number of replacements found: {len(self.replacements)} "
            f"({(len(self.replacements) / total_possible_comparisons) * 100:.2f}% of all comparisons)[/green]"
        )

        for item in input_data:
            item[link_key] = [
                self.replacements.get(value, value) for value in item[link_key]
            ]

        return input_data, total_cost

    def compare(self, link_idx, id_idx, link_value, id_value, item):
        prompt = strict_render(
            self.prompt_template,
            {"link_value": link_value, "id_value": id_value, "item": item},
        )

        schema = {"is_same": "bool"}

        def validation_fn(response: dict[str, Any]):
            output = self.runner.api.parse_llm_response(
                response,
                schema=schema,
                manually_fix_errors=self.manually_fix_errors,
            )[0]
            if self.runner.api.validate_output(self.config, output, self.console):
                return output, True
            return output, False

        response = self.runner.api.call_llm(
            model=self.config.get("comparison_model", self.default_model),
            op_type="link_resolve",
            messages=[{"role": "user", "content": prompt}],
            output_schema=schema,
            timeout_seconds=self.config.get("timeout", 120),
            bypass_cache=self.config.get("bypass_cache", self.bypass_cache),
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
            litellm_completion_kwargs=self.config.get("litellm_completion_kwargs", {}),
            op_config=self.config,
        )

        if response.validated:
            output = self.runner.api.parse_llm_response(
                response.response,
                schema=schema,
                manually_fix_errors=self.manually_fix_errors,
            )[0]
            if output["is_same"]:
                self.replacements[link_value] = id_value

        return response.total_cost
