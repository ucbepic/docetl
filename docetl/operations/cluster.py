from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
from jinja2 import Template

from .base import BaseOperation
from .clustering_utils import get_embeddings_for_clustering
from .utils import RichLoopBar, strict_render
from docetl.utils import has_jinja_syntax, prompt_user_for_non_jinja_confirmation


class ClusterOperation(BaseOperation):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.max_batch_size: int = self.config.get(
            "max_batch_size", kwargs.get("max_batch_size", float("inf"))
        )
        # Check for non-Jinja prompts and prompt user for confirmation
        if "summary_prompt" in self.config and not has_jinja_syntax(
            self.config["summary_prompt"]
        ):
            if not prompt_user_for_non_jinja_confirmation(
                self.config["summary_prompt"], self.config["name"], "summary_prompt"
            ):
                raise ValueError(
                    f"Operation '{self.config['name']}' cancelled by user. Please add Jinja2 template syntax to your summary_prompt."
                )
            # Mark that we need to append document statement (cluster uses inputs)
            self.config["_append_document_to_prompt"] = True
            self.config["_is_reduce_operation"] = True

    def syntax_check(self) -> None:
        """
        Checks the configuration of the ClusterOperation for required keys and valid structure.

        Raises:
            ValueError: If required keys are missing or invalid in the configuration.
            TypeError: If configuration values have incorrect types.
        """
        required_keys = ["embedding_keys", "summary_schema", "summary_prompt"]
        for key in required_keys:
            if key not in self.config:
                raise ValueError(
                    f"Missing required key '{key}' in ClusterOperation configuration"
                )

        if not isinstance(self.config["embedding_keys"], list):
            raise TypeError("'embedding_keys' must be a list of strings")

        if "output_key" in self.config:
            if not isinstance(self.config["output_key"], str):
                raise TypeError("'output_key' must be a string")

        if not isinstance(self.config["summary_schema"], dict):
            raise TypeError("'summary_schema' must be a dictionary")

        if not isinstance(self.config["summary_prompt"], str):
            raise TypeError("'prompt' must be a string")

        # Check if the prompt has Jinja syntax
        if not has_jinja_syntax(self.config["summary_prompt"]):
            # This will be handled during initialization with user confirmation
            pass
        else:
            # Check if the prompt is a valid Jinja2 template
            try:
                Template(self.config["summary_prompt"])
            except Exception as e:
                raise ValueError(f"Invalid Jinja2 template in 'prompt': {str(e)}")

        # Check optional parameters
        if "max_batch_size" in self.config:
            if not isinstance(self.config["max_batch_size"], int):
                raise TypeError("'max_batch_size' must be an integer")

        if "embedding_model" in self.config:
            if not isinstance(self.config["embedding_model"], str):
                raise TypeError("'embedding_model' must be a string")

        if "model" in self.config:
            if not isinstance(self.config["model"], str):
                raise TypeError("'model' must be a string")

        if "validate" in self.config:
            if not isinstance(self.config["validate"], list):
                raise TypeError("'validate' must be a list of strings")
            for rule in self.config["validate"]:
                if not isinstance(rule, str):
                    raise TypeError("Each validation rule must be a string")

    def execute(
        self, input_data: list[dict], is_build: bool = False
    ) -> tuple[list[dict], float]:
        """
        Executes the cluster operation on the input data. Modifies the
        input data and returns it in place.

        Args:
            input_data (list[dict]): A list of dictionaries to process.
            is_build (bool): Whether the operation is being executed
              in the build phase. Defaults to False.

        Returns:
            tuple[list[dict], float]: A tuple containing the clustered
              list of dictionaries and the total cost of the operation.
        """
        if not input_data:
            return input_data, 0

        if len(input_data) == 1:
            input_data[0][self.config.get("output_key", "clusters")] = ()
            return input_data, 0

        embeddings, cost = get_embeddings_for_clustering(
            input_data, self.config, self.runner.api
        )

        tree = self.agglomerative_cluster_of_embeddings(input_data, embeddings)

        if "collapse" in self.config:
            tree = self.collapse_tree(tree, collapse=self.config["collapse"])

        self.prompt_template = Template(self.config["summary_prompt"])
        cost += self.annotate_clustering_tree(tree)
        self.annotate_leaves(tree)

        return input_data, cost

    def agglomerative_cluster_of_embeddings(self, input_data, embeddings):
        import sklearn.cluster

        cl = sklearn.cluster.AgglomerativeClustering(
            compute_full_tree=True, compute_distances=True
        )
        cl.fit(embeddings)

        nsamples = len(embeddings)

        def build_tree(i):
            if i < nsamples:
                res = input_data[i]
                #                res["embedding"] = list(embeddings[i])
                return res
            return {
                "children": [
                    build_tree(cl.children_[i - nsamples, 0]),
                    build_tree(cl.children_[i - nsamples, 1]),
                ],
                "distance": cl.distances_[i - nsamples],
            }

        return build_tree(nsamples + len(cl.children_) - 1)

    def get_tree_distances(self, t):
        res = set()
        if "distance" in t:
            res.update(
                set(
                    [
                        t["distance"] - child["distance"]
                        for child in t["children"]
                        if "distance" in child
                    ]
                )
            )
        if "children" in t:
            for child in t["children"]:
                res.update(self.get_tree_distances(child))
        return res

    def _collapse_tree(self, t, parent_dist=None, collapse=None):
        if "children" in t:
            if (
                "distance" in t
                and parent_dist is not None
                and collapse is not None
                and parent_dist - t["distance"] < collapse
            ):
                return [
                    grandchild
                    for child in t["children"]
                    for grandchild in self._collapse_tree(
                        child, parent_dist=parent_dist, collapse=collapse
                    )
                ]
            else:
                res = dict(t)
                res["children"] = [
                    grandchild
                    for idx, child in enumerate(t["children"])
                    for grandchild in self._collapse_tree(
                        child, parent_dist=t["distance"], collapse=collapse
                    )
                ]
                return [res]
        else:
            return [t]

    def collapse_tree(self, tree, collapse=None):
        if collapse is not None:
            tree_distances = np.array(sorted(self.get_tree_distances(tree)))
            collapse = tree_distances[int(len(tree_distances) * collapse)]
        return self._collapse_tree(tree, collapse=collapse)[0]

    def annotate_clustering_tree(self, t):
        if "children" in t:
            with ThreadPoolExecutor(max_workers=self.max_batch_size) as executor:
                futures = [
                    executor.submit(self.annotate_clustering_tree, child)
                    for child in t["children"]
                ]

                total_cost = 0
                pbar = RichLoopBar(
                    range(len(futures)),
                    desc=f"Processing {self.config['name']} (map) on all documents",
                    console=self.console,
                )
                for i in pbar:
                    total_cost += futures[i].result()
                    pbar.update(i)

            prompt = strict_render(self.prompt_template, {"inputs": t["children"]})

            def validation_fn(response: dict[str, Any]):
                output = self.runner.api.parse_llm_response(
                    response,
                    schema=self.config["summary_schema"],
                    manually_fix_errors=self.manually_fix_errors,
                )[0]
                if self.runner.api.validate_output(self.config, output, self.console):
                    return output, True
                return output, False

            response = self.runner.api.call_llm(
                model=self.config.get("model", self.default_model),
                op_type="cluster",
                messages=[{"role": "user", "content": prompt}],
                output_schema=self.config["summary_schema"],
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
                litellm_completion_kwargs=self.config.get(
                    "litellm_completion_kwargs", {}
                ),
                op_config=self.config,
            )
            total_cost += response.total_cost
            if response.validated:
                output = self.runner.api.parse_llm_response(
                    response.response,
                    schema=self.config["summary_schema"],
                    manually_fix_errors=self.manually_fix_errors,
                )[0]
                t.update(output)

            return total_cost
        return 0

    def annotate_leaves(self, tree, path=()):
        if "children" in tree:
            item = dict(tree)
            item.pop("children")
            for child in tree["children"]:
                self.annotate_leaves(child, path=(item,) + path)
        else:
            tree[self.config.get("output_key", "clusters")] = path
