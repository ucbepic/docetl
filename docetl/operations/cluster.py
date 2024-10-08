from jinja2 import Environment, Template
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple
from .base import BaseOperation
from .utils import RichLoopBar
from .clustering_utils import get_embeddings_for_clustering

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

    def syntax_check(self) -> None:
        """
        Checks the configuration of the ClusterOperation for required keys and valid structure.

        Raises:
            ValueError: If required keys are missing
        """

        pass

    
    def execute(
        self, input_data: List[Dict], is_build: bool = False
    ) -> Tuple[List[Dict], float]:
        """
        Executes the cluster operation on the input data. Modifies the
        input data and returns it in place.

        Args:
            input_data (List[Dict]): A list of dictionaries to process.
            is_build (bool): Whether the operation is being executed
              in the build phase. Defaults to False.

        Returns:
            Tuple[List[Dict], float]: A tuple containing the clustered
              list of dictionaries and the total cost of the operation.
        """
        
        embeddings, cost = get_embeddings_for_clustering(
            input_data, self.config, self.runner.api
        )
        
        tree = self.agglomerative_cluster_of_embeddings(
            input_data, embeddings)

        self.prompt_template = Template(self.config["summary_prompt"])        
        cost += self.annotate_clustering_tree(tree)
        self.annotate_leaves(tree)
        
        return input_data, cost
        
    def agglomerative_cluster_of_embeddings(self, input_data, embeddings):
        import sklearn.cluster
        
        cl = sklearn.cluster.AgglomerativeClustering(compute_full_tree=True, compute_distances=True)
        cl.fit(embeddings)

        nsamples = len(embeddings)
        def build_tree(i):
            if i < nsamples:
                res = input_data[i]
#                res["embedding"] = list(embeddings[i])
                return res
            return {"children": [build_tree(cl.children_[i-nsamples, 0]), build_tree(cl.children_[i-nsamples, 1])],
                    "distance": cl.distances_[i-nsamples]}

        return build_tree(nsamples + len(cl.children_)-1)

    def annotate_clustering_tree(self, t):
        if "children" in t:
            with ThreadPoolExecutor(max_workers=self.max_batch_size) as executor:
                futures = [executor.submit(self.annotate_clustering_tree, child) for child in t["children"]]

                total_cost = 0
                pbar = RichLoopBar(
                    range(len(futures)),
                    desc=f"Processing {self.config['name']} (map) on all documents",
                    console=self.console,
                )
                for i in pbar:
                    total_cost += futures[i].result()
                    pbar.update(i)

            assert len(t["children"]) == 2, "Agglomerative clustering is supposed to generate clusters with 2 children each, but this cluster has %s" % len(t["children"])
            prompt = self.prompt_template.render(
                left=t["children"][0], right=t["children"][1])
            
            def validation_fn(response: Dict[str, Any]):
                output = self.runner.api.parse_llm_response(
                    response,
                    schema=self.config["summary_schema"],
                    tools=self.config.get("tools", None),
                    manually_fix_errors=self.manually_fix_errors,
                )[0]
                if self.runner.api.validate_output(self.config, output, self.console):
                    return output, True
                return output, False

            output, cost, success = self.runner.api.call_llm_with_validation(
                [{"role": "user", "content": prompt}],
                model=self.config.get("model", self.default_model),
                operation_type="cluster",
                schema=self.config["summary_schema"],
                llm_call_fn=lambda messages: self.runner.api.call_llm(
                    self.config.get("model", self.default_model),
                    "cluster",
                    messages,
                    self.config["summary_schema"],
                    tools=self.config.get("tools", None),
                    console=self.console,
                    timeout_seconds=self.config.get("timeout", 120),
                    max_retries_per_timeout=self.config.get(
                        "max_retries_per_timeout", 2
                    ),
                ),
                validation_fn=validation_fn,
                val_rule=self.config.get("validate", []),
                num_retries=self.num_retries_on_validate_failure,
                console=self.console,
            )

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
        
