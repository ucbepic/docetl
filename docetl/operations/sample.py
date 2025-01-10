from typing import Dict, List, Tuple

import numpy as np

from docetl.operations.base import BaseOperation
from docetl.operations.clustering_utils import get_embeddings_for_clustering


class SampleOperation(BaseOperation):
    """
    Params:
    - method: "uniform", "stratify", "outliers", "custom"
    - samples: int, float, or list
    - method_kwargs: dict, optional
        - embedding_model: str, optional
        - embedding_keys: list, optional
        - center: dict, optional
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    def syntax_check(self) -> None:
        """
        Checks the configuration of the SampleOperation for required keys and valid structure.

        Raises:
            ValueError: If required keys are missing or invalid in the configuration.
            TypeError: If configuration values have incorrect types.
        """
        if "method" not in self.config:
            raise ValueError("Must specify 'method' in SampleOperation configuration")

        valid_methods = ["uniform", "stratify", "outliers", "custom"]
        if self.config["method"] not in valid_methods:
            raise ValueError(f"'method' must be one of {valid_methods}")

        if self.config["method"] == "custom":
            # Samples must be a list
            if not isinstance(self.config["samples"], list):
                raise TypeError("'samples' must be a list for custom sampling")

        if self.config["method"] in ["random", "stratify"]:
            if "samples" not in self.config:
                raise ValueError(
                    "Must specify 'samples' for random or stratify sampling"
                )
            if not isinstance(self.config["samples"], (int, float, list)) or (
                isinstance(self.config["samples"], (int, float))
                and self.config["samples"] <= 0
            ):
                raise TypeError("'samples' must be a positive integer, float, or list")

        if self.config["method"] == "stratify":
            if "stratify_key" not in self.config.get("method_kwargs", {}):
                raise ValueError("Must specify 'stratify_key' for stratify sampling")
            if not isinstance(
                self.config.get("method_kwargs", {})["stratify_key"], str
            ):
                raise TypeError("'stratify_key' must be a string")

        if self.config["method"] == "outliers":
            outliers_config = self.config.get("method_kwargs", {})
            if "std" not in outliers_config and "samples" not in outliers_config:
                raise ValueError(
                    "Must specify either 'std' or 'samples' in outliers configuration"
                )

            if "std" in outliers_config:
                if (
                    not isinstance(outliers_config["std"], (int, float))
                    or outliers_config["std"] <= 0
                ):
                    raise TypeError("'std' in outliers must be a positive number")

            if "samples" in outliers_config:
                if (
                    not isinstance(outliers_config["samples"], (int, float))
                    or outliers_config["samples"] <= 0
                ):
                    raise TypeError(
                        "'samples' in outliers must be a positive integer or float"
                    )

            if "embedding_keys" not in outliers_config:
                raise ValueError(
                    "'embedding_keys' must be specified in outliers configuration"
                )

            if not isinstance(outliers_config["embedding_keys"], list) or not all(
                isinstance(key, str) for key in outliers_config["embedding_keys"]
            ):
                raise TypeError(
                    "'embedding_keys' in outliers must be a list of strings"
                )

        if "center" in self.config.get("method_kwargs", {}):
            if not isinstance(self.config.get("method_kwargs", {})["center"], dict):
                raise TypeError("'center' must be a dictionary")

    def execute(
        self, input_data: List[Dict], is_build: bool = False
    ) -> Tuple[List[Dict], float]:
        """
        Executes the sample operation on the input data.

        Args:
            input_data (List[Dict]): A list of dictionaries to process.
            is_build (bool): Whether the operation is being executed
              in the build phase. Defaults to False.

        Returns:
            Tuple[List[Dict], float]: A tuple containing the filtered
              list of dictionaries and the total cost of the operation.
        """
        cost = 0
        if not input_data:
            return [], cost

        if self.config["method"] == "outliers":
            # Outlier functionality
            outliers_config = self.config.get("method_kwargs", {})
            embeddings, embedding_cost = get_embeddings_for_clustering(
                input_data, outliers_config, self.runner.api
            )
            cost += embedding_cost
            embeddings = np.array(embeddings)

            if "center" in outliers_config:
                center_embeddings, cost2 = get_embeddings_for_clustering(
                    [outliers_config["center"]], outliers_config, self.runner.api
                )
                cost += cost2
                center = np.array(center_embeddings[0])

            else:
                center = embeddings.mean(axis=0)

            distances = np.sqrt(((embeddings - center) ** 2).sum(axis=1))

            if "std" in outliers_config:
                cutoff = (
                    np.sqrt((embeddings.std(axis=0) ** 2).sum())
                    * outliers_config["std"]
                )
            else:  # "samples" in config
                distance_distribution = np.sort(distances)
                samples = self.config["samples"]
                if isinstance(samples, float):
                    samples = int(samples * (len(distance_distribution) - 1))
                cutoff = distance_distribution[samples]

            keep = outliers_config.get("keep", False)
            include = distances > cutoff if keep else distances <= cutoff

            output_data = [item for idx, item in enumerate(input_data) if include[idx]]
        else:
            samples = self.config["samples"]
            if self.config["method"] == "custom":
                keys = list(samples[0].keys())
                key_to_doc = {
                    tuple([doc[key] for key in keys]): doc for doc in input_data
                }

                output_data = [
                    key_to_doc[tuple([sample[key] for key in keys])]
                    for sample in samples
                ]
            else:
                stratify = None
                if self.config["method"] == "stratify":
                    stratify = [
                        data[self.config.get("method_kwargs", {})["stratify_key"]]
                        for data in input_data
                    ]

                import sklearn.model_selection

                output_data, _ = sklearn.model_selection.train_test_split(
                    input_data,
                    train_size=samples,
                    random_state=self.config.get("random_state", None),
                    stratify=stratify,
                )

        return output_data, cost
