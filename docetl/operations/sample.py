from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from docetl.operations.base import BaseOperation
from docetl.operations.clustering_utils import get_embeddings_for_clustering


class SampleOperation(BaseOperation):
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
        if "samples" not in self.config and "outliers" not in self.config:
            raise ValueError(
                "Must specify either 'samples' or 'outliers' in SampleOperation configuration"
            )

        if "samples" in self.config:
            if not isinstance(self.config["samples"], (int, float, list)) or (
                isinstance(self.config["samples"], (int, float))
                and self.config["samples"] <= 0
            ):
                raise TypeError("'samples' must be a positive integer, float, or list")

        if "outliers" in self.config:
            outliers_config = self.config["outliers"]
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

        if "outliers" in self.config:
            # Outlier functionality
            outliers_config = self.config["outliers"]
            embeddings, embedding_cost = get_embeddings_for_clustering(
                input_data, outliers_config, self.runner.api
            )
            cost += embedding_cost
            embeddings = np.array(embeddings)

            center = embeddings.mean(axis=0)
            distances = np.sqrt(((embeddings - center) ** 2).sum(axis=1))

            if "std" in outliers_config:
                cutoff = (
                    np.sqrt((embeddings.std(axis=0) ** 2).sum())
                    * outliers_config["std"]
                )
            else:  # "samples" in outliers_config
                distance_distribution = np.sort(distances)
                samples = outliers_config["samples"]
                if isinstance(samples, float):
                    samples = int(samples * (len(distance_distribution) - 1))
                cutoff = distance_distribution[samples]

            keep = outliers_config.get("keep", False)
            include = distances > cutoff if keep else distances <= cutoff

            output_data = [item for idx, item in enumerate(input_data) if include[idx]]
        else:
            samples = self.config["samples"]
            if isinstance(samples, list):
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
                if "stratify" in self.config:
                    stratify = [data[self.config["stratify"]] for data in input_data]

                import sklearn.model_selection

                output_data, dummy = sklearn.model_selection.train_test_split(
                    input_data,
                    train_size=samples,
                    random_state=self.config.get("random_state", None),
                    stratify=stratify,
                )

        return output_data, cost
