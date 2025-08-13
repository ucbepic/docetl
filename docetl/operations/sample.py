from typing import Any, Literal, Union

import numpy as np
from pydantic import Field, field_validator, model_validator

from docetl.operations.base import BaseOperation
from docetl.operations.clustering_utils import get_embeddings_for_clustering


class SampleOperation(BaseOperation):
    """
    Params:
    - method: "uniform", "stratify", "outliers", "custom", "first"
    - samples: int, float, or list
    - method_kwargs: dict, optional
        - embedding_model: str, optional
        - embedding_keys: list, optional
        - center: dict, optional
        - stratify_key: str or list of str (for stratify method)
            Single key or list of keys to stratify by. Multiple keys create groups by combination of values.
        - samples_per_group: bool, optional (for stratify method)
            If True, sample 'samples' items from each stratify group instead of dividing total samples across groups
    """

    class schema(BaseOperation.schema):
        type: str = "sample"
        method: Literal["uniform", "stratify", "outliers", "custom", "first"]
        samples: Union[int, float, list] | None = None
        method_kwargs: dict[str, Any] | None = Field(default_factory=dict)
        random_state: int | None = Field(None, ge=0)

        @field_validator("samples")
        def validate_samples(cls, v, info):
            if v is not None:
                # For custom method, samples must be a list
                if hasattr(info.data, "method") and info.data.get("method") == "custom":
                    if not isinstance(v, list):
                        raise TypeError("'samples' must be a list for custom sampling")
                elif isinstance(v, (int, float)):
                    if v <= 0:
                        raise ValueError("'samples' must be a positive number")
            return v

        @field_validator("method_kwargs")
        def validate_method_kwargs(cls, v):
            if v is not None:
                if not isinstance(v, dict):
                    raise TypeError("'method_kwargs' must be a dictionary")

                # Validate stratify_key - can be string or list of strings
                if "stratify_key" in v:
                    stratify_key = v["stratify_key"]
                    if isinstance(stratify_key, str):
                        # Single key is fine
                        pass
                    elif isinstance(stratify_key, list):
                        if not all(isinstance(key, str) for key in stratify_key):
                            raise TypeError("All items in 'stratify_key' list must be strings")
                        if len(stratify_key) == 0:
                            raise ValueError("'stratify_key' list cannot be empty")
                    else:
                        raise TypeError("'stratify_key' must be a string or list of strings")

                # Validate samples_per_group if present
                if "samples_per_group" in v:
                    if not isinstance(v["samples_per_group"], bool):
                        raise TypeError("'samples_per_group' must be a boolean")

                if "center" in v and not isinstance(v["center"], dict):
                    raise TypeError("'center' must be a dictionary")

                if "embedding_keys" in v:
                    if not isinstance(v["embedding_keys"], list) or not all(
                        isinstance(key, str) for key in v["embedding_keys"]
                    ):
                        raise TypeError("'embedding_keys' must be a list of strings")

                if "std" in v:
                    if not isinstance(v["std"], (int, float)) or v["std"] <= 0:
                        raise TypeError("'std' must be a positive number")

                if "samples" in v:
                    if not isinstance(v["samples"], (int, float)) or v["samples"] <= 0:
                        raise TypeError(
                            "'samples' in method_kwargs must be a positive number"
                        )

            return v

        @model_validator(mode="after")
        def validate_method_specific_requirements(self):
            method = self.method

            if method in ["uniform", "stratify"] and self.samples is None:
                raise ValueError(f"Must specify 'samples' for {method} sampling")

            if method == "stratify":
                method_kwargs = self.method_kwargs or {}
                if "stratify_key" not in method_kwargs:
                    raise ValueError(
                        "Must specify 'stratify_key' for stratify sampling"
                    )

            if method == "outliers":
                method_kwargs = self.method_kwargs or {}
                if "std" not in method_kwargs and "samples" not in method_kwargs:
                    raise ValueError(
                        "Must specify either 'std' or 'samples' in outliers configuration"
                    )

                if "embedding_keys" not in method_kwargs:
                    raise ValueError(
                        "'embedding_keys' must be specified in outliers configuration"
                    )

            if method == "custom" and self.samples is None:
                raise ValueError("Must specify 'samples' for custom sampling")

            return self

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
        super().syntax_check()
        
        # Additional validation for stratify_key when it's a list
        if self.config["method"] == "stratify":
            stratify_key = self.config.get("method_kwargs", {}).get("stratify_key")
            if isinstance(stratify_key, list):
                # Check if every key is mentioned as a "doc_id_key" in the pipeline
                pipeline_config = self.runner.config
                doc_id_keys = set()
                
                # Collect all doc_id_key values from the pipeline operations
                if 'operations' in pipeline_config:
                    for op in pipeline_config['operations']:
                        if 'doc_id_key' in op:
                            doc_id_keys.add(op['doc_id_key'])
                
                # Check if all stratify keys are mentioned as doc_id_keys
                for key in stratify_key:
                    if key not in doc_id_keys:
                        print(f"Warning: stratify_key '{key}' is not found as a doc_id_key in any pipeline operation")
                        print(f"Available doc_id_keys in pipeline: {sorted(doc_id_keys) if doc_id_keys else 'None'}")

    def execute(
        self, input_data: list[dict], is_build: bool = False
    ) -> tuple[list[dict], float]:
        """
        Executes the sample operation on the input data.

        Args:
            input_data (list[dict]): A list of dictionaries to process.
            is_build (bool): Whether the operation is being executed
              in the build phase. Defaults to False.

        Returns:
            tuple[list[dict], float]: A tuple containing the filtered
              list of dictionaries and the total cost of the operation.
        """
        cost = 0
        if not input_data:
            return [], cost

        if self.config["method"] == "first":
            return input_data[: self.config["samples"]], cost

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
                if self.config["method"] == "stratify":
                    method_kwargs = self.config.get("method_kwargs", {})
                    stratify_key = method_kwargs["stratify_key"]
                    samples_per_group = method_kwargs.get("samples_per_group", False)

                    # Helper function to get stratify value(s) for an item
                    def get_stratify_value(item):
                        if isinstance(stratify_key, str):
                            return item[stratify_key]
                        else:  # list of keys
                            return tuple(item[key] for key in stratify_key)

                    if samples_per_group:
                        # Sample N items from each group
                        import random
                        from collections import defaultdict

                        # Group data by stratify key(s)
                        groups = defaultdict(list)
                        for item in input_data:
                            groups[get_stratify_value(item)].append(item)

                        # Sample from each group
                        output_data = []
                        random_state = self.config.get("random_state", None)
                        if random_state is not None:
                            random.seed(random_state)

                        for group_items in groups.values():
                            if not isinstance(samples, int):
                                group_samples = int(samples * len(group_items))
                            else:
                                group_samples = min(samples, len(group_items))

                            sampled_items = random.sample(group_items, group_samples)
                            output_data.extend(sampled_items)
                    else:
                        # Original stratified sampling behavior
                        stratify = [get_stratify_value(data) for data in input_data]

                        import sklearn.model_selection

                        output_data, _ = sklearn.model_selection.train_test_split(
                            input_data,
                            train_size=samples,
                            random_state=self.config.get("random_state", None),
                            stratify=stratify,
                        )
                else:
                    import sklearn.model_selection

                    output_data, _ = sklearn.model_selection.train_test_split(
                        input_data,
                        train_size=samples,
                        random_state=self.config.get("random_state", None),
                    )

        return output_data, cost
