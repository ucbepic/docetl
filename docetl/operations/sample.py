from typing import Any, Literal, Union

import numpy as np
from pydantic import Field, field_validator, model_validator

from docetl.operations.base import BaseOperation
from docetl.operations.clustering_utils import get_embeddings_for_clustering


class SampleOperation(BaseOperation):
    """
    A sampling operation that can select a subset of items from input data.
    
    This operation supports multiple sampling methods:
    - uniform: Random uniform sampling
    - stratify: Stratified sampling based on a key
    - outliers: Sample based on embedding distance from center
    - custom: Sample specific items by matching keys
    - first: Take the first N items
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

                # Validate specific keys in method_kwargs
                if "stratify_key" in v and not isinstance(v["stratify_key"], str):
                    raise TypeError("'stratify_key' must be a string")

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

        method = self.config["method"]
        
        if method == "first":
            return self._sample_first(input_data), cost
        elif method == "outliers":
            return self._sample_outliers(input_data)
        elif method == "custom":
            return self._sample_custom(input_data), cost
        elif method == "uniform":
            return self._sample_uniform(input_data), cost
        elif method == "stratify":
            return self._sample_stratified(input_data), cost
        else:
            raise ValueError(f"Unknown sampling method: {method}")

    def _sample_first(self, input_data: list[dict]) -> list[dict]:
        """Take the first N items from the input data."""
        return input_data[: self.config["samples"]]

    def _sample_uniform(self, input_data: list[dict]) -> list[dict]:
        """Perform uniform random sampling."""
        import sklearn.model_selection
        
        output_data, _ = sklearn.model_selection.train_test_split(
            input_data,
            train_size=self.config["samples"],
            random_state=self.config.get("random_state", None),
            stratify=None,
        )
        return output_data

    def _sample_stratified(self, input_data: list[dict]) -> list[dict]:
        """Perform stratified sampling based on a key."""
        import sklearn.model_selection
        
        stratify_key = self.config.get("method_kwargs", {})["stratify_key"]
        stratify = [data[stratify_key] for data in input_data]
        
        output_data, _ = sklearn.model_selection.train_test_split(
            input_data,
            train_size=self.config["samples"],
            random_state=self.config.get("random_state", None),
            stratify=stratify,
        )
        return output_data

    def _sample_custom(self, input_data: list[dict]) -> list[dict]:
        """Sample specific items based on matching keys."""
        samples = self.config["samples"]
        keys = list(samples[0].keys())
        
        # Create a mapping from key tuples to documents
        key_to_doc = {
            tuple([doc[key] for key in keys]): doc for doc in input_data
        }
        
        # Find matching documents
        output_data = []
        for sample in samples:
            key_tuple = tuple([sample[key] for key in keys])
            if key_tuple in key_to_doc:
                output_data.append(key_to_doc[key_tuple])
        
        return output_data

    def _sample_outliers(self, input_data: list[dict]) -> tuple[list[dict], float]:
        """Sample based on embedding distance from a center point."""
        outliers_config = self.config.get("method_kwargs", {})
        
        # Get embeddings for all input data
        embeddings, embedding_cost = get_embeddings_for_clustering(
            input_data, outliers_config, self.runner.api
        )
        embeddings = np.array(embeddings)
        
        # Determine the center point
        if "center" in outliers_config:
            center_embeddings, center_cost = get_embeddings_for_clustering(
                [outliers_config["center"]], outliers_config, self.runner.api
            )
            center = np.array(center_embeddings[0])
            total_cost = embedding_cost + center_cost
        else:
            center = embeddings.mean(axis=0)
            total_cost = embedding_cost
        
        # Calculate distances from center
        distances = np.sqrt(((embeddings - center) ** 2).sum(axis=1))
        
        # Determine cutoff threshold
        if "std" in outliers_config:
            # Use standard deviation based cutoff
            cutoff = (
                np.sqrt((embeddings.std(axis=0) ** 2).sum())
                * outliers_config["std"]
            )
        else:  # "samples" in config
            # Use percentile based cutoff
            distance_distribution = np.sort(distances)
            samples = outliers_config["samples"]
            if isinstance(samples, float):
                samples = int(samples * (len(distance_distribution) - 1))
            cutoff = distance_distribution[samples]
        
        # Determine which items to include
        keep = outliers_config.get("keep", False)
        include = distances > cutoff if keep else distances <= cutoff
        
        output_data = [item for idx, item in enumerate(input_data) if include[idx]]
        
        return output_data, total_cost
