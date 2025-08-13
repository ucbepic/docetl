from typing import Any, Literal, Union
import random
from collections import defaultdict

import numpy as np
from pydantic import Field, field_validator, model_validator

from docetl.operations.base import BaseOperation
from docetl.operations.clustering_utils import get_embeddings_for_clustering


class SampleOperation(BaseOperation):
    """
    Samples items from input data using various methods.
    
    Params:
    - method: "uniform", "stratify", "outliers", "custom", "first"
    - samples: int, float, or list depending on method
    - method_kwargs: dict with method-specific parameters:
        For stratify:
            - stratify_key: str or list[str] - Keys to stratify by
            - samples_per_group: bool - Sample N items per group vs. dividing total
        For outliers:
            - embedding_keys: list[str] - Keys for embeddings
            - std: float - Std deviation cutoff
            - samples: int/float - Number of outliers
            - keep: bool - Keep outliers (True) or remove them (False)
            - center: dict - Optional center point
    """

    class schema(BaseOperation.schema):
        type: str = "sample"
        method: Literal["uniform", "stratify", "outliers", "custom", "first"]
        samples: Union[int, float, list] | None = None
        method_kwargs: dict[str, Any] | None = Field(default_factory=dict)
        random_state: int | None = Field(None, ge=0)

        @field_validator("samples")
        def validate_samples(cls, v, info):
            if v is None:
                return v
                
            method = info.data.get("method") if hasattr(info, "data") else None
            
            if method == "custom":
                if not isinstance(v, list):
                    raise TypeError("'samples' must be a list for custom sampling")
            elif isinstance(v, (int, float)):
                if v <= 0:
                    raise ValueError("'samples' must be a positive number")
                    
            return v

        @field_validator("method_kwargs")
        def validate_method_kwargs(cls, v, info):
            if v is None:
                return {}
                
            if not isinstance(v, dict):
                raise TypeError("'method_kwargs' must be a dictionary")

            # Get method from context if available
            method = info.data.get("method") if hasattr(info, "data") else None

            # Validate stratify-specific kwargs
            if "stratify_key" in v:
                stratify_key = v["stratify_key"]
                if isinstance(stratify_key, str):
                    pass  # Single key is valid
                elif isinstance(stratify_key, list):
                    if not stratify_key:
                        raise ValueError("'stratify_key' list cannot be empty")
                    if not all(isinstance(key, str) for key in stratify_key):
                        raise TypeError("All items in 'stratify_key' must be strings")
                else:
                    raise TypeError("'stratify_key' must be a string or list of strings")

            if "samples_per_group" in v and not isinstance(v["samples_per_group"], bool):
                raise TypeError("'samples_per_group' must be a boolean")

            # Validate outlier-specific kwargs
            if "embedding_keys" in v:
                if not isinstance(v["embedding_keys"], list) or not all(
                    isinstance(key, str) for key in v["embedding_keys"]
                ):
                    raise TypeError("'embedding_keys' must be a list of strings")

            if "std" in v:
                if not isinstance(v["std"], (int, float)) or v["std"] <= 0:
                    raise ValueError("'std' must be a positive number")

            if "samples" in v:
                if not isinstance(v["samples"], (int, float)) or v["samples"] <= 0:
                    raise ValueError("'samples' in method_kwargs must be a positive number")

            if "center" in v and not isinstance(v["center"], dict):
                raise TypeError("'center' must be a dictionary")

            return v

        @model_validator(mode="after")
        def validate_method_specific_requirements(self):
            """Validate that required parameters are present for each method."""
            method = self.method
            method_kwargs = self.method_kwargs or {}

            # Methods that require samples parameter
            if method in ["uniform", "stratify", "first"] and self.samples is None:
                raise ValueError(f"Must specify 'samples' for {method} sampling")

            # Stratify method requirements
            if method == "stratify" and "stratify_key" not in method_kwargs:
                raise ValueError("Must specify 'stratify_key' for stratify sampling")

            # Outliers method requirements
            if method == "outliers":
                if "std" not in method_kwargs and "samples" not in method_kwargs:
                    raise ValueError(
                        "Must specify either 'std' or 'samples' in outliers configuration"
                    )
                if "embedding_keys" not in method_kwargs:
                    raise ValueError(
                        "'embedding_keys' must be specified in outliers configuration"
                    )

            # Custom method requirements
            if method == "custom" and self.samples is None:
                raise ValueError("Must specify 'samples' for custom sampling")

            return self

        @model_validator(mode="after")
        def validate_stratify_keys_in_pipeline(self):
            """Warn if stratify keys are not found as doc_id_keys in pipeline."""
            if self.method == "stratify" and hasattr(self, "_runner"):
                stratify_key = self.method_kwargs.get("stratify_key")
                if isinstance(stratify_key, list):
                    # Check pipeline config for doc_id_keys
                    pipeline_config = getattr(self._runner, "config", {})
                    doc_id_keys = set()
                    
                    for op in pipeline_config.get("operations", []):
                        if "doc_id_key" in op:
                            doc_id_keys.add(op["doc_id_key"])
                    
                    # Warn about missing keys
                    for key in stratify_key:
                        if key not in doc_id_keys:
                            print(f"Warning: stratify_key '{key}' is not found as a doc_id_key in any pipeline operation")
                            if doc_id_keys:
                                print(f"Available doc_id_keys: {sorted(doc_id_keys)}")
            
            return self

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Store runner reference for validation
        if hasattr(self, "runner"):
            self.schema._runner = self.runner

    def execute(
        self, input_data: list[dict], is_build: bool = False
    ) -> tuple[list[dict], float]:
        """
        Execute the sample operation on input data.

        Args:
            input_data: List of dictionaries to sample from
            is_build: Whether this is a build phase execution

        Returns:
            Tuple of (sampled_data, cost)
        """
        if not input_data:
            return [], 0.0

        method = self.config["method"]
        
        # Dispatch to appropriate sampling method
        if method == "first":
            return self._sample_first(input_data)
        elif method == "uniform":
            return self._sample_uniform(input_data)
        elif method == "stratify":
            return self._sample_stratify(input_data)
        elif method == "outliers":
            return self._sample_outliers(input_data)
        elif method == "custom":
            return self._sample_custom(input_data)
        else:
            raise ValueError(f"Unknown sampling method: {method}")

    def _sample_first(self, input_data: list[dict]) -> tuple[list[dict], float]:
        """Take the first N samples."""
        n_samples = self.config["samples"]
        return input_data[:n_samples], 0.0

    def _sample_uniform(self, input_data: list[dict]) -> tuple[list[dict], float]:
        """Uniformly sample from input data."""
        import sklearn.model_selection
        
        output_data, _ = sklearn.model_selection.train_test_split(
            input_data,
            train_size=self.config["samples"],
            random_state=self.config.get("random_state"),
        )
        return output_data, 0.0

    def _sample_stratify(self, input_data: list[dict]) -> tuple[list[dict], float]:
        """Stratified sampling based on one or more keys."""
        method_kwargs = self.config.get("method_kwargs", {})
        stratify_key = method_kwargs["stratify_key"]
        samples_per_group = method_kwargs.get("samples_per_group", False)
        samples = self.config["samples"]
        
        # Helper to get stratify value(s)
        def get_stratify_value(item):
            if isinstance(stratify_key, str):
                return item[stratify_key]
            else:  # list of keys
                return tuple(item[key] for key in stratify_key)

        if samples_per_group:
            return self._sample_per_group(input_data, get_stratify_value, samples)
        else:
            return self._sample_stratified(input_data, get_stratify_value, samples)

    def _sample_per_group(
        self, input_data: list[dict], get_group_fn, samples
    ) -> tuple[list[dict], float]:
        """Sample N items from each group."""
        # Group data
        groups = defaultdict(list)
        for item in input_data:
            groups[get_group_fn(item)].append(item)

        # Set random seed if specified
        random_state = self.config.get("random_state")
        if random_state is not None:
            random.seed(random_state)

        # Sample from each group
        output_data = []
        for group_items in groups.values():
            if isinstance(samples, float):
                n_samples = int(samples * len(group_items))
            else:
                n_samples = min(samples, len(group_items))
            
            if n_samples > 0:
                sampled = random.sample(group_items, n_samples)
                output_data.extend(sampled)

        return output_data, 0.0

    def _sample_stratified(
        self, input_data: list[dict], get_group_fn, samples
    ) -> tuple[list[dict], float]:
        """Traditional stratified sampling."""
        import sklearn.model_selection
        
        stratify_values = [get_group_fn(item) for item in input_data]
        
        output_data, _ = sklearn.model_selection.train_test_split(
            input_data,
            train_size=samples,
            random_state=self.config.get("random_state"),
            stratify=stratify_values,
        )
        return output_data, 0.0

    def _sample_outliers(self, input_data: list[dict]) -> tuple[list[dict], float]:
        """Sample based on outlier detection using embeddings."""
        outliers_config = self.config.get("method_kwargs", {})
        
        # Get embeddings
        embeddings, cost = get_embeddings_for_clustering(
            input_data, outliers_config, self.runner.api
        )
        embeddings = np.array(embeddings)

        # Get or compute center
        if "center" in outliers_config:
            center_embeddings, center_cost = get_embeddings_for_clustering(
                [outliers_config["center"]], outliers_config, self.runner.api
            )
            cost += center_cost
            center = np.array(center_embeddings[0])
        else:
            center = embeddings.mean(axis=0)

        # Calculate distances
        distances = np.sqrt(((embeddings - center) ** 2).sum(axis=1))

        # Determine cutoff
        if "std" in outliers_config:
            cutoff = np.sqrt((embeddings.std(axis=0) ** 2).sum()) * outliers_config["std"]
        else:  # "samples" in method_kwargs
            distance_distribution = np.sort(distances)
            n_samples = outliers_config["samples"]
            if isinstance(n_samples, float):
                n_samples = int(n_samples * len(distance_distribution))
            cutoff = distance_distribution[min(n_samples, len(distance_distribution) - 1)]

        # Filter based on cutoff
        keep = outliers_config.get("keep", False)
        include = distances > cutoff if keep else distances <= cutoff
        
        output_data = [item for idx, item in enumerate(input_data) if include[idx]]
        return output_data, cost

    def _sample_custom(self, input_data: list[dict]) -> tuple[list[dict], float]:
        """Sample specific items based on provided keys."""
        samples = self.config["samples"]
        if not samples:
            return [], 0.0
            
        # Get keys from first sample
        keys = list(samples[0].keys())
        
        # Create lookup map
        key_to_doc = {
            tuple(doc.get(key) for key in keys): doc 
            for doc in input_data
        }
        
        # Extract requested samples
        output_data = []
        for sample in samples:
            lookup_key = tuple(sample.get(key) for key in keys)
            if lookup_key in key_to_doc:
                output_data.append(key_to_doc[lookup_key])
                
        return output_data, 0.0
