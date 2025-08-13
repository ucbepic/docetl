from typing import Any, Literal, Union
import random
from collections import defaultdict

import numpy as np
from pydantic import Field, field_validator, model_validator

from docetl.operations.base import BaseOperation
from docetl.operations.clustering_utils import get_embeddings_for_clustering


class SampleOperation(BaseOperation):
    """
    Samples items from input data using various methods with optional stratification.
    
    Params:
    - method: "uniform", "outliers", "custom", "first"
    - samples: int, float, or list depending on method
    - stratify_key: str or list[str] - Optional keys to stratify by (works with all methods except custom)
    - samples_per_group: bool - When stratifying, sample N items per group vs. dividing total
    - method_kwargs: dict with method-specific parameters:
        For outliers:
            - embedding_keys: list[str] - Keys for embeddings
            - std: float - Std deviation cutoff
            - samples: int/float - Number of outliers
            - keep: bool - Keep outliers (True) or remove them (False)
            - center: dict - Optional center point
    """

    class schema(BaseOperation.schema):
        type: str = "sample"
        method: Literal["uniform", "outliers", "custom", "first"]
        samples: Union[int, float, list] | None = None
        stratify_key: Union[str, list[str]] | None = None
        samples_per_group: bool = False
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

        @field_validator("stratify_key")
        def validate_stratify_key(cls, v):
            if v is None:
                return v
                
            if isinstance(v, str):
                pass  # Single key is valid
            elif isinstance(v, list):
                if not v:
                    raise ValueError("'stratify_key' list cannot be empty")
                if not all(isinstance(key, str) for key in v):
                    raise TypeError("All items in 'stratify_key' must be strings")
            else:
                raise TypeError("'stratify_key' must be a string or list of strings")
                
            return v

        @field_validator("method_kwargs")
        def validate_method_kwargs(cls, v, info):
            if v is None:
                return {}
                
            if not isinstance(v, dict):
                raise TypeError("'method_kwargs' must be a dictionary")

            # Get method from context if available
            method = info.data.get("method") if hasattr(info, "data") else None

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
            if method in ["uniform", "first"] and self.samples is None:
                raise ValueError(f"Must specify 'samples' for {method} sampling")

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
            if method == "custom":
                if self.samples is None:
                    raise ValueError("Must specify 'samples' for custom sampling")
                if self.stratify_key is not None:
                    raise ValueError("Stratification is not supported with custom sampling")
                    
            # Validate samples_per_group only makes sense with stratification
            if self.samples_per_group and not self.stratify_key:
                raise ValueError("'samples_per_group' requires 'stratify_key' to be set")

            return self

        @model_validator(mode="after")
        def validate_stratify_keys_in_pipeline(self):
            """Warn if stratify keys are not found as doc_id_keys in pipeline."""
            if self.stratify_key and hasattr(self, "_runner"):
                stratify_key = self.stratify_key
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
        stratify_key = self.config.get("stratify_key")
        
        # If stratification is requested, handle it here
        if stratify_key:
            return self._sample_with_stratification(input_data, method)
        
        # Otherwise, dispatch to appropriate sampling method
        if method == "first":
            return self._sample_first(input_data)
        elif method == "uniform":
            return self._sample_uniform(input_data)
        elif method == "outliers":
            return self._sample_outliers(input_data)
        elif method == "custom":
            return self._sample_custom(input_data)
        else:
            raise ValueError(f"Unknown sampling method: {method}")

    def _get_stratify_value(self, item: dict) -> Union[str, tuple]:
        """Get the stratification value(s) for an item."""
        stratify_key = self.config.get("stratify_key")
        if isinstance(stratify_key, str):
            return item[stratify_key]
        else:  # list of keys
            return tuple(item[key] for key in stratify_key)

    def _sample_with_stratification(
        self, input_data: list[dict], method: str
    ) -> tuple[list[dict], float]:
        """Apply stratification to any sampling method."""
        samples_per_group = self.config.get("samples_per_group", False)
        
        # Group data by stratification key(s)
        groups = defaultdict(list)
        for item in input_data:
            groups[self._get_stratify_value(item)].append(item)

        # Apply sampling method to each group
        output_data = []
        total_cost = 0.0
        
        if samples_per_group:
            # Sample N items from each group
            for group_items in groups.values():
                if method == "first":
                    sampled, cost = self._sample_first(group_items)
                elif method == "uniform":
                    sampled, cost = self._sample_uniform(group_items)
                elif method == "outliers":
                    sampled, cost = self._sample_outliers(group_items)
                else:
                    raise ValueError(f"Method {method} not supported with stratification")
                
                output_data.extend(sampled)
                total_cost += cost
        else:
            # Traditional stratified sampling - sample proportionally from each group
            if method == "uniform":
                return self._sample_stratified_proportional(input_data, groups)
            elif method == "first":
                # For "first" method with stratification, take proportional first items from each group
                return self._sample_stratified_first(input_data, groups)
            elif method == "outliers":
                # For outliers, we need to handle differently
                return self._sample_stratified_outliers(input_data, groups)
            else:
                raise ValueError(f"Method {method} not supported with proportional stratification")
        
        return output_data, total_cost

    def _sample_stratified_proportional(
        self, input_data: list[dict], groups: dict
    ) -> tuple[list[dict], float]:
        """Traditional stratified sampling with proportional allocation."""
        import sklearn.model_selection
        
        stratify_values = [self._get_stratify_value(item) for item in input_data]
        
        output_data, _ = sklearn.model_selection.train_test_split(
            input_data,
            train_size=self.config["samples"],
            random_state=self.config.get("random_state"),
            stratify=stratify_values,
        )
        return output_data, 0.0

    def _sample_stratified_first(
        self, input_data: list[dict], groups: dict
    ) -> tuple[list[dict], float]:
        """Take first N items proportionally from each stratum."""
        samples = self.config["samples"]
        if isinstance(samples, float):
            n_samples = int(samples * len(input_data))
        else:
            n_samples = samples
        
        # Calculate proportional samples per group
        output_data = []
        for group_key, group_items in groups.items():
            group_proportion = len(group_items) / len(input_data)
            group_samples = int(n_samples * group_proportion)
            if group_samples > 0:
                output_data.extend(group_items[:group_samples])
        
        # If we're short due to rounding, add more from largest group
        while len(output_data) < n_samples and len(output_data) < len(input_data):
            largest_group = max(groups.values(), key=len)
            for item in largest_group:
                if item not in output_data:
                    output_data.append(item)
                    break
        
        return output_data[:n_samples], 0.0

    def _sample_stratified_outliers(
        self, input_data: list[dict], groups: dict
    ) -> tuple[list[dict], float]:
        """Apply outlier detection within each stratum and combine results."""
        output_data = []
        total_cost = 0.0
        
        # Apply outlier detection to each group
        for group_items in groups.values():
            if len(group_items) > 0:
                sampled, cost = self._sample_outliers(group_items)
                output_data.extend(sampled)
                total_cost += cost
        
        return output_data, total_cost

    def _sample_first(self, input_data: list[dict]) -> tuple[list[dict], float]:
        """Take the first N samples."""
        samples = self.config["samples"]
        if isinstance(samples, float):
            n_samples = int(samples * len(input_data))
        else:
            n_samples = samples
        return input_data[:n_samples], 0.0

    def _sample_uniform(self, input_data: list[dict]) -> tuple[list[dict], float]:
        """Uniformly sample from input data."""
        import sklearn.model_selection
        
        samples = self.config["samples"]
        
        # Handle the case where we have very few items
        if isinstance(samples, int) and samples >= len(input_data):
            return input_data, 0.0
        
        output_data, _ = sklearn.model_selection.train_test_split(
            input_data,
            train_size=samples,
            random_state=self.config.get("random_state"),
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
