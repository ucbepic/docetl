"""
Pandas DataFrame accessor that provides semantic operations using large language models.

This accessor adds semantic capabilities to pandas DataFrames through the .semantic namespace,
enabling LLM-powered operations like mapping, filtering, merging, aggregation, splitting, gathering, and unnesting.

Basic Usage:
    >>> import pandas as pd
    >>> df = pd.DataFrame({"text": ["Apple is a tech company", "Microsoft makes Windows"]})

    # Configure the accessor with key-value pairs you would use in a DocETL pipeline config (https://ucbepic.github.io/docetl/concepts/pipelines/)
    >>> df.semantic.set_config(default_model="gpt-4o-mini")

    # Semantic mapping
    >>> df.semantic.map(
    ...     prompt="Extract company name from: {{input.text}}",
    ...     output_schema={"company": "str"}
    ... )

Documentation Links:
- Map Operation: https://ucbepic.github.io/docetl/operators/map/
- Filter Operation: https://ucbepic.github.io/docetl/operators/filter/
- Resolve Operation: https://ucbepic.github.io/docetl/operators/resolve/
- Reduce Operation: https://ucbepic.github.io/docetl/operators/reduce/
- Split Operation: https://ucbepic.github.io/docetl/operators/split/
- Gather Operation: https://ucbepic.github.io/docetl/operators/gather/
- Unnest Operation: https://ucbepic.github.io/docetl/operators/unnest/

Cost Tracking:
    All operations track their LLM usage costs:
    >>> df.semantic.total_cost  # Returns total cost in USD
    >>> df.semantic.history     # Returns operation history
"""

from typing import Any, NamedTuple

import pandas as pd
from rich.panel import Panel

from docetl.operations.equijoin import EquijoinOperation
from docetl.operations.filter import FilterOperation
from docetl.operations.gather import GatherOperation
from docetl.operations.map import MapOperation
from docetl.operations.reduce import ReduceOperation
from docetl.operations.resolve import ResolveOperation
from docetl.operations.split import SplitOperation
from docetl.operations.unnest import UnnestOperation
from docetl.optimizer import Optimizer
from docetl.optimizers.join_optimizer import JoinOptimizer
from docetl.runner import DSLRunner


class OpHistory(NamedTuple):
    """Record of an operation that was run."""

    op_type: str  # 'map', 'filter', 'merge', 'agg', 'split', 'gather', 'unnest'
    config: dict[str, Any]  # Full config used
    output_columns: list[str]  # Columns created/modified


@pd.api.extensions.register_dataframe_accessor("semantic")
class SemanticAccessor:
    def __init__(self, df: pd.DataFrame):
        self._df = df
        # Get history and costs from parent DataFrame if it exists
        self._history = df.attrs.get("_semantic_history", []).copy()
        self._costs = df.attrs.get("_semantic_costs", 0.0)

        config = df.attrs.get(
            "_semantic_config",
            {
                "default_model": "gpt-4o-mini",
                "operations": [],
                "datasets": {},
                "pipeline": {"steps": []},
            },
        )

        # Initialize runner
        self.runner = DSLRunner(config, from_df_accessors=True)

        builder = Optimizer(
            self.runner,
        )
        self.runner.optimizer = builder

    def set_config(self, **config):
        """
        Configure the semantic accessor with custom settings.

        Args:
            **config: Configuration options including:
                - default_model: Default LLM model to use
                - max_threads: Maximum number of concurrent threads
                - other DocETL configuration options
        """
        self.runner.config.update(config)

        builder = Optimizer(self.runner)
        self.runner.optimizer = builder

    def _record_operation(
        self, data: list[dict], op_type: str, config: dict[str, Any], cost: float
    ) -> pd.DataFrame:
        """Record an operation and return the history entry."""
        # Find new columns by comparing with current DataFrame
        result_df = pd.DataFrame(data)
        new_cols = list(set(result_df.columns) - set(self._df.columns))

        # Make a serializable copy of the config
        serializable_config = self._make_serializable(config)

        entry = OpHistory(op_type, serializable_config, new_cols)
        self._history.append(entry)
        self._costs += cost

        # Store history and costs in the DataFrame's attrs dictionary
        result_df.attrs["_semantic_history"] = self._history
        result_df.attrs["_semantic_costs"] = self._costs
        # Make a serializable copy of the runner config
        result_df.attrs["_semantic_config"] = self._make_serializable(
            self.runner.config
        )
        return result_df

    def _make_serializable(self, obj: Any) -> Any:
        """Recursively convert an object to be serializable/picklable.

        Handles:
        - dict_keys/dict_values/dict_items -> list
        - dictionaries -> dictionaries with serializable values
        - lists/tuples -> lists/tuples with serializable values
        - other objects are returned as-is
        """
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(x) for x in obj]
        elif isinstance(obj, (type({}.keys()), type({}.values()), type({}.items()))):
            return list(obj)
        else:
            return obj

    def _get_column_history(self, column: str) -> list[OpHistory]:
        """Get history of operations that created/modified a column."""
        return [op for op in self._history if column in op.output_columns]

    def _synthesize_comparison_context(self, keys: list[str]) -> str:
        """Generate context about how the keys were created, if they were."""
        context_parts = []

        for key in keys:
            history = self._get_column_history(key)
            if history:
                # Get the most recent operation that created/modified this key
                last_op = history[-1]
                if "prompt" in last_op.config:
                    last_op_prompt = last_op.config["prompt"]
                    # if any {{ }} in the prompt, replace them with just { }
                    last_op_prompt = last_op_prompt.replace("{{", "{").replace(
                        "}}", "}"
                    )

                    context_parts.append(
                        f"The field '{key}' was created using this prompt: {last_op_prompt}"
                    )

        if context_parts:
            return "\n\nContext about these fields:\n" + "\n".join(context_parts)
        return ""

    def map(
        self,
        prompt: str,
        output: dict[str, Any] = None,
        *,
        output_schema: dict[str, Any] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Apply semantic mapping to each row using a language model.

        Documentation: https://ucbepic.github.io/docetl/operators/map/

        Args:
            prompt: Jinja template string for generating prompts. Use {{input.column_name}}
                   to reference input columns.
            output: Dictionary containing output configuration with keys:
                   - "schema": Dictionary defining the expected output structure and types.
                              Example: {"entities": "list[str]", "sentiment": "str"}
                   - "mode": Optional output mode. Either "tools" (default) or "structured_output".
                            "structured_output" uses native JSON schema mode for supported models.
                   - "n": Optional number of outputs to generate for each input (synthetic data generation)
            output_schema: DEPRECATED. Use 'output' parameter instead.
                          Dictionary defining the expected output structure for backward compatibility.
            **kwargs: Additional configuration options:
                - model: LLM model to use (default: from config)
                - batch_prompt: Template for processing multiple documents in a single prompt
                - max_batch_size: Maximum number of documents to process in a single batch
                - optimize: Flag to enable operation optimization (default: True)
                - recursively_optimize: Flag to enable recursive optimization (default: False)
                - sample: Number of samples to use for the operation
                - tools: List of tool definitions for LLM use
                - validate: List of Python expressions to validate output
                - num_retries_on_validate_failure: Number of retry attempts (default: 0)
                - gleaning: Configuration for LLM-based refinement
                - drop_keys: List of keys to drop from input
                - timeout: Timeout for each LLM call in seconds (default: 120)
                - max_retries_per_timeout: Maximum retries per timeout (default: 2)
                - litellm_completion_kwargs: Additional parameters for LiteLLM
                - skip_on_error: Skip operation if LLM returns error (default: False)
                - bypass_cache: Bypass cache for this operation (default: False)
                - n: Number of outputs to generate for each input (synthetic data generation)

        Returns:
            pd.DataFrame: A new DataFrame containing the transformed data with columns
                         matching the output schema.

        Examples:
            >>> # Extract entities and sentiment
            >>> df.semantic.map(
            ...     prompt="Analyze this text: {{input.text}}",
            ...     output={
            ...         "schema": {
            ...             "entities": "list[str]",
            ...             "sentiment": "str"
            ...         }
            ...     },
            ...     validate=["len(output['entities']) <= 5"],
            ...     num_retries_on_validate_failure=2
            ... )

            >>> # Generate synthetic data with multiple variations per input
            >>> df.semantic.map(
            ...     prompt="Create a headline for: {{input.topic}}",
            ...     output={"schema": {"headline": "str"}, "n": 5}
            ... )

            >>> # Use structured output mode for better JSON schema support
            >>> df.semantic.map(
            ...     prompt="Extract structured data: {{input.text}}",
            ...     output={
            ...         "schema": {"name": "str", "age": "int", "tags": "list[str]"},
            ...         "mode": "structured_output"
            ...     }
            ... )
        """
        # Convert DataFrame to list of dicts for DocETL
        input_data = self._df.to_dict("records")

        # Handle backward compatibility: if output_schema is provided, convert it to output format
        if output_schema is not None and output is None:
            output = {"schema": output_schema}
            if "n" in kwargs:
                output["n"] = kwargs.pop("n")
        elif output is None and output_schema is None:
            raise ValueError("Either 'output' or 'output_schema' must be provided")
        elif output is not None and output_schema is not None:
            raise ValueError(
                "Cannot provide both 'output' and 'output_schema' parameters"
            )

        # Validate output parameter
        if not isinstance(output, dict):
            raise ValueError("output must be a dictionary")

        if "schema" not in output:
            raise ValueError("output must contain a 'schema' key")

        # Validate output mode if provided
        output_mode = output.get("mode", "tools")
        if output_mode not in ["tools", "structured_output"]:
            raise ValueError(
                f"output mode must be 'tools' or 'structured_output', got '{output_mode}'"
            )

        # Create map operation config
        map_config = {
            "type": "map",
            "name": f"semantic_map_{len(self._history)}",
            "prompt": prompt,
            "output": output,
            **kwargs,
        }

        # Create and execute map operation
        map_op = MapOperation(
            runner=self.runner,
            config=map_config,
            default_model=self.runner.config["default_model"],
            max_threads=self.runner.max_threads,
            console=self.runner.console,
            status=self.runner.status,
        )
        results, cost = map_op.execute(input_data)

        return self._record_operation(results, "map", map_config, cost)

    def merge(
        self,
        right: pd.DataFrame,
        comparison_prompt: str,
        *,
        fuzzy: bool = False,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Semantically merge two DataFrames based on flexible matching criteria.

        Documentation: https://ucbepic.github.io/docetl/operators/equijoin/

        When fuzzy=True and no blocking parameters are provided, this method automatically
        invokes the JoinOptimizer to generate efficient blocking conditions. The optimizer
        will suggest blocking thresholds and conditions to improve performance while
        maintaining the target recall. The optimized configuration will be displayed
        for future reuse.

        Args:
            right: Right DataFrame to merge with
            comparison_prompt: Prompt template for comparing records
            fuzzy: Whether to use fuzzy matching with optimization (default: False)
            **kwargs: Additional configuration options:
                - model: LLM model to use
                - blocking_threshold: Threshold for blocking optimization
                - blocking_conditions: Custom blocking conditions
                - target_recall: Target recall for optimization (default: 0.95)
                - estimated_selectivity: Estimated match rate
                - validate: List of validation expressions
                - num_retries_on_validate_failure: Number of retries
                - timeout: Timeout in seconds (default: 120)
                - max_retries_per_timeout: Max retries per timeout (default: 2)

        Returns:
            pd.DataFrame: Merged DataFrame containing matched records

        Examples:
            >>> # Simple merge
            >>> merged_df = df1.semantic.merge(
            ...     df2,
            ...     comparison_prompt="Are these records about the same entity? {{input1}} vs {{input2}}"
            ... )

            >>> # Fuzzy merge with automatic optimization
            >>> merged_df = df1.semantic.merge(
            ...     df2,
            ...     comparison_prompt="Compare: {{input1}} vs {{input2}}",
            ...     fuzzy=True,
            ...     target_recall=0.9
            ... )

            >>> # Fuzzy merge with manual blocking parameters
            >>> merged_df = df1.semantic.merge(
            ...     df2,
            ...     comparison_prompt="Compare: {{input1}} vs {{input2}}",
            ...     fuzzy=False,
            ...     blocking_threshold=0.8,
            ...     blocking_conditions=["input1.category == input2.category"]
            ... )
        """
        # Convert DataFrames to lists of dicts
        left_data = self._df.to_dict("records")
        right_data = right.to_dict("records")

        # Create equijoin operation config
        join_config = {
            "type": "equijoin",
            "name": f"semantic_merge_{len(self._history)}",
            "comparison_prompt": comparison_prompt,
            **kwargs,
        }

        # If fuzzy matching and no blocking params provided, use JoinOptimizer
        if (
            fuzzy
            and not kwargs.get("blocking_threshold")
            and not kwargs.get("blocking_conditions")
        ):
            join_optimizer = JoinOptimizer(
                self.runner,
                join_config,
                target_recall=kwargs.get("target_recall", 0.95),
                estimated_selectivity=kwargs.get("estimated_selectivity", None),
            )
            optimized_config, optimizer_cost, _ = join_optimizer.optimize_equijoin(
                left_data, right_data, skip_map_gen=True, skip_containment_gen=True
            )

            # Print optimized config for reuse
            self.runner.console.log(
                Panel.fit(
                    "[bold cyan]Optimized Configuration for Merge[/bold cyan]\n"
                    "[yellow]Consider adding these parameters to avoid re-running optimization:[/yellow]\n\n"
                    f"blocking_threshold: {optimized_config.get('blocking_threshold')}\n"
                    f"blocking_keys: {optimized_config.get('blocking_keys')}\n"
                    f"blocking_conditions: {optimized_config.get('blocking_conditions', [])}\n",
                    title="Optimization Results",
                )
            )
            join_config = optimized_config
            optimizer_cost_value = optimizer_cost
        else:
            optimizer_cost_value = 0.0

        # Create and execute equijoin operation
        join_op = EquijoinOperation(
            runner=self.runner,
            config=join_config,
            default_model=self.runner.config["default_model"],
            max_threads=self.runner.max_threads,
            console=self.runner.console,
            status=self.runner.status,
        )
        results, cost = join_op.execute(left_data, right_data)

        return self._record_operation(
            results, "equijoin", join_config, cost + optimizer_cost_value
        )

    def agg(
        self,
        *,
        # Reduction phase params (required)
        reduce_prompt: str,
        output: dict[str, Any] | None = None,
        output_schema: dict[str, Any] | None = None,
        # Resolution and reduce phase params (optional)
        fuzzy: bool = False,
        comparison_prompt: str | None = None,
        resolution_prompt: str | None = None,
        resolution_output: dict[str, Any] | None = None,
        resolution_output_schema: dict[str, Any] | None = None,
        reduce_keys: str | list[str] = ["_all"],
        resolve_kwargs: dict[str, Any] = {},
        reduce_kwargs: dict[str, Any] = {},
    ) -> pd.DataFrame:
        """
        Semantically aggregate data with optional fuzzy matching.

        Documentation:
        - Resolve Operation: https://ucbepic.github.io/docetl/operators/resolve/
        - Reduce Operation: https://ucbepic.github.io/docetl/operators/reduce/

        When fuzzy=True and no blocking parameters are provided in resolve_kwargs,
        this method automatically invokes the JoinOptimizer to generate efficient
        blocking conditions for the resolve phase. The optimizer will suggest
        blocking thresholds and conditions to improve performance while maintaining
        the target recall. The optimized configuration will be displayed for future reuse.

        The resolve phase is skipped if:
        - fuzzy=False
        - reduce_keys=["_all"]
        - input data has 5 or fewer rows

        Args:
            reduce_prompt: Prompt template for the reduction phase
            output: Output configuration with keys:
                - "schema": Dictionary defining the expected output structure and types
                - "mode": Optional output mode. Either "tools" (default) or "structured_output"
            output_schema: DEPRECATED. Use 'output' parameter instead. Backward-compatible schema dict
            fuzzy: Whether to use fuzzy matching for resolution (default: False)
            comparison_prompt: Prompt template for comparing records during resolution
            resolution_prompt: Prompt template for resolving conflicts
            resolution_output: Output configuration for resolution (new API with schema key)
            resolution_output_schema: Schema for resolution output (deprecated, use resolution_output)
            reduce_keys: Keys to group by for reduction (default: ["_all"])
            resolve_kwargs: Additional kwargs for resolve operation:
                - model: LLM model to use
                - blocking_threshold: Threshold for blocking optimization
                - blocking_conditions: Custom blocking conditions
                - target_recall: Target recall for optimization (default: 0.95)
                - estimated_selectivity: Estimated match rate
                - validate: List of validation expressions
                - num_retries_on_validate_failure: Number of retries
                - timeout: Timeout in seconds (default: 120)
                - max_retries_per_timeout: Max retries per timeout (default: 2)
            reduce_kwargs: Additional kwargs for reduce operation:
                - model: LLM model to use
                - validate: List of validation expressions
                - num_retries_on_validate_failure: Number of retries
                - timeout: Timeout in seconds (default: 120)
                - max_retries_per_timeout: Max retries per timeout (default: 2)

        Returns:
            pd.DataFrame: Aggregated DataFrame with columns matching output['schema']

        Examples:
            >>> # Simple aggregation
            >>> df.semantic.agg(
            ...     reduce_prompt="Summarize these items: {{input.text}}",
            ...     output={"schema": {"summary": "str"}}
            ... )

            >>> # Fuzzy matching with automatic optimization
            >>> df.semantic.agg(
            ...     reduce_prompt="Combine these items: {{input.text}}",
            ...     output={"schema": {"combined": "str"}},
            ...     fuzzy=True,
            ...     comparison_prompt="Are these items similar: {{input1.text}} vs {{input2.text}}",
            ...     resolution_prompt="Resolve conflicts between: {{items}}",
            ...     resolution_output={"schema": {"resolved": "str"}}
            ... )

            >>> # Fuzzy matching with manual blocking parameters
            >>> df.semantic.agg(
            ...     reduce_prompt="Combine these items: {{input.text}}",
            ...     output={"schema": {"combined": "str"}},
            ...     fuzzy=False,
            ...     comparison_prompt="Compare items: {{input1.text}} vs {{input2.text}}",
            ...     resolve_kwargs={
            ...         "blocking_threshold": 0.8,
            ...         "blocking_conditions": ["input1.category == input2.category"]
            ...     }
            ... )
        """
        # Convert DataFrame to list of dicts
        input_data = self._df.to_dict("records")

        # Handle backward compatibility: if output_schema is provided, convert it to output format
        if output_schema is not None and output is None:
            output = {"schema": output_schema}
        elif output is None and output_schema is None:
            raise ValueError("Either 'output' or 'output_schema' must be provided")
        elif output is not None and output_schema is not None:
            raise ValueError(
                "Cannot provide both 'output' and 'output_schema' parameters"
            )

        # Validate output parameter
        if not isinstance(output, dict):
            raise ValueError("output must be a dictionary")
        if "schema" not in output:
            raise ValueError("output must contain a 'schema' key")

        # Handle backward compatibility for resolution_output_schema
        if resolution_output_schema is not None and resolution_output is None:
            resolution_output = {"schema": resolution_output_schema}
        elif resolution_output is not None and resolution_output_schema is not None:
            raise ValueError(
                "Cannot provide both 'resolution_output' and 'resolution_output_schema' parameters"
            )

        # Validate output mode if provided
        output_mode = output.get("mode", "tools")
        if output_mode not in ["tools", "structured_output"]:
            raise ValueError(
                f"output mode must be 'tools' or 'structured_output', got '{output_mode}'"
            )

        # Change keys to list
        if isinstance(reduce_keys, str):
            reduce_keys = [reduce_keys]

        # Skip resolution if using _all or fuzzy is False
        if reduce_keys == ["_all"] or not fuzzy or len(input_data) <= 5:
            resolved_data = input_data
        else:
            # Synthesize comparison prompt if not provided
            if comparison_prompt is None:
                # Build record template from reduce_keys
                record_template = ", ".join(
                    f"{key}: {{{{ input{0}.{key} }}}}" for key in reduce_keys
                )

                # Add context about how fields were created
                context = self._synthesize_comparison_context(reduce_keys)

                comparison_prompt = f"""Do the following two records represent the same concept? Your answer should be true or false.{context}

Record 1: {record_template.replace('input0', 'input1')}
Record 2: {record_template.replace('input0', 'input2')}"""

            # Configure resolution
            resolve_config = {
                "type": "resolve",
                "name": f"semantic_resolve_{len(self._history)}",
                "comparison_prompt": comparison_prompt,
                **resolve_kwargs,
            }

            # Add resolution prompt and schema if provided
            if resolution_prompt:
                resolve_config["resolution_prompt"] = resolution_prompt
                if resolution_output:
                    # Use the new resolution_output format
                    resolve_config["output"] = resolution_output
                    if "keys" not in resolve_config["output"]:
                        # Add keys from schema if not explicitly provided
                        resolve_config["output"]["keys"] = list(
                            resolution_output["schema"].keys()
                        )
                else:
                    # No resolution output provided, use reduce_keys
                    resolve_config["output"] = {"keys": reduce_keys}
            else:
                resolve_config["output"] = {"keys": reduce_keys}

            # If blocking params not provided, use JoinOptimizer to synthesize them
            if not resolve_kwargs or (
                "blocking_threshold" not in resolve_kwargs
                and "blocking_conditions" not in resolve_kwargs
            ):
                join_optimizer = JoinOptimizer(
                    self.runner,
                    resolve_config,
                    target_recall=(
                        resolve_kwargs.get("target_recall", 0.95)
                        if resolve_kwargs
                        else 0.95
                    ),
                    estimated_selectivity=(
                        resolve_kwargs.get("estimated_selectivity", None)
                        if resolve_kwargs
                        else None
                    ),
                )
                optimized_config, optimizer_cost = join_optimizer.optimize_resolve(
                    input_data
                )

                # Print optimized config for reuse
                self.runner.console.log(
                    Panel.fit(
                        "[bold cyan]Optimized Configuration for Resolve[/bold cyan]\n"
                        "[yellow]Consider adding these parameters to avoid re-running optimization:[/yellow]\n\n"
                        f"blocking_threshold: {optimized_config.get('blocking_threshold')}\n"
                        f"blocking_keys: {optimized_config.get('blocking_keys')}\n"
                        f"blocking_conditions: {optimized_config.get('blocking_conditions', [])}\n",
                        title="Optimization Results",
                    )
                )
            else:
                # Use provided blocking params
                optimized_config = resolve_config.copy()
                optimizer_cost = 0.0

            # Execute resolution with optimized config
            resolve_op = ResolveOperation(
                runner=self.runner,
                config=optimized_config,
                default_model=self.runner.config["default_model"],
                max_threads=self.runner.max_threads,
                console=self.runner.console,
                status=self.runner.status,
            )
            resolved_data, resolve_cost = resolve_op.execute(input_data)
            _ = self._record_operation(
                resolved_data,
                "resolve",
                optimized_config,
                resolve_cost + optimizer_cost,
            )

        # Configure reduction
        reduce_config = {
            "type": "reduce",
            "name": f"semantic_reduce_{len(self._history)}",
            "reduce_key": reduce_keys,
            "prompt": reduce_prompt,
            "output": output,
            **reduce_kwargs,
        }

        # Execute reduction
        reduce_op = ReduceOperation(
            runner=self.runner,
            config=reduce_config,
            default_model=self.runner.config["default_model"],
            max_threads=self.runner.max_threads,
            console=self.runner.console,
            status=self.runner.status,
        )
        results, reduce_cost = reduce_op.execute(resolved_data)

        return self._record_operation(results, "reduce", reduce_config, reduce_cost)

    def reduce(
        self,
        prompt: str,
        output: dict[str, Any] | None = None,
        *,
        output_schema: dict[str, Any] | None = None,
        reduce_keys: str | list[str] = ["_all"],
        **kwargs,
    ) -> pd.DataFrame:
        """
        Reduce/aggregate all rows using a language model.

        This is a simplified wrapper around the agg() method for common reduce operations.

        Documentation: https://ucbepic.github.io/docetl/operators/reduce/

        Args:
            prompt: Jinja template string for the reduction prompt. Use {% for item in inputs %}
                   to iterate over input rows.
            output: Output configuration with keys:
                - "schema": Dictionary defining the expected output structure and types
                - "mode": Optional output mode. Either "tools" (default) or "structured_output"
            output_schema: DEPRECATED. Use 'output' parameter instead.
            reduce_keys: Keys to group by for reduction (default: ["_all"] for all rows)
            **kwargs: Additional configuration options passed to agg():
                - model: LLM model to use
                - validate: List of validation expressions
                - num_retries_on_validate_failure: Number of retries
                - timeout: Timeout in seconds (default: 120)
                - max_retries_per_timeout: Max retries per timeout (default: 2)

        Returns:
            pd.DataFrame: Aggregated DataFrame with columns matching output['schema']

        Examples:
            >>> # Summarize all texts into one summary
            >>> df.semantic.reduce(
            ...     prompt=\"\"\"Summarize the following items:
            ...     {% for item in inputs %}
            ...     - {{ item.text }}
            ...     {% endfor %}\"\"\",
            ...     output={"schema": {"summary": "str"}}
            ... )

            >>> # Reduce by group
            >>> df.semantic.reduce(
            ...     prompt=\"\"\"Combine items for {{ reduce_key }}:
            ...     {% for item in inputs %}
            ...     - {{ item.description }}
            ...     {% endfor %}\"\"\",
            ...     output={"schema": {"combined": "str"}},
            ...     reduce_keys=["category"]
            ... )
        """
        return self.agg(
            reduce_prompt=prompt,
            output=output,
            output_schema=output_schema,
            reduce_keys=reduce_keys,
            reduce_kwargs=kwargs,
        )

    def filter(
        self,
        prompt: str,
        *,
        output: dict[str, Any] | None = None,
        output_schema: dict[str, Any] | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Filter DataFrame rows based on semantic conditions.

        Documentation: https://ucbepic.github.io/docetl/operators/filter/

        Args:
            prompt: Jinja template string for generating prompts
            output: Output configuration with keys:
                - "schema": Dictionary defining the expected output structure and types
                  For filtering, this must be a single boolean field (e.g., {"keep": "bool"})
                - "mode": Optional output mode. Either "tools" (default) or "structured_output"
            output_schema: DEPRECATED. Use 'output' parameter instead. Backward-compatible schema dict.
            **kwargs: Additional configuration options:
                - model: LLM model to use
                - validate: List of validation expressions
                - num_retries_on_validate_failure: Number of retries
                - timeout: Timeout in seconds (default: 120)
                - max_retries_per_timeout: Max retries per timeout (default: 2)
                - skip_on_error: Skip rows on LLM error (default: False)
                - bypass_cache: Bypass cache for this operation (default: False)

        Returns:
            pd.DataFrame: Filtered DataFrame containing only rows where the model
                         returned True

        Examples:
            >>> # Simple filtering
            >>> df.semantic.filter(
            ...     prompt="Is this about technology? {{input.text}}"
            ... )

            >>> # Custom output schema
            >>> df.semantic.filter(
            ...     prompt="Analyze if this is relevant: {{input.text}}",
            ...     output={"schema": {"keep": "bool", "reason": "str"}}
            ... )
        """
        # Convert DataFrame to list of dicts
        input_data = self._df.to_dict("records")

        # Backward compatibility and defaults
        if output is None and output_schema is not None:
            output = {"schema": output_schema}
        if output is None and output_schema is None:
            output = {"schema": {"keep": "bool"}}

        # Validate output
        if not isinstance(output, dict):
            raise ValueError("output must be a dictionary")
        if "schema" not in output:
            raise ValueError("output must contain a 'schema' key")
        output_mode = output.get("mode", "tools")
        if output_mode not in ["tools", "structured_output"]:
            raise ValueError(
                f"output mode must be 'tools' or 'structured_output', got '{output_mode}'"
            )

        # Create map operation config for filtering
        filter_config = {
            "type": "map",
            "name": f"semantic_filter_{len(self._history)}",
            "prompt": prompt,
            "output": output,
            **kwargs,
        }

        # Create and execute filter operation
        filter_op = FilterOperation(
            runner=self.runner,
            config=filter_config,
            default_model=self.runner.config["default_model"],
            max_threads=self.runner.max_threads,
            console=self.runner.console,
            status=self.runner.status,
        )
        results, cost = filter_op.execute(input_data)

        return self._record_operation(results, "filter", filter_config, cost)

    def split(
        self, split_key: str, method: str, method_kwargs: dict[str, Any], **kwargs
    ) -> pd.DataFrame:
        """
        Split DataFrame rows into multiple chunks based on content.

        Documentation: https://ucbepic.github.io/docetl/operators/split/

        Args:
            split_key: The column containing content to split
            method: Splitting method, either "token_count" or "delimiter"
            method_kwargs: Dictionary containing method-specific parameters:
                - For "token_count": {"num_tokens": int, "model": str (optional)}
                - For "delimiter": {"delimiter": str, "num_splits_to_group": int (optional)}
            **kwargs: Additional configuration options:
                - model: LLM model to use for tokenization (default: from config)

        Returns:
            pd.DataFrame: DataFrame with split content, including:
                - {split_key}_chunk: The content of each chunk
                - {operation_name}_id: Unique identifier for the original document
                - {operation_name}_chunk_num: Sequential chunk number within the document

        Examples:
            >>> # Split by token count
            >>> df.semantic.split(
            ...     split_key="content",
            ...     method="token_count",
            ...     method_kwargs={"num_tokens": 100}
            ... )

            >>> # Split by delimiter
            >>> df.semantic.split(
            ...     split_key="text",
            ...     method="delimiter",
            ...     method_kwargs={"delimiter": "\n\n", "num_splits_to_group": 2}
            ... )
        """
        # Convert DataFrame to list of dicts
        input_data = self._df.to_dict("records")

        # Create split operation config
        split_config = {
            "type": "split",
            "name": f"semantic_split_{len(self._history)}",
            "split_key": split_key,
            "method": method,
            "method_kwargs": method_kwargs,
            **kwargs,
        }

        # Create and execute split operation
        split_op = SplitOperation(
            runner=self.runner,
            config=split_config,
            default_model=self.runner.config["default_model"],
            max_threads=self.runner.max_threads,
            console=self.runner.console,
            status=self.runner.status,
        )
        results, cost = split_op.execute(input_data)

        return self._record_operation(results, "split", split_config, cost)

    def gather(
        self,
        content_key: str,
        doc_id_key: str,
        order_key: str,
        peripheral_chunks: dict[str, Any] | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Gather contextual information from surrounding chunks to enhance each chunk.

        Documentation: https://ucbepic.github.io/docetl/operators/gather/

        Args:
            content_key: The column containing the main content to be enhanced
            doc_id_key: The column containing document identifiers to group chunks
            order_key: The column containing chunk order numbers within documents
            peripheral_chunks: Configuration for surrounding context:
                - previous: {"head": {"count": int}, "tail": {"count": int}, "middle": {}}
                - next: {"head": {"count": int}, "tail": {"count": int}, "middle": {}}
            **kwargs: Additional configuration options:
                - main_chunk_start: Start marker for main chunk (default: "--- Begin Main Chunk ---")
                - main_chunk_end: End marker for main chunk (default: "--- End Main Chunk ---")
                - doc_header_key: Column containing document headers (optional)

        Returns:
            pd.DataFrame: DataFrame with enhanced content including:
                - {content_key}_rendered: The main content with surrounding context

        Examples:
            >>> # Basic gathering with surrounding context
            >>> df.semantic.gather(
            ...     content_key="chunk_content",
            ...     doc_id_key="document_id",
            ...     order_key="chunk_number",
            ...     peripheral_chunks={
            ...         "previous": {"head": {"count": 2}, "tail": {"count": 1}},
            ...         "next": {"head": {"count": 1}, "tail": {"count": 2}}
            ...     }
            ... )

            >>> # Simple gathering without peripheral chunks
            >>> df.semantic.gather(
            ...     content_key="content",
            ...     doc_id_key="doc_id",
            ...     order_key="order"
            ... )
        """
        # Convert DataFrame to list of dicts
        input_data = self._df.to_dict("records")

        # Create gather operation config
        gather_config = {
            "type": "gather",
            "name": f"semantic_gather_{len(self._history)}",
            "content_key": content_key,
            "doc_id_key": doc_id_key,
            "order_key": order_key,
            **kwargs,
        }

        # Add peripheral_chunks config if provided
        if peripheral_chunks is not None:
            gather_config["peripheral_chunks"] = peripheral_chunks

        # Create and execute gather operation
        gather_op = GatherOperation(
            runner=self.runner,
            config=gather_config,
            default_model=self.runner.config["default_model"],
            max_threads=self.runner.max_threads,
            console=self.runner.console,
            status=self.runner.status,
        )
        results, cost = gather_op.execute(input_data)

        return self._record_operation(results, "gather", gather_config, cost)

    def unnest(
        self,
        unnest_key: str,
        keep_empty: bool = False,
        expand_fields: list[str] | None = None,
        recursive: bool = False,
        depth: int | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Unnest list-like or dictionary values into multiple rows.

        Documentation: https://ucbepic.github.io/docetl/operators/unnest/

        Args:
            unnest_key: The column containing list-like or dictionary values to unnest
            keep_empty: Whether to keep rows with empty/null values (default: False)
            expand_fields: For dictionary values, which fields to expand (default: all)
            recursive: Whether to recursively unnest nested structures (default: False)
            depth: Maximum depth for recursive unnesting (default: 1, or unlimited if recursive=True)
            **kwargs: Additional configuration options

        Returns:
            pd.DataFrame: DataFrame with unnested values, where:
                - For lists: Each list element becomes a separate row
                - For dicts: Specified fields are expanded into the parent row

        Examples:
            >>> # Unnest a list column
            >>> df.semantic.unnest(
            ...     unnest_key="tags"
            ... )
            # Input:  [{"id": 1, "tags": ["a", "b"]}]
            # Output: [{"id": 1, "tags": "a"}, {"id": 1, "tags": "b"}]

            >>> # Unnest a dictionary column with specific fields
            >>> df.semantic.unnest(
            ...     unnest_key="user_info",
            ...     expand_fields=["name", "age"]
            ... )
            # Input:  [{"id": 1, "user_info": {"name": "Alice", "age": 30, "email": "alice@example.com"}}]
            # Output: [{"id": 1, "user_info": {...}, "name": "Alice", "age": 30}]

            >>> # Recursive unnesting
            >>> df.semantic.unnest(
            ...     unnest_key="nested_lists",
            ...     recursive=True,
            ...     depth=2
            ... )
        """
        # Convert DataFrame to list of dicts
        input_data = self._df.to_dict("records")

        # Create unnest operation config
        unnest_config = {
            "type": "unnest",
            "name": f"semantic_unnest_{len(self._history)}",
            "unnest_key": unnest_key,
            "keep_empty": keep_empty,
            "recursive": recursive,
            **kwargs,
        }

        # Add optional parameters if provided
        if expand_fields is not None:
            unnest_config["expand_fields"] = expand_fields
        if depth is not None:
            unnest_config["depth"] = depth

        # Create and execute unnest operation
        unnest_op = UnnestOperation(
            runner=self.runner,
            config=unnest_config,
            default_model=self.runner.config["default_model"],
            max_threads=self.runner.max_threads,
            console=self.runner.console,
            status=self.runner.status,
        )
        results, cost = unnest_op.execute(input_data)

        return self._record_operation(results, "unnest", unnest_config, cost)

    @property
    def total_cost(self) -> float:
        """
        Return total cost of LLM operations in USD.

        Returns:
            float: Total cost of all operations performed on this DataFrame
        """
        return self._costs

    @property
    def history(self) -> list[OpHistory]:
        """
        Return the operation history.

        Returns:
            list[OpHistory]: List of operations performed on this DataFrame,
                            including their configurations and affected columns
        """
        return self._history.copy()
