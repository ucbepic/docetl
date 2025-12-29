"""
Fast should_optimize analyzer using a single LLM call.

This module provides a lightweight alternative to the full MapOptimizer/ReduceOptimizer/JoinOptimizer
flow for quickly determining if an operation should be decomposed.
"""

import json
import os
from typing import Any

import litellm
from litellm import completion, model_cost

from docetl.utils import completion_cost, count_tokens

# Drop unsupported params for models like gpt-5 that don't support temperature=0
litellm.drop_params = True


class FastShouldOptimizeAnalyzer:
    """
    Analyzes whether an operation should be optimized using a single LLM call.

    Instead of running the operation on sample data and using complex evaluation logic,
    this reads cached outputs from intermediate files and makes one judgment call.
    """

    def __init__(
        self,
        intermediate_dir: str,
        optimizer_model: str = "gpt-5.1",
        litellm_kwargs: dict[str, Any] | None = None,
    ):
        """
        Initialize the analyzer.

        Args:
            intermediate_dir: Path to the directory containing intermediate outputs
            optimizer_model: The LLM model to use for analysis (default: gpt-5.1)
            litellm_kwargs: Additional kwargs to pass to litellm.completion
        """
        self.intermediate_dir = intermediate_dir
        self.optimizer_model = optimizer_model
        self.litellm_kwargs = litellm_kwargs or {}
        if "temperature" not in self.litellm_kwargs:
            self.litellm_kwargs["temperature"] = 0.0

    def load_operation_data(self, step_name: str, op_name: str) -> list[dict[str, Any]]:
        """
        Load data from the intermediate file for an operation.

        Args:
            step_name: Name of the pipeline step
            op_name: Name of the operation

        Returns:
            List of dictionaries

        Raises:
            FileNotFoundError: If the intermediate file doesn't exist
        """
        output_path = os.path.join(self.intermediate_dir, step_name, f"{op_name}.json")
        if not os.path.exists(output_path):
            raise FileNotFoundError(
                f"No output file found at {output_path}. "
                "Run the operation first to generate outputs."
            )
        with open(output_path, "r") as f:
            return json.load(f)

    def find_previous_operation(
        self, operations: list[dict[str, Any]], op_name: str
    ) -> str | None:
        """
        Find the operation that comes before op_name in the pipeline.

        Args:
            operations: List of operation configs from the pipeline
            op_name: Name of the current operation

        Returns:
            Name of the previous operation, or None if this is the first operation
        """
        op_names = [op.get("name") for op in operations]
        try:
            idx = op_names.index(op_name)
            if idx > 0:
                return op_names[idx - 1]
        except ValueError:
            pass
        return None

    def get_max_context_tokens(self) -> int:
        """
        Get the maximum input tokens for the optimizer model.

        Returns:
            Maximum number of input tokens the model can handle
        """
        model_info = model_cost.get(self.optimizer_model, {})
        # Try without provider prefix if not found
        if not model_info:
            model_name = self.optimizer_model.split("/")[-1]
            model_info = model_cost.get(model_name, {})
        return model_info.get("max_input_tokens", 128000)  # Default to 128k

    def calculate_samples_that_fit(
        self,
        op_config: dict[str, Any],
        outputs: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Calculate how many output samples fit in the context window.

        Reserves space for system prompt, operation config, and response buffer,
        then fills remaining space with as many samples as possible.

        Args:
            op_config: The operation configuration dictionary
            outputs: List of all output documents

        Returns:
            List of samples that fit in the context window
        """
        max_tokens = self.get_max_context_tokens()

        # Reserve tokens for fixed parts
        system_prompt_tokens = 500
        op_config_tokens = count_tokens(
            json.dumps(op_config, default=str), self.optimizer_model
        )
        response_buffer = 2000

        available_for_samples = (
            max_tokens - system_prompt_tokens - op_config_tokens - response_buffer
        )

        # Collect samples that fit
        samples_to_include = []
        tokens_used = 0

        for output in outputs:
            sample_json = json.dumps(output, default=str)
            sample_tokens = count_tokens(sample_json, self.optimizer_model)
            if tokens_used + sample_tokens <= available_for_samples:
                samples_to_include.append(output)
                tokens_used += sample_tokens
            else:
                break

        return samples_to_include

    def build_analysis_prompt(
        self,
        op_config: dict[str, Any],
        samples: list[dict[str, Any]],
    ) -> tuple[str, str]:
        """
        Build the system prompt and user prompt for the analysis LLM call.

        Args:
            op_config: The operation configuration dictionary
            samples: List of output samples to analyze

        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        system_prompt = """You are an expert at analyzing LLM-powered data processing operations.
Your task is to determine if an operation would benefit from being decomposed into multiple
smaller, focused operations (also known as "optimization" or "decomposition").

An operation SHOULD be decomposed when:
1. The prompt asks the LLM to do multiple distinct tasks that could be done separately
2. The task is complex enough that breaking it into sequential steps would improve accuracy
3. The outputs show inconsistency, incompleteness, or quality issues that iterative refinement could fix
4. Long documents need to be processed in chunks rather than all at once
5. The prompt asks for both extraction AND analysis/synthesis in one step

An operation should NOT be decomposed when:
1. It performs a single, focused task well
2. The outputs are consistently high quality and complete
3. The task is simple and atomic (e.g., simple classification, single field extraction)
4. The operation is already well-scoped and produces reliable results

Be conservative - only recommend decomposition if there's clear evidence it would help."""

        output_schema = op_config.get("output", {}).get("schema", {})
        prompt_template = op_config.get("prompt", "No prompt specified")

        # Truncate very long prompts for display
        if len(prompt_template) > 3000:
            prompt_template = prompt_template[:3000] + "\n... [truncated]"

        user_prompt = f"""Analyze this data processing operation and its outputs:

## Operation Configuration

**Name:** {op_config.get('name', 'unknown')}
**Type:** {op_config.get('type', 'unknown')}

**Prompt Template:**
```
{prompt_template}
```

**Output Schema:**
```json
{json.dumps(output_schema, indent=2)}
```

## Sample Outputs ({len(samples)} samples from the operation)

```json
{json.dumps(samples, indent=2, default=str)}
```

## Your Task

Based on the operation configuration and sample outputs, determine:
1. Should this operation be decomposed/optimized?
2. If yes, what specific improvements would help?

Consider the quality, completeness, and consistency of the outputs when making your assessment."""

        return system_prompt, user_prompt

    def analyze(
        self,
        op_config: dict[str, Any],
        step_name: str,
        op_name: str,
    ) -> tuple[str, list[dict[str, Any]], int, float]:
        """
        Analyze whether an operation should be optimized.

        This is the main entry point. It loads outputs, builds the prompt,
        makes a single LLM call, and returns the assessment.

        Args:
            op_config: The operation configuration dictionary
            step_name: Name of the pipeline step
            op_name: Name of the operation

        Returns:
            Tuple of (rationale, output_samples, num_docs_analyzed, cost):
            - rationale: Empty string if no optimization needed, explanation if it should be optimized
            - output_samples: The samples that were analyzed
            - num_docs_analyzed: Number of documents that fit in the LLM prompt
            - cost: LLM API cost in USD

        Raises:
            ValueError: If the operation is not an LLM-powered map operation
        """
        # Validate operation type - only LLM-powered map operations
        op_type = op_config.get("type", "")
        if op_type != "map":
            raise ValueError(
                f"should_optimize only supports 'map' operations, got '{op_type}'. "
                "Only LLM-powered map operations can be analyzed for decomposition."
            )

        # Load outputs from intermediate file
        outputs = self.load_operation_data(step_name, op_name)

        if not outputs:
            return "No output samples available for analysis.", [], 0, 0.0

        # Calculate samples that fit in context
        samples = self.calculate_samples_that_fit(op_config, outputs)

        if not samples:
            return "Could not fit any samples in context window.", outputs[:5], 0, 0.0

        # Build prompt
        system_prompt, user_prompt = self.build_analysis_prompt(op_config, samples)

        # Make LLM call with structured output
        response = completion(
            model=self.optimizer_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "optimization_analysis",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "should_optimize": {
                                "type": "boolean",
                                "description": "True if operation should be decomposed/optimized",
                            },
                            "rationale": {
                                "type": "string",
                                "description": "Explanation of why the operation should or should not be optimized",
                            },
                            "suggested_improvements": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Specific improvements if optimization is recommended (empty if not)",
                            },
                        },
                        "required": [
                            "should_optimize",
                            "rationale",
                            "suggested_improvements",
                        ],
                        "additionalProperties": False,
                    },
                },
            },
            **self.litellm_kwargs,
        )

        # Calculate cost
        cost = completion_cost(response)

        # Parse response
        result = json.loads(response.choices[0].message.content)

        num_docs_analyzed = len(samples)

        if result["should_optimize"]:
            # Build rationale string with improvements
            rationale_parts = [result["rationale"]]
            if result["suggested_improvements"]:
                rationale_parts.append("\n\nSuggested improvements:")
                for imp in result["suggested_improvements"]:
                    rationale_parts.append(f"- {imp}")
            return "\n".join(rationale_parts), samples, num_docs_analyzed, cost
        else:
            # Return empty string to indicate no optimization needed
            return "", samples, num_docs_analyzed, cost
