"""
Fast decomposition analyzer for map operations.

This module provides a fast way to decompose map operations using directives,
running candidates on samples, and selecting the best via pairwise LLM comparison.
"""

import json
import os
import tempfile
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from typing import Any

import litellm
from litellm import completion, model_cost
from rich.console import Console

from docetl.console import get_console
from docetl.reasoning_optimizer.directives import (
    ChainingDirective,
    ClarifyInstructionsDirective,
    DeterministicDocCompressionDirective,
    DocumentChunkingDirective,
    GleaningDirective,
    IsolatingSubtasksDirective,
)
from docetl.runner import DSLRunner
from docetl.utils import completion_cost, count_tokens

# Drop unsupported params for models like gpt-5 that don't support temperature=0
litellm.drop_params = True

# Base directives always applicable to map operations
BASE_MAP_DIRECTIVES = [
    ChainingDirective(),
    IsolatingSubtasksDirective(),
    GleaningDirective(),
    ClarifyInstructionsDirective(),
]

# Directives that depend on document size
CHUNKING_DIRECTIVE = DocumentChunkingDirective()
COMPRESSION_DIRECTIVE = DeterministicDocCompressionDirective()

# Threshold for enabling DeterministicDocCompression (in characters)
DOC_COMPRESSION_CHAR_THRESHOLD = 1000

# Threshold for enabling DocumentChunking (as fraction of context window)
DOC_CHUNKING_CONTEXT_THRESHOLD = 0.10  # 10%


class FastDecomposer:
    """
    Fast decomposition of map operations using directives and pairwise comparison.

    Instead of the full optimizer flow, this:
    1. Tries multiple directives to generate candidate decompositions
    2. Runs each candidate on sample documents
    3. Uses LLM judge for pairwise comparison
    4. Returns the winning decomposition
    """

    def __init__(
        self,
        yaml_config_path: str,
        optimizer_model: str = "gpt-5.1",
        sample_size: int = 5,
        litellm_kwargs: dict[str, Any] | None = None,
        console: Console | None = None,
    ):
        """
        Initialize the decomposer.

        Args:
            yaml_config_path: Path to the pipeline YAML config file
            optimizer_model: LLM model to use for directive instantiation and judging
            sample_size: Number of sample documents to run candidates on
            litellm_kwargs: Additional kwargs to pass to litellm.completion
            console: Rich console for output (uses default if not provided)
        """
        self.yaml_config_path = yaml_config_path
        self.optimizer_model = optimizer_model
        self.sample_size = sample_size
        self.litellm_kwargs = litellm_kwargs or {}
        if "temperature" not in self.litellm_kwargs:
            self.litellm_kwargs["temperature"] = 0.0

        self.total_cost = 0.0
        self.console = console or get_console()

        # Load the config
        import yaml

        with open(yaml_config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.operators = self.config.get("operations", [])
        self.default_model = self.config.get("default_model", "gpt-4o-mini")
        self.intermediate_dir = (
            self.config.get("pipeline", {}).get("output", {}).get("intermediate_dir")
        )

    def _log(self, message: str) -> None:
        """Log a message to the console."""
        self.console.log(message)

    def get_model_context_limit(self, model: str) -> int:
        """
        Get the context window limit for a model.

        Args:
            model: The model name (e.g., 'gpt-4o', 'azure/gpt-4')

        Returns:
            Maximum number of input tokens the model can handle
        """
        model_info = model_cost.get(model, {})
        # Try without provider prefix if not found
        if not model_info:
            model_name = model.split("/")[-1]
            model_info = model_cost.get(model_name, {})
        return model_info.get("max_input_tokens", 128000)  # Default to 128k

    def get_avg_doc_size(
        self, sample_data: list[dict[str, Any]], op_config: dict[str, Any]
    ) -> tuple[float, float]:
        """
        Calculate the average document size in characters and tokens.

        Extracts the document content from sample data based on the operation's
        prompt template (looks for {{ input.field_name }} patterns).

        Args:
            sample_data: List of sample documents
            op_config: The operation configuration

        Returns:
            Tuple of (avg_chars, avg_tokens) for the document content
        """
        import re

        if not sample_data:
            return 0.0, 0.0

        prompt = op_config.get("prompt", "")
        model = op_config.get("model", self.default_model)

        # Extract field names from prompt template ({{ input.field_name }})
        field_pattern = r"\{\{\s*input\.(\w+)\s*\}\}"
        fields = re.findall(field_pattern, prompt)

        if not fields:
            # Fallback: use all string values from the first document
            if sample_data:
                fields = [
                    k
                    for k, v in sample_data[0].items()
                    if isinstance(v, str) and len(v) > 100
                ]

        total_chars = 0
        total_tokens = 0

        for doc in sample_data:
            doc_content = ""
            for field in fields:
                if field in doc:
                    value = doc[field]
                    if isinstance(value, str):
                        doc_content += value
                    else:
                        doc_content += str(value)

            total_chars += len(doc_content)
            if doc_content:
                total_tokens += count_tokens(doc_content, model)

        n = len(sample_data)
        return total_chars / n if n > 0 else 0.0, total_tokens / n if n > 0 else 0.0

    def get_applicable_directives(
        self,
        sample_data: list[dict[str, Any]],
        op_config: dict[str, Any],
    ) -> list:
        """
        Get the list of directives applicable based on data characteristics.

        - DocumentChunkingDirective: only if avg doc size > 10% of context window
        - DeterministicDocCompressionDirective: only if avg doc size > 1000 chars

        Args:
            sample_data: List of sample documents
            op_config: The operation configuration

        Returns:
            List of applicable directive instances
        """
        directives = []  # Build list with priority ordering

        model = op_config.get("model", self.default_model)
        context_limit = self.get_model_context_limit(model)
        avg_chars, avg_tokens = self.get_avg_doc_size(sample_data, op_config)

        self._log(
            f"Document analysis: avg_chars={avg_chars:.0f}, avg_tokens={avg_tokens:.0f}, "
            f"context_limit={context_limit}"
        )

        # Add DeterministicDocCompression FIRST if doc size > 1000 chars (high priority)
        if avg_chars > DOC_COMPRESSION_CHAR_THRESHOLD:
            self._log(
                f"[cyan]Enabling DeterministicDocCompression (priority)[/cyan] "
                f"(avg {avg_chars:.0f} chars > {DOC_COMPRESSION_CHAR_THRESHOLD})"
            )
            directives.append(COMPRESSION_DIRECTIVE)

        # Add base directives
        directives.extend(BASE_MAP_DIRECTIVES)

        # Add DocumentChunking if avg tokens > 10% of context window
        token_threshold = context_limit * DOC_CHUNKING_CONTEXT_THRESHOLD
        if avg_tokens > token_threshold:
            self._log(
                f"[cyan]Enabling DocumentChunking[/cyan] "
                f"(avg {avg_tokens:.0f} tokens > {token_threshold:.0f} = 10% of {context_limit})"
            )
            directives.append(CHUNKING_DIRECTIVE)
        else:
            self._log(
                f"[dim]Skipping DocumentChunking[/dim] "
                f"(avg {avg_tokens:.0f} tokens <= {token_threshold:.0f} = 10% of {context_limit})"
            )

        return directives

    def load_sample_data(self, step_name: str, op_name: str) -> list[dict[str, Any]]:
        """
        Load sample input data for an operation.

        For the first operation, loads from the dataset.
        For subsequent operations, loads from the previous operation's intermediate output.

        Args:
            step_name: Name of the pipeline step
            op_name: Name of the operation

        Returns:
            List of sample documents
        """
        # Find the operation's position
        op_names = [op.get("name") for op in self.operators]
        try:
            op_idx = op_names.index(op_name)
        except ValueError:
            raise ValueError(f"Operation '{op_name}' not found in config")

        if op_idx == 0:
            # First operation - load from dataset
            datasets = self.config.get("datasets", {})
            # Get the first dataset (or the one used by this step)
            for dataset_name, dataset_config in datasets.items():
                dataset_path = dataset_config.get("path")
                if dataset_path and os.path.exists(dataset_path):
                    with open(dataset_path, "r") as f:
                        data = json.load(f)
                    return data[: self.sample_size]
            raise FileNotFoundError("No dataset found in config")
        else:
            # Load from previous operation's intermediate output
            prev_op_name = op_names[op_idx - 1]
            output_path = os.path.join(
                self.intermediate_dir, step_name, f"{prev_op_name}.json"
            )
            if not os.path.exists(output_path):
                raise FileNotFoundError(
                    f"No intermediate output found at {output_path}. "
                    "Run the previous operation first."
                )
            with open(output_path, "r") as f:
                data = json.load(f)
            return data[: self.sample_size]

    def get_input_file_path(self) -> str:
        """Get the input file path for the pipeline."""
        datasets = self.config.get("datasets", {})
        for dataset_name, dataset_config in datasets.items():
            path = dataset_config.get("path")
            if path:
                return path
        return ""

    def generate_candidates(
        self,
        op_name: str,
        sample_data: list[dict[str, Any]],
        target_op: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """
        Generate candidate decompositions using directives.

        Directives are selected based on data characteristics:
        - DocumentChunkingDirective: only if avg doc size > 10% of context window
        - DeterministicDocCompressionDirective: only if avg doc size > 1000 chars

        Args:
            op_name: Name of the operation to decompose
            sample_data: Sample data for analyzing document characteristics
            target_op: The target operation configuration

        Returns:
            List of candidate dictionaries with 'name', 'ops', 'cost' keys
        """
        candidates = []

        # Add original as baseline
        self._log("Adding original operation as baseline candidate")
        candidates.append(
            {
                "name": "original",
                "ops": deepcopy(self.operators),
                "cost": 0.0,
                "error": None,
            }
        )

        input_file_path = self.get_input_file_path()

        # Get applicable directives based on data characteristics
        applicable_directives = self.get_applicable_directives(sample_data, target_op)

        self._log(
            f"Generating candidates using {len(applicable_directives)} directives..."
        )
        for i, directive in enumerate(applicable_directives, 1):
            with self.console.status(
                f"[bold cyan]({i}/{len(applicable_directives)}) Trying directive: {directive.name}...[/bold cyan]",
                spinner="dots",
            ):
                try:
                    new_ops_list, _, cost = directive.instantiate(
                        operators=deepcopy(self.operators),
                        target_ops=[op_name],
                        agent_llm=self.optimizer_model,
                        message_history=[],
                        global_default_model=self.default_model,
                        input_file_path=input_file_path,
                    )
                    self.total_cost += cost
                    self._log(
                        f"  [green]✓[/green] {directive.name} generated {len(new_ops_list)} operations (cost: ${cost:.4f})"
                    )
                    candidates.append(
                        {
                            "name": directive.name,
                            "ops": new_ops_list,
                            "cost": cost,
                            "error": None,
                        }
                    )
                except Exception as e:
                    # Directive not applicable or failed - skip it
                    self._log(f"  [red]✗[/red] {directive.name} failed: {str(e)}")
                    candidates.append(
                        {
                            "name": directive.name,
                            "ops": None,
                            "cost": 0.0,
                            "error": str(e),
                        }
                    )

        return candidates

    def extract_ops_to_run(
        self, ops_list: list[dict], original_op_name: str
    ) -> list[dict]:
        """
        Extract the operations that replaced the original operation.

        Args:
            ops_list: The full transformed operations list
            original_op_name: Name of the original operation that was decomposed

        Returns:
            List of operations that should be run on samples
        """
        # Find ops that are new (not in original) or modified
        original_names = {op["name"] for op in self.operators}

        # Find the position where the original op was
        original_idx = None
        for i, op in enumerate(self.operators):
            if op["name"] == original_op_name:
                original_idx = i
                break

        if original_idx is None:
            return ops_list

        # Find new ops (those not in original list)
        new_ops = []
        for op in ops_list:
            if op["name"] not in original_names or op["name"] == original_op_name:
                new_ops.append(op)

        return new_ops if new_ops else [ops_list[original_idx]]

    def run_candidate_on_samples(
        self,
        candidate: dict[str, Any],
        sample_data: list[dict[str, Any]],
        original_op_name: str,
    ) -> list[dict[str, Any]]:
        """
        Run a candidate's operations on sample data.

        Args:
            candidate: Candidate dictionary with 'ops' key
            sample_data: List of sample documents
            original_op_name: Name of the original operation

        Returns:
            List of output documents
        """
        if candidate["ops"] is None:
            return []

        # Extract ops to run
        ops_to_run = self.extract_ops_to_run(candidate["ops"], original_op_name)

        if not ops_to_run:
            return []

        # Create a minimal config for running these ops
        # Write sample data to a temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(sample_data, f)
            temp_input_path = f.name

        # Create a temp output file (required by DSLRunner validation)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_output_path = f.name

        try:
            # Create a minimal pipeline config
            temp_config = {
                "default_model": self.default_model,
                "operations": ops_to_run,
                "datasets": {
                    "sample_data": {
                        "type": "file",
                        "path": temp_input_path,
                    }
                },
                "pipeline": {
                    "steps": [
                        {
                            "name": "decompose_test",
                            "input": "sample_data",
                            "operations": [op["name"] for op in ops_to_run],
                        }
                    ],
                    "output": {
                        "type": "file",
                        "path": temp_output_path,
                    },
                },
            }

            # Create runner and execute
            runner = DSLRunner(temp_config, max_threads=4)

            # Run operations sequentially on the data
            current_data = sample_data
            for op_config in ops_to_run:
                current_data = runner._run_operation(op_config, current_data)

            self.total_cost += runner.total_cost

            return current_data

        finally:
            # Clean up temp files
            for temp_path in [temp_input_path, temp_output_path]:
                try:
                    os.unlink(temp_path)
                except Exception:
                    pass

    def pairwise_compare(
        self,
        candidate_a: dict[str, Any],
        candidate_b: dict[str, Any],
        original_prompt: str,
        output_schema: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Compare two candidates using LLM judge.

        Args:
            candidate_a: First candidate with 'name', 'outputs' keys
            candidate_b: Second candidate with 'name', 'outputs' keys
            original_prompt: The original operation's prompt
            output_schema: The expected output schema

        Returns:
            The winning candidate
        """
        # If one has no outputs, the other wins
        if not candidate_a.get("outputs"):
            return candidate_b
        if not candidate_b.get("outputs"):
            return candidate_a

        system_prompt = """You are an expert judge comparing outputs from two data processing pipeline variants.

Your task is to determine which variant produces BETTER outputs based on:
1. **Completeness**: Does the output contain all required information?
2. **Accuracy**: Is the extracted/generated information correct?
3. **Consistency**: Are the outputs consistent across different samples?
4. **Quality**: Is the output well-structured and useful?

Be objective and focus on the actual output quality, not the approach used."""

        user_prompt = f"""Compare outputs from two pipeline variants for this task:

## Original Task
**Prompt:**
```
{original_prompt[:2000]}{"..." if len(original_prompt) > 2000 else ""}
```

**Expected Output Schema:**
```json
{json.dumps(output_schema, indent=2)}
```

## Variant A: {candidate_a["name"]}
**Sample Outputs:**
```json
{json.dumps(candidate_a["outputs"][:3], indent=2, default=str)}
```

## Variant B: {candidate_b["name"]}
**Sample Outputs:**
```json
{json.dumps(candidate_b["outputs"][:3], indent=2, default=str)}
```

## Your Task
Which variant produces better outputs? Consider completeness, accuracy, consistency, and quality.
Respond with your analysis and final choice."""

        response = completion(
            model=self.optimizer_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "comparison_result",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "winner": {
                                "type": "string",
                                "enum": ["A", "B"],
                                "description": "Which variant is better: A or B",
                            },
                            "rationale": {
                                "type": "string",
                                "description": "Explanation of why this variant is better",
                            },
                            "a_strengths": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Strengths of variant A",
                            },
                            "b_strengths": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Strengths of variant B",
                            },
                        },
                        "required": [
                            "winner",
                            "rationale",
                            "a_strengths",
                            "b_strengths",
                        ],
                        "additionalProperties": False,
                    },
                },
            },
            **self.litellm_kwargs,
        )

        cost = completion_cost(response)
        self.total_cost += cost

        result = json.loads(response.choices[0].message.content)

        winner = candidate_a if result["winner"] == "A" else candidate_b
        winner["comparison_rationale"] = result["rationale"]

        return winner

    def decompose(
        self,
        step_name: str,
        op_name: str,
    ) -> tuple[list[dict[str, Any]], str, int, float]:
        """
        Decompose an operation using fast directive-based approach.

        This is the main entry point. It:
        1. Generates candidate decompositions using directives
        2. Runs each on sample data
        3. Uses pairwise LLM comparison to select the best
        4. Returns the winning decomposition

        Args:
            step_name: Name of the pipeline step
            op_name: Name of the operation to decompose

        Returns:
            Dict with:
            - decomposed_ops: List of operations that replace the original
            - winning_directive: Name of the winning directive
            - candidates_evaluated: Number of candidates that were compared
            - original_outputs: Sample outputs from the original operation
            - decomposed_outputs: Sample outputs from the winning decomposition
            - comparison_rationale: LLM's explanation of why the winner was chosen
            - cost: Total LLM API cost
        """
        self.console.rule("[bold blue]Fast Decomposition[/bold blue]")
        self._log(f"[bold]Target operation:[/bold] {op_name}")

        # Find the target operation config
        target_op = None
        for op in self.operators:
            if op["name"] == op_name:
                target_op = op
                break

        if target_op is None:
            raise ValueError(f"Operation '{op_name}' not found in config")

        # Verify it's a map operation
        if target_op.get("type") != "map":
            raise ValueError(
                f"Operation '{op_name}' is type '{target_op.get('type')}', "
                "but fast decomposition only supports 'map' operations"
            )

        # Load sample data
        self._log(f"Loading sample data from step '{step_name}'...")
        sample_data = self.load_sample_data(step_name, op_name)
        self._log(f"[green]✓[/green] Loaded {len(sample_data)} sample documents")

        # Generate candidates
        self.console.rule("[bold]Generating Candidates[/bold]")
        candidates = self.generate_candidates(op_name, sample_data, target_op)

        # Filter out failed candidates
        valid_candidates = [c for c in candidates if c["ops"] is not None]
        self._log(f"[bold]{len(valid_candidates)}[/bold] valid candidates generated")

        if len(valid_candidates) < 2:
            # Only original (or nothing) - return original
            self._log(
                "[yellow]No alternative decompositions generated. Keeping original.[/yellow]"
            )
            return {
                "decomposed_ops": self.operators,
                "winning_directive": "original",
                "candidates_evaluated": len(valid_candidates),
                "original_outputs": [],
                "decomposed_outputs": [],
                "comparison_rationale": "No alternative decompositions were generated.",
                "cost": self.total_cost,
            }

        # Run each candidate on samples IN PARALLEL
        self.console.rule("[bold]Running Candidates on Samples[/bold]")
        self._log(f"Running {len(valid_candidates)} candidates in parallel...")

        def run_single_candidate(candidate):
            """Run a single candidate and return results."""
            name = candidate["name"]
            try:
                outputs = self.run_candidate_on_samples(candidate, sample_data, op_name)
                return {"name": name, "outputs": outputs, "error": None}
            except Exception as e:
                error_tb = traceback.format_exc()
                return {
                    "name": name,
                    "outputs": [],
                    "error": str(e),
                    "traceback": error_tb,
                }

        # Run all candidates in parallel
        with self.console.status(
            "[bold cyan]Running candidates on samples...[/bold cyan]", spinner="dots"
        ):
            with ThreadPoolExecutor(max_workers=len(valid_candidates)) as executor:
                future_to_candidate = {
                    executor.submit(run_single_candidate, c): c
                    for c in valid_candidates
                }

                for future in as_completed(future_to_candidate):
                    candidate = future_to_candidate[future]
                    result = future.result()

                    if result["error"]:
                        candidate["outputs"] = []
                        candidate["run_error"] = result["error"]
                        self._log(
                            f"  [red]✗[/red] {result['name']} failed: {result['error']}"
                        )
                    else:
                        candidate["outputs"] = result["outputs"]
                        self._log(
                            f"  [green]✓[/green] {result['name']}: {len(result['outputs'])} outputs"
                        )

        # Filter to candidates with outputs
        candidates_with_outputs = [c for c in valid_candidates if c.get("outputs")]
        self._log(
            f"[bold]{len(candidates_with_outputs)}[/bold] candidates produced outputs"
        )

        if not candidates_with_outputs:
            # All failed - return original
            self._log("[red]All decomposition candidates failed to execute.[/red]")
            # Log the errors for debugging
            for c in valid_candidates:
                if c.get("run_error"):
                    self._log(f"  [red]{c['name']}:[/red] {c['run_error']}")
            return {
                "decomposed_ops": self.operators,
                "winning_directive": "original",
                "candidates_evaluated": 0,
                "original_outputs": [],
                "decomposed_outputs": [],
                "comparison_rationale": "All decomposition candidates failed to execute.",
                "cost": self.total_cost,
            }

        # Find the original candidate's outputs
        original_candidate = next(
            (c for c in candidates_with_outputs if c["name"] == "original"), None
        )
        original_outputs = original_candidate["outputs"] if original_candidate else []

        # Parallel pairwise comparison: compare all candidates against original
        self.console.rule("[bold]Pairwise Comparison[/bold]")
        original_prompt = target_op.get("prompt", "")
        output_schema = target_op.get("output", {}).get("schema", {})

        # If only original exists, it wins by default
        if len(candidates_with_outputs) == 1:
            winner = candidates_with_outputs[0]
        elif original_candidate is None:
            # No original - just pick the first candidate
            winner = candidates_with_outputs[0]
        else:
            # Compare all non-original candidates against original IN PARALLEL
            challengers = [
                c for c in candidates_with_outputs if c["name"] != "original"
            ]

            if not challengers:
                winner = original_candidate
            else:
                self._log(
                    f"Comparing {len(challengers)} candidates against original in parallel..."
                )

                def compare_against_original(challenger):
                    """Compare a single challenger against original."""
                    try:
                        result = self.pairwise_compare(
                            original_candidate,
                            challenger,
                            original_prompt,
                            output_schema,
                        )
                        won = result["name"] == challenger["name"]
                        return {
                            "challenger": challenger,
                            "won": won,
                            "rationale": result.get("comparison_rationale", ""),
                        }
                    except Exception as e:
                        return {
                            "challenger": challenger,
                            "won": False,
                            "error": str(e),
                        }

                # Run all comparisons in parallel
                comparison_results = []
                with self.console.status(
                    f"[bold cyan]Running {len(challengers)} comparisons in parallel...[/bold cyan]",
                    spinner="dots",
                ):
                    with ThreadPoolExecutor(max_workers=len(challengers)) as executor:
                        future_to_challenger = {
                            executor.submit(compare_against_original, c): c
                            for c in challengers
                        }

                        for future in as_completed(future_to_challenger):
                            result = future.result()
                            comparison_results.append(result)
                            challenger_name = result["challenger"]["name"]
                            if result.get("error"):
                                self._log(
                                    f"  [red]✗[/red] {challenger_name} vs original: error - {result['error']}"
                                )
                            elif result["won"]:
                                self._log(
                                    f"  [green]✓[/green] {challenger_name} beat original"
                                )
                            else:
                                self._log(
                                    f"  [dim]○[/dim] {challenger_name} lost to original"
                                )

                # Find winners (candidates that beat original)
                winners = [r for r in comparison_results if r.get("won")]

                if not winners:
                    # Original beats all challengers
                    winner = original_candidate
                    self._log("[bold]Original wins against all challengers[/bold]")
                elif len(winners) == 1:
                    # Single winner
                    winner = winners[0]["challenger"]
                    winner["comparison_rationale"] = winners[0].get("rationale", "")
                else:
                    # Multiple winners beat original - run tiebreaker comparisons in parallel
                    self._log(
                        f"[bold]{len(winners)} candidates beat original - running tiebreaker...[/bold]"
                    )

                    # Compare all winners against each other in parallel (round-robin)
                    winner_candidates = [w["challenger"] for w in winners]
                    win_counts = {c["name"]: 0 for c in winner_candidates}

                    # Generate all pairwise matchups
                    matchups = []
                    for i, a in enumerate(winner_candidates):
                        for b in winner_candidates[i + 1 :]:
                            matchups.append((a, b))

                    if matchups:

                        def run_matchup(matchup):
                            a, b = matchup
                            try:
                                result = self.pairwise_compare(
                                    a, b, original_prompt, output_schema
                                )
                                return {
                                    "winner": result["name"],
                                    "a": a["name"],
                                    "b": b["name"],
                                }
                            except Exception:
                                return {
                                    "winner": a["name"],
                                    "a": a["name"],
                                    "b": b["name"],
                                }  # Default to first

                        with self.console.status(
                            f"[bold cyan]Running {len(matchups)} tiebreaker comparisons...[/bold cyan]",
                            spinner="dots",
                        ):
                            with ThreadPoolExecutor(
                                max_workers=len(matchups)
                            ) as executor:
                                for result in executor.map(run_matchup, matchups):
                                    win_counts[result["winner"]] += 1
                                    self._log(
                                        f"  [dim]{result['a']} vs {result['b']} → {result['winner']}[/dim]"
                                    )

                    # Pick candidate with most wins
                    best_name = max(win_counts, key=win_counts.get)
                    winner = next(
                        c for c in winner_candidates if c["name"] == best_name
                    )
                    self._log(
                        f"[bold]Tiebreaker winner: {best_name} ({win_counts[best_name]} wins)[/bold]"
                    )

        # Extract the decomposed operations
        decomposed_ops = self.extract_ops_to_run(winner["ops"], op_name)

        # Final summary
        self.console.rule("[bold green]Decomposition Complete[/bold green]")
        self._log(f"[bold]Winner:[/bold] [green]{winner['name']}[/green]")
        self._log(f"[bold]Candidates evaluated:[/bold] {len(candidates_with_outputs)}")
        self._log(f"[bold]New operations:[/bold] {len(decomposed_ops)}")
        self._log(f"[bold]Total cost:[/bold] ${self.total_cost:.4f}")

        return {
            "decomposed_ops": decomposed_ops,
            "winning_directive": winner["name"],
            "candidates_evaluated": len(candidates_with_outputs),
            "original_outputs": original_outputs,
            "decomposed_outputs": winner.get("outputs", []),
            "comparison_rationale": winner.get("comparison_rationale", ""),
            "cost": self.total_cost,
        }
