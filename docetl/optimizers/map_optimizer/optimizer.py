import concurrent
import copy
import random
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

from jinja2 import Template
from litellm import model_cost
from rich.table import Table

from docetl.optimizers.map_optimizer.evaluator import Evaluator
from docetl.optimizers.map_optimizer.plan_generators import PlanGenerator
from docetl.optimizers.map_optimizer.prompt_generators import PromptGenerator
from docetl.optimizers.map_optimizer.utils import select_evaluation_samples
from docetl.utils import StageType, count_tokens


class MapOptimizer:
    """
    A class for optimizing map operations in data processing pipelines.

    This optimizer analyzes the input operation configuration and data,
    and generates optimized plans for executing the operation. It can
    create plans for chunking, metadata extraction, gleaning, chain
    decomposition, and parallel execution.

    Attributes:
        config (dict[str, Any]): The configuration dictionary for the optimizer.
        console (Console): A Rich console object for pretty printing.
        llm_client (LLMClient): A client for interacting with a language model.
        _run_operation (Callable): A function to execute operations.
        max_threads (int): The maximum number of threads to use for parallel execution.
        timeout (int): The timeout in seconds for operation execution.

    """

    def __init__(
        self,
        runner,
        run_operation: Callable,
        timeout: int = 10,
        is_filter: bool = False,
        depth: int = 1,
    ):
        """
        Initialize the MapOptimizer.

        Args:
            runner (Runner): The runner object.
            run_operation (Callable): A function to execute operations.
            timeout (int, optional): The timeout in seconds for operation execution. Defaults to 10.
            is_filter (bool, optional): If True, the operation is a filter operation. Defaults to False.
        """
        self.runner = runner
        self.config = runner.config
        self.console = runner.console
        self.llm_client = runner.optimizer.llm_client
        self._run_operation = run_operation
        self.max_threads = runner.max_threads
        self.timeout = runner.optimizer.timeout
        self._num_plans_to_evaluate_in_parallel = 5
        self.is_filter = is_filter
        self.k_to_pairwise_compare = 6

        self.plan_generator = PlanGenerator(
            runner,
            self.llm_client,
            self.console,
            self.config,
            run_operation,
            self.max_threads,
            is_filter,
            depth,
        )
        self.evaluator = Evaluator(
            self.llm_client,
            self.console,
            self._run_operation,
            self.timeout,
            self._num_plans_to_evaluate_in_parallel,
            self.is_filter,
        )
        self.prompt_generator = PromptGenerator(
            self.runner,
            self.llm_client,
            self.console,
            self.config,
            self.max_threads,
            self.is_filter,
        )

    def should_optimize(
        self, op_config: dict[str, Any], input_data: list[dict[str, Any]]
    ) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
        """
        Determine if the given operation configuration should be optimized.
        """
        (
            input_data,
            output_data,
            _,
            _,
            validator_prompt,
            assessment,
            data_exceeds_limit,
        ) = self._should_optimize_helper(op_config, input_data)
        if data_exceeds_limit or assessment.get("needs_improvement", True):
            assessment_str = (
                "\n".join(assessment.get("reasons", []))
                + "\n\nHere are some improvements that may help:\n"
                + "\n".join(assessment.get("improvements", []))
            )
            if data_exceeds_limit:
                assessment_str += "\nAlso, the input data exceeds the token limit."
            return assessment_str, input_data, output_data
        else:
            return "", input_data, output_data

    def _should_optimize_helper(
        self, op_config: dict[str, Any], input_data: list[dict[str, Any]]
    ) -> tuple[
        list[dict[str, Any]],
        list[dict[str, Any]],
        int,
        float,
        str,
        dict[str, Any],
        bool,
    ]:
        """
        Determine if the given operation configuration should be optimized.
        Create a custom validator prompt and assess the operation's performance
        using the validator.
        """
        self.console.post_optimizer_status(StageType.SAMPLE_RUN)
        input_data = copy.deepcopy(input_data)
        # Add id to each input_data
        for i in range(len(input_data)):
            input_data[i]["_map_opt_id"] = str(uuid.uuid4())

        # Define the token limit (adjust as needed)
        model_input_context_length = model_cost.get(
            op_config.get("model", self.config.get("default_model")), {}
        ).get("max_input_tokens", 8192)

        # Render the prompt with all sample inputs and count tokens
        total_tokens = 0
        exceed_count = 0
        for sample in input_data:
            rendered_prompt = Template(op_config["prompt"]).render(input=sample)
            prompt_tokens = count_tokens(
                rendered_prompt,
                op_config.get("model", self.config.get("default_model")),
            )
            total_tokens += prompt_tokens

            if prompt_tokens > model_input_context_length:
                exceed_count += 1

        # Calculate average tokens and percentage of samples exceeding limit
        avg_tokens = total_tokens / len(input_data)
        exceed_percentage = (exceed_count / len(input_data)) * 100

        data_exceeds_limit = exceed_count > 0
        if exceed_count > 0:
            self.console.log(
                f"[yellow]Warning: {exceed_percentage:.2f}% of prompts exceed token limit. "
                f"Average token count: {avg_tokens:.2f}. "
                f"Truncating input data when generating validators.[/yellow]"
            )

        # Execute the original operation on the sample data
        no_change_start = time.time()
        output_data = self._run_operation(op_config, input_data, is_build=True)
        no_change_runtime = time.time() - no_change_start

        # Capture output for the sample run
        self.runner.optimizer.captured_output.save_optimizer_output(
            stage_type=StageType.SAMPLE_RUN,
            output={
                "operation_config": op_config,
                "input_data": input_data,
                "output_data": output_data,
            },
        )

        # Generate custom validator prompt
        self.console.post_optimizer_status(StageType.SHOULD_OPTIMIZE)
        validator_prompt = self.prompt_generator._generate_validator_prompt(
            op_config, input_data, output_data
        )

        # Log the validator prompt
        self.console.log("[bold]Validator Prompt:[/bold]")
        self.console.log(validator_prompt)
        self.console.log("\n")  # Add a newline for better readability

        # Step 2: Use the validator prompt to assess the operation's performance
        assessment = self.evaluator._assess_operation(
            op_config, input_data, output_data, validator_prompt
        )

        # Print out the assessment
        self.console.log(
            f"[bold]Assessment for whether we should improve operation {op_config['name']}:[/bold]"
        )
        for key, value in assessment.items():
            self.console.log(f"[bold cyan]{key}:[/bold cyan] [yellow]{value}[/yellow]")
        self.console.log("\n")  # Add a newline for better readability

        self.runner.optimizer.captured_output.save_optimizer_output(
            stage_type=StageType.SHOULD_OPTIMIZE,
            output={
                "validator_prompt": validator_prompt,
                "needs_improvement": assessment.get("needs_improvement", True),
                "reasons": assessment.get("reasons", []),
                "improvements": assessment.get("improvements", []),
            },
        )
        self.console.post_optimizer_rationale(
            assessment.get("needs_improvement", True),
            "\n".join(assessment.get("reasons", []))
            + "\n\n"
            + "\n".join(assessment.get("improvements", [])),
            validator_prompt,
        )

        return (
            input_data,
            output_data,
            model_input_context_length,
            no_change_runtime,
            validator_prompt,
            assessment,
            data_exceeds_limit,
        )

    def optimize(
        self,
        op_config: dict[str, Any],
        input_data: list[dict[str, Any]],
        plan_types: list[str] | None = ["chunk", "proj_synthesis", "glean"],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float]:
        """
        Optimize the given operation configuration for the input data.
        Uses a staged evaluation approach:
        1. For data exceeding limits: Try all plan types at once
        2. For data within limits:
            - First try gleaning/proj synthesis
            - Compare with baseline
            - Selectively try chunking plans based on initial results
        """
        # Verify that the plan types are valid
        for plan_type in plan_types:
            if plan_type not in ["chunk", "proj_synthesis", "glean"]:
                raise ValueError(
                    f"Invalid plan type: {plan_type}. Valid plan types are: chunk, proj_synthesis, glean."
                )

        (
            input_data,
            output_data,
            model_input_context_length,
            no_change_runtime,
            validator_prompt,
            assessment,
            data_exceeds_limit,
        ) = self._should_optimize_helper(op_config, input_data)

        if not self.config.get("optimizer_config", {}).get("force_decompose", False):
            if not data_exceeds_limit and not assessment.get("needs_improvement", True):
                self.console.log(
                    f"[green]No improvement needed for operation {op_config['name']}[/green]"
                )
                return (
                    [op_config],
                    output_data,
                    self.plan_generator.subplan_optimizer_cost,
                )

        # Select consistent evaluation samples
        num_evaluations = min(5, len(input_data))
        evaluation_samples = select_evaluation_samples(input_data, num_evaluations)

        if data_exceeds_limit:
            # For data exceeding limits, try all plan types at once
            return self._evaluate_all_plans(
                op_config,
                input_data,
                evaluation_samples,
                validator_prompt,
                plan_types,
                model_input_context_length,
                data_exceeds_limit=True,
            )

        # For data within limits, use staged evaluation
        return self._staged_evaluation(
            op_config,
            input_data,
            evaluation_samples,
            validator_prompt,
            plan_types,
            no_change_runtime,
            model_input_context_length,
        )

    def _select_best_plan(
        self,
        results: dict[str, tuple[float, float, list[dict[str, Any]]]],
        op_config: dict[str, Any],
        evaluation_samples: list[dict[str, Any]],
        validator_prompt: str,
        candidate_plans: dict[str, list[dict[str, Any]]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str, dict[str, int]]:
        """
        Select the best plan from evaluation results using top-k comparison.

        Returns:
            Tuple of (best plan, best output, best plan name, pairwise rankings)
        """
        # Sort results by score in descending order
        sorted_results = sorted(results.items(), key=lambda x: x[1][0], reverse=True)

        # Take the top k plans
        top_plans = sorted_results[: self.k_to_pairwise_compare]

        # Check if there are no top plans
        if len(top_plans) == 0:
            raise ValueError(
                "No valid plans were generated. Unable to proceed with optimization."
            )

        # Include any additional plans that are tied with the last plan
        tail_score = (
            top_plans[-1][1][0]
            if len(top_plans) == self.k_to_pairwise_compare
            else float("-inf")
        )
        filtered_results = dict(
            top_plans
            + [
                item
                for item in sorted_results[len(top_plans) :]
                if item[1][0] == tail_score
            ]
        )

        # Perform pairwise comparisons on filtered plans
        if len(filtered_results) > 1:
            pairwise_rankings = self.evaluator._pairwise_compare_plans(
                filtered_results, validator_prompt, op_config, evaluation_samples
            )
            best_plan_name = max(pairwise_rankings, key=pairwise_rankings.get)
        else:
            pairwise_rankings = {k: 0 for k in results.keys()}
            best_plan_name = next(iter(filtered_results))

        # Display results table
        self.console.log(
            f"\n[bold]Plan Evaluation Results for {op_config['name']} ({op_config['type']}, {len(results)} plans, {len(evaluation_samples)} samples):[/bold]"
        )
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Plan", style="dim")
        table.add_column("Score", justify="right", width=10)
        table.add_column("Runtime", justify="right", width=10)
        table.add_column("Pairwise Wins", justify="right", width=10)

        for plan_name, (score, runtime, _) in sorted_results:
            table.add_row(
                plan_name,
                f"{score:.2f}",
                f"{runtime:.2f}s",
                f"{pairwise_rankings.get(plan_name, 0)}",
            )

        self.console.log(table)
        self.console.log("\n")

        try:
            best_plan = candidate_plans[best_plan_name]
            best_output = results[best_plan_name][2]
        except KeyError:
            raise ValueError(
                f"Best plan name {best_plan_name} not found in candidate plans. Candidate plan names: {candidate_plans.keys()}"
            )

        self.console.log(
            f"[green]Current best plan: {best_plan_name} for operation {op_config['name']} "
            f"(Score: {results[best_plan_name][0]:.2f}, "
            f"Runtime: {results[best_plan_name][1]:.2f}s)[/green]"
        )

        return best_plan, best_output, best_plan_name, pairwise_rankings

    def _staged_evaluation(
        self,
        op_config: dict[str, Any],
        input_data: list[dict[str, Any]],
        evaluation_samples: list[dict[str, Any]],
        validator_prompt: str,
        plan_types: list[str],
        no_change_runtime: float,
        model_input_context_length: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float]:
        """Stage 1: Try gleaning and proj synthesis plans first"""
        candidate_plans = {"no_change": [op_config]}

        # Generate initial plans (gleaning and proj synthesis)
        if "glean" in plan_types:
            self.console.log(
                "[bold magenta]Generating gleaning plans...[/bold magenta]"
            )
            gleaning_plans = self.plan_generator._generate_gleaning_plans(
                op_config, validator_prompt
            )
            candidate_plans.update(gleaning_plans)

        if "proj_synthesis" in plan_types and not self.is_filter:
            self.console.log(
                "[bold magenta]Generating independent projection synthesis plans...[/bold magenta]"
            )
            parallel_plans = self.plan_generator._generate_parallel_plans(
                op_config, input_data
            )
            candidate_plans.update(parallel_plans)

            self.console.log(
                "[bold magenta]Generating chain projection synthesis plans...[/bold magenta]"
            )
            chain_plans = self.plan_generator._generate_chain_plans(
                op_config, input_data
            )
            candidate_plans.update(chain_plans)

        # Evaluate initial plans
        initial_results = self._evaluate_plans(
            candidate_plans,
            op_config,
            evaluation_samples,
            validator_prompt,
            no_change_runtime,
        )

        # Get best initial plan
        best_plan, best_output, best_plan_name, pairwise_rankings = (
            self._select_best_plan(
                initial_results,
                op_config,
                evaluation_samples,
                validator_prompt,
                candidate_plans,
            )
        )
        best_is_better_than_baseline = best_plan_name != "no_change"

        # Stage 2: Decide whether/how to try chunking plans
        if "chunk" in plan_types:
            if best_is_better_than_baseline:
                # Try 2 random chunking plans first
                self.console.log(
                    "[bold magenta]Trying sample of chunking plans...[/bold magenta]"
                )
                chunk_plans = self.plan_generator._generate_chunk_size_plans(
                    op_config, input_data, validator_prompt, model_input_context_length
                )

                if chunk_plans:
                    # Sample 2 random plans
                    chunk_items = list(chunk_plans.items())
                    sample_plans = dict(
                        random.sample(chunk_items, min(2, len(chunk_items)))
                    )
                    sample_results = self._evaluate_plans(
                        sample_plans, op_config, evaluation_samples, validator_prompt
                    )

                    # Do pairwise comparison between sampled plans and current best
                    current_best = {best_plan_name: initial_results[best_plan_name]}
                    current_best.update(sample_results)

                    _, _, new_best_name, new_pairwise_rankings = self._select_best_plan(
                        current_best,
                        op_config,
                        evaluation_samples,
                        validator_prompt,
                        {**{best_plan_name: best_plan}, **sample_plans},
                    )

                    if new_best_name == best_plan_name:
                        self.console.log(
                            "[yellow]Sample chunking plans did not improve results. Keeping current best plan.[/yellow]"
                        )
                        return (
                            best_plan,
                            best_output,
                            self.plan_generator.subplan_optimizer_cost,
                        )

                    # If a sampled plan wins, evaluate all chunking plans
                    self.console.log(
                        "[bold magenta]Generating all chunking plans...[/bold magenta]"
                    )
                    chunk_results = self._evaluate_plans(
                        chunk_plans, op_config, evaluation_samples, validator_prompt
                    )
                    initial_results.update(chunk_results)
                    candidate_plans.update(chunk_plans)
            else:
                # Try all chunking plans since no improvement found yet
                self.console.log(
                    "[bold magenta]Generating chunking plans...[/bold magenta]"
                )
                chunk_plans = self.plan_generator._generate_chunk_size_plans(
                    op_config, input_data, validator_prompt, model_input_context_length
                )
                chunk_results = self._evaluate_plans(
                    chunk_plans, op_config, evaluation_samples, validator_prompt
                )
                initial_results.update(chunk_results)
                candidate_plans.update(chunk_plans)

        # Final selection of best plan
        best_plan, best_output, _, final_pairwise_rankings = self._select_best_plan(
            initial_results,
            op_config,
            evaluation_samples,
            validator_prompt,
            candidate_plans,
        )

        # Capture evaluation results with pairwise rankings
        ratings = {k: v[0] for k, v in initial_results.items()}
        runtime = {k: v[1] for k, v in initial_results.items()}
        sample_outputs = {k: v[2] for k, v in initial_results.items()}
        self.runner.optimizer.captured_output.save_optimizer_output(
            stage_type=StageType.EVALUATION_RESULTS,
            output={
                "input_data": evaluation_samples,
                "all_plan_ratings": ratings,
                "all_plan_runtimes": runtime,
                "all_plan_sample_outputs": sample_outputs,
                "all_plan_pairwise_rankings": final_pairwise_rankings,
            },
        )

        self.console.post_optimizer_status(StageType.END)
        return best_plan, best_output, self.plan_generator.subplan_optimizer_cost

    def _evaluate_plans(
        self,
        plans: dict[str, list[dict[str, Any]]],
        op_config: dict[str, Any],
        evaluation_samples: list[dict[str, Any]],
        validator_prompt: str,
        no_change_runtime: float | None = None,
    ) -> dict[str, tuple[float, float, list[dict[str, Any]]]]:
        """Helper method to evaluate a set of plans in parallel"""
        results = {}
        plans_list = list(plans.items())

        for i in range(0, len(plans_list), self._num_plans_to_evaluate_in_parallel):
            batch = plans_list[i : i + self._num_plans_to_evaluate_in_parallel]
            with ThreadPoolExecutor(
                max_workers=self._num_plans_to_evaluate_in_parallel
            ) as executor:
                futures = {
                    executor.submit(
                        self.evaluator._evaluate_plan,
                        plan_name,
                        op_config,
                        plan,
                        copy.deepcopy(evaluation_samples),
                        validator_prompt,
                    ): plan_name
                    for plan_name, plan in batch
                }
                for future in as_completed(futures):
                    plan_name = futures[future]
                    try:
                        score, runtime, output = future.result(timeout=self.timeout)
                        results[plan_name] = (score, runtime, output)
                    except concurrent.futures.TimeoutError:
                        self.console.log(
                            f"[yellow]Plan {plan_name} timed out and will be skipped.[/yellow]"
                        )
                    except Exception as e:
                        self.console.log(
                            f"[red]Error in plan {plan_name}: {str(e)}[/red]"
                        )

        if "no_change" in results and no_change_runtime is not None:
            results["no_change"] = (
                results["no_change"][0],
                no_change_runtime,
                results["no_change"][2],
            )

        return results

    def _evaluate_all_plans(
        self,
        op_config: dict[str, Any],
        input_data: list[dict[str, Any]],
        evaluation_samples: list[dict[str, Any]],
        validator_prompt: str,
        plan_types: list[str],
        model_input_context_length: int,
        data_exceeds_limit: bool,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float]:
        """
        Evaluate all plans for a given operation configuration.
        """
        candidate_plans = {}

        # Generate all plans
        self.console.post_optimizer_status(StageType.CANDIDATE_PLANS)
        self.console.log(
            f"[bold magenta]Generating {len(plan_types)} plans...[/bold magenta]"
        )
        for plan_type in plan_types:
            if plan_type == "chunk":
                self.console.log(
                    "[bold magenta]Generating chunking plans...[/bold magenta]"
                )
                chunk_size_plans = self.plan_generator._generate_chunk_size_plans(
                    op_config, input_data, validator_prompt, model_input_context_length
                )
                candidate_plans.update(chunk_size_plans)
            elif plan_type == "proj_synthesis":
                if not self.is_filter:
                    self.console.log(
                        "[bold magenta]Generating independent projection synthesis plans...[/bold magenta]"
                    )
                    parallel_plans = self.plan_generator._generate_parallel_plans(
                        op_config, input_data
                    )
                    candidate_plans.update(parallel_plans)

                    self.console.log(
                        "[bold magenta]Generating chain projection synthesis plans...[/bold magenta]"
                    )
                    chain_plans = self.plan_generator._generate_chain_plans(
                        op_config, input_data
                    )
                    candidate_plans.update(chain_plans)
            elif plan_type == "glean":
                self.console.log(
                    "[bold magenta]Generating gleaning plans...[/bold magenta]"
                )
                gleaning_plans = self.plan_generator._generate_gleaning_plans(
                    op_config, validator_prompt
                )
                candidate_plans.update(gleaning_plans)

        # Capture candidate plans
        self.runner.optimizer.captured_output.save_optimizer_output(
            stage_type=StageType.CANDIDATE_PLANS,
            output=candidate_plans,
        )

        self.console.post_optimizer_status(StageType.EVALUATION_RESULTS)
        self.console.log(
            f"[bold magenta]Evaluating {len(candidate_plans)} plans...[/bold magenta]"
        )

        results = self._evaluate_plans(
            candidate_plans, op_config, evaluation_samples, validator_prompt
        )

        # Select best plan using the centralized method
        best_plan, best_output, _, pairwise_rankings = self._select_best_plan(
            results, op_config, evaluation_samples, validator_prompt, candidate_plans
        )

        # Capture evaluation results with pairwise rankings
        ratings = {k: v[0] for k, v in results.items()}
        runtime = {k: v[1] for k, v in results.items()}
        sample_outputs = {k: v[2] for k, v in results.items()}
        self.runner.optimizer.captured_output.save_optimizer_output(
            stage_type=StageType.EVALUATION_RESULTS,
            output={
                "input_data": evaluation_samples,
                "all_plan_ratings": ratings,
                "all_plan_runtimes": runtime,
                "all_plan_sample_outputs": sample_outputs,
                "all_plan_pairwise_rankings": pairwise_rankings,
            },
        )

        self.console.post_optimizer_status(StageType.END)
        return best_plan, best_output, self.plan_generator.subplan_optimizer_cost
