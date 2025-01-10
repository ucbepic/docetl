import concurrent
import copy
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple

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
        config (Dict[str, Any]): The configuration dictionary for the optimizer.
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
        self, op_config: Dict[str, Any], input_data: List[Dict[str, Any]]
    ) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
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
        self, op_config: Dict[str, Any], input_data: List[Dict[str, Any]]
    ) -> Tuple[
        List[Dict[str, Any]],
        List[Dict[str, Any]],
        int,
        float,
        str,
        Dict[str, Any],
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
        op_config: Dict[str, Any],
        input_data: List[Dict[str, Any]],
        plan_types: Optional[List[str]] = ["chunk", "proj_synthesis", "glean"],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], float]:
        """
        Optimize the given operation configuration for the input data.
        This method analyzes the operation and input data, generates various
        optimization plans, evaluates them, and returns the best plan along
        with its output. A key part of this process is creating a custom
        validator prompt for evaluation. The validator prompt is generated
        based on the specific task, input data, and output data. It serves
        as a critical tool for assessing the quality and correctness of
        each optimization plan's output. This custom prompt ensures that
        the evaluation is tailored to the unique requirements and nuances
        of the given operation. The types of optimization plans include:

        1. Improved Prompt Plan: Enhances the original prompt based on evaluation, aiming to improve output quality.

        2. Chunk Size Plan: Splits input data into chunks of different sizes,
           processes each chunk separately, and then combines the results. This
           can improve performance for large inputs.

        3. Gleaning Plans: Implements an iterative refinement process where the
           output is validated and improved over multiple rounds, enhancing accuracy.

        4. Chain Decomposition Plan: Breaks down complex operations into a series
           of simpler sub-operations, potentially improving overall performance
           and interpretability.

        5. Parallel Map Plan: Decomposes the task into subtasks that can be
           executed in parallel, potentially speeding up processing for
           independent operations.

        The method generates these plans, evaluates their performance using
        a custom validator, and selects the best performing plan based on
        output quality and execution time.

        Args:
            op_config (Dict[str, Any]): The configuration of the operation to optimize.
            input_data (List[Dict[str, Any]]): The input data for the operation.

        Returns:
            Tuple[List[Dict[str, Any]], List[Dict[str, Any]], float]: A tuple containing
            the best optimization plan and its output. The plan is a list of
            operation configurations that achieve the best performance.
            The cost is the cost of the optimizer (from possibly synthesizing resolves).

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

        # Check if improvement is needed based on the assessment
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

        candidate_plans = {}

        # Generate improved prompt plan
        if not data_exceeds_limit:
            #     improved_prompt_plan = self.prompt_generator._get_improved_prompt(
            #         op_config, assessment, input_data
            #     )
            #     candidate_plans["improved_instructions"] = improved_prompt_plan
            candidate_plans["no_change"] = [op_config]

        # Generate chunk size plans
        self.console.post_optimizer_status(StageType.CANDIDATE_PLANS)
        if "chunk" in plan_types:
            self.console.log(
                "[bold magenta]Generating chunking plans...[/bold magenta]"
            )
            chunk_size_plans = self.plan_generator._generate_chunk_size_plans(
                op_config, input_data, validator_prompt, model_input_context_length
            )
            for pname, plan in chunk_size_plans.items():
                candidate_plans[pname] = plan

        # Generate gleaning plans
        if not data_exceeds_limit and "glean" in plan_types:
            self.console.log(
                "[bold magenta]Generating gleaning plans...[/bold magenta]"
            )
            gleaning_plans = self.plan_generator._generate_gleaning_plans(
                op_config, validator_prompt
            )
            for pname, plan in gleaning_plans.items():
                candidate_plans[pname] = plan

        # Generate chain decomposition plans
        if not data_exceeds_limit and "proj_synthesis" in plan_types:
            if not self.is_filter:
                self.console.log(
                    "[bold magenta]Generating chain projection synthesis plans...[/bold magenta]"
                )
                chain_plans = self.plan_generator._generate_chain_plans(
                    op_config, input_data
                )
                for pname, plan in chain_plans.items():
                    candidate_plans[pname] = plan

                # Generate parallel map plans
                self.console.log(
                    "[bold magenta]Generating independent projection synthesis plans...[/bold magenta]"
                )
                parallel_plans = self.plan_generator._generate_parallel_plans(
                    op_config, input_data
                )
                for pname, plan in parallel_plans.items():
                    candidate_plans[pname] = plan

        # Select consistent evaluation samples
        num_evaluations = min(5, len(input_data))
        evaluation_samples = select_evaluation_samples(input_data, num_evaluations)

        results = {}
        plans_list = list(candidate_plans.items())

        # Capture candidate plans
        self.runner.optimizer.captured_output.save_optimizer_output(
            stage_type=StageType.CANDIDATE_PLANS,
            output=candidate_plans,
        )

        self.console.post_optimizer_status(StageType.EVALUATION_RESULTS)
        self.console.log(
            f"[bold magenta]Evaluating {len(plans_list)} plans...[/bold magenta]"
        )
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
                        # TODO: raise this error if the error is related to a Jinja error
                        self.console.log(
                            f"[red]Error in plan {plan_name}: {str(e)}[/red]"
                        )
                        import traceback

                        print(traceback.format_exc())

        # Add no change plan
        if not data_exceeds_limit:
            results["no_change"] = (
                results["no_change"][0],
                no_change_runtime,
                results["no_change"][2],
            )

        # Create a table of scores sorted in descending order
        scores = sorted(
            [(score, runtime, plan) for plan, (score, runtime, _) in results.items()],
            reverse=True,
        )

        # Sort results by score in descending order
        sorted_results = sorted(results.items(), key=lambda x: x[1][0], reverse=True)

        # Take the top 6 plans
        top_plans = sorted_results[: self.k_to_pairwise_compare]

        # Check if there are no top plans
        if len(top_plans) == 0:
            self.console.post_optimizer_status(StageType.END)
            raise ValueError(
                "Agent did not generate any plans. Unable to proceed with optimization. Try again."
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
            best_plan_name = (
                next(iter(filtered_results))
                if filtered_results
                else max(results, key=lambda x: results[x][0])
            )

        self.console.log(
            f"\n[bold]Plan Evaluation Results for {op_config['name']} ({op_config['type']}, {len(scores)} plans, {num_evaluations} samples):[/bold]"
        )
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Plan", style="dim")
        table.add_column("Score", justify="right", width=10)
        table.add_column("Runtime", justify="right", width=10)
        table.add_column("Pairwise Wins", justify="right", width=10)

        for score, runtime, plan in scores:
            table.add_row(
                plan,
                f"{score:.2f}",
                f"{runtime:.2f}s",
                f"{pairwise_rankings.get(plan, 0)}",
            )

        self.console.log(table)
        self.console.log("\n")

        _, _, best_output = results[best_plan_name]
        self.console.log(
            f"[green]Choosing {best_plan_name} for operation {op_config['name']} (Score: {results[best_plan_name][0]:.2f}, Runtime: {results[best_plan_name][1]:.2f}s)[/green]"
        )

        # Capture evaluation results
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
        return (
            candidate_plans[best_plan_name],
            best_output,
            self.plan_generator.subplan_optimizer_cost,
        )
