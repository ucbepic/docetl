import hashlib
import json
import time
from typing import Any, Dict, List, Callable, Optional, Tuple, Union
import uuid
from motion.optimizers.utils import (
    LLMClient,
    extract_jinja_variables,
)
import random
from motion.operations import get_operation
from rich.console import Console
import jinja2
import copy
from rich.table import Table
from concurrent.futures import ThreadPoolExecutor, as_completed
import concurrent


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
        config: Dict[str, Any],
        console: Console,
        llm_client: LLMClient,
        max_threads: int,
        run_operation: Callable,
        timeout: int = 10,
    ):
        """
        Initialize the MapOptimizer.

        Args:
            config (Dict[str, Any]): The configuration dictionary for the optimizer.
            console (Console): A Rich console object for pretty printing.
            llm_client (LLMClient): A client for interacting with a language model.
            max_threads (int): The maximum number of threads to use for parallel execution.
            run_operation (Callable): A function to execute operations.
            timeout (int, optional): The timeout in seconds for operation execution. Defaults to 10.
        """
        self.config = config
        self.console = console
        self.llm_client = llm_client
        self._run_operation = run_operation
        self.max_threads = max_threads
        self.timeout = timeout
        self._num_plans_to_evaluate_in_parallel = 5

    def optimize(
        self, op_config: Dict[str, Any], input_data: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
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
            Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]: A tuple containing
            the best optimization plan and its output. The plan is a list of
            operation configurations that achieve the best performance.

        """
        input_data = copy.deepcopy(input_data)
        # Add id to each input_data
        for i in range(len(input_data)):
            input_data[i]["_map_opt_id"] = str(uuid.uuid4())

        # Execute the original operation on the sample data
        no_change_start = time.time()
        output_data = self._run_operation(op_config, input_data)
        no_change_runtime = time.time() - no_change_start

        # Generate custom validator prompt
        validator_prompt = self._generate_validator_prompt(
            op_config, input_data, output_data
        )

        # Log the validator prompt
        self.console.log("[bold]Validator Prompt:[/bold]")
        self.console.log(validator_prompt)
        self.console.log("\n")  # Add a newline for better readability

        # Step 2: Use the validator prompt to assess the operation's performance
        assessment = self._assess_operation(
            op_config, input_data, output_data, validator_prompt
        )

        # Print out the assessment
        self.console.log(
            f"[bold]Assessment for whether we should improve operation {op_config['name']}:[/bold]"
        )
        self.console.log(json.dumps(assessment, indent=2))
        self.console.log("\n")  # Add a newline for better readability

        # Check if improvement is needed based on the assessment
        if assessment.get("needs_improvement", True) == False:
            self.console.log(
                f"[green]No improvement needed for operation {op_config['name']}[/green]"
            )
            return [op_config], output_data

        # Generate improved prompt plan
        improved_prompt_plan = self._get_improved_prompt(
            op_config, assessment, input_data
        )

        # Generate chunk size plans
        chunk_size_plans = self._generate_chunk_size_plans(op_config, input_data)

        # Generate gleaning plans
        gleaning_plans = self._generate_gleaning_plans(op_config, validator_prompt)

        # Generate chain decomposition plans
        chain_plans = self._generate_chain_plans(op_config, input_data)

        # Generate parallel map plans
        parallel_plans = self._generate_parallel_plans(op_config, input_data)

        # Evaluate all plans
        plans_to_evaluate = {
            "improved_instructions": improved_prompt_plan,
            "no_change": [op_config],
            **chunk_size_plans,
            **gleaning_plans,
            **chain_plans,
            **parallel_plans,
        }

        # Select consistent evaluation samples
        num_evaluations = min(5, len(input_data))
        evaluation_samples = self._select_evaluation_samples(
            input_data, num_evaluations
        )

        results = {}
        plans_list = list(plans_to_evaluate.items())
        for i in range(0, len(plans_list), self._num_plans_to_evaluate_in_parallel):
            batch = plans_list[i : i + self._num_plans_to_evaluate_in_parallel]
            with ThreadPoolExecutor(
                max_workers=self._num_plans_to_evaluate_in_parallel
            ) as executor:
                futures = {
                    executor.submit(
                        self._evaluate_plan,
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

        self.console.log(
            f"\n[bold]Plan Evaluation Results for {op_config['name']} ({op_config['type']}, {len(scores)} plans, {num_evaluations} samples):[/bold]"
        )
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Plan", style="dim")
        table.add_column("Score", justify="right", width=10)
        table.add_column("Runtime", justify="right", width=10)
        table.add_column("Pairwise Wins", justify="right", width=10)

        # Sort results by score in descending order
        sorted_results = sorted(results.items(), key=lambda x: x[1][0], reverse=True)

        # Take the top 6 plans
        top_plans = sorted_results[:6]

        # Include any additional plans that are tied with the last plan
        tail_score = top_plans[-1][1][0] if len(top_plans) == 6 else float("-inf")
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
            pairwise_rankings = self._pairwise_compare_plans(
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

        return plans_to_evaluate[best_plan_name], best_output

    # High-level planning methods

    def _generate_chunk_size_plans(
        self, op_config: Dict[str, Any], input_data: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Generate plans with different chunk sizes for the given operation.

        This method analyzes the input data and operation configuration to create
        multiple plans with varying chunk sizes. It also determines if metadata
        extraction is necessary and includes it in the plans if needed.

        Args:
            op_config (Dict[str, Any]): The configuration of the operation.
            input_data (List[Dict[str, Any]]): The input data for the operation.

        Returns:
            Dict[str, List[Dict[str, Any]]]: A dictionary of plans, where each key
            is a plan name and each value is a list of operation configurations
            that make up the plan.

        """
        split_result = self._get_split_config(op_config, input_data)

        chunk_sizes = self._generate_chunk_sizes(split_result["split_key"], input_data)

        self.console.log("[bold]Chunk Sizes to Evaluate:[/bold]")
        self.console.log(f"{chunk_sizes}")

        avg_doc_size = sum(
            len(doc[split_result["split_key"]].split()) for doc in input_data
        ) // len(input_data)
        avg_chunk_size = sum(chunk_sizes) // len(chunk_sizes)

        def determine_metadata_with_retry():
            try:
                metadata_info = self._determine_metadata_needs(
                    op_config,
                    split_result["subprompt"],
                    avg_chunk_size,
                    split_result["split_key"],
                    input_data,
                )
                return metadata_info
            except Exception as e:
                self.console.log(
                    f"[yellow]Error determining metadata needs: {e}. Retrying...[/yellow]"
                )
                try:
                    # Retry once
                    return self._determine_metadata_needs(
                        op_config,
                        split_result["subprompt"],
                        avg_chunk_size,
                        split_result["split_key"],
                        input_data,
                    )
                except Exception:
                    # Silently fail on second attempt
                    return {"needs_metadata": False}

        metadata_info = determine_metadata_with_retry()
        # Print the metadata info
        self.console.log(f"Needs metadata: {metadata_info['needs_metadata']}")
        if metadata_info["needs_metadata"]:
            self.console.log(
                f"Metadata prompt and output schema: {metadata_info.get('metadata_prompt', 'N/A')}; {metadata_info.get('output_schema', 'N/A')}"
            )
            self.console.log(f"Reason: {metadata_info.get('reason', 'N/A')}")

        # Create base operations
        # TODO: try with and without metadata
        base_operations = []
        if metadata_info["needs_metadata"]:
            base_operations.append(
                self._create_metadata_operation(
                    op_config,
                    metadata_info["metadata_prompt"],
                    metadata_info["output_schema"],
                )
            )

        # Generate sample output for the max chunk size to create the combine prompt
        max_chunk_size = max(chunk_sizes)
        peripheral_configs = self._generate_peripheral_configs(
            max_chunk_size, avg_doc_size
        )

        avg_chunk_size = sum(chunk_sizes) // len(chunk_sizes)

        # Get 2 consecutive chunks of size min_chunk_size / 2.5 words
        chunk_word_size = int(avg_chunk_size / 2.5)
        sample_chunks = []

        # Sample the largest element of split_result["split_key"] from input_data
        largest_input = max(
            input_data, key=lambda x: len(x[split_result["split_key"]].split())
        )
        sample_chunks = [
            largest_input[split_result["split_key"]].split()[i : i + chunk_word_size]
            for i in range(
                0,
                len(largest_input[split_result["split_key"]].split()),
                chunk_word_size,
            )
        ]

        if not sample_chunks or len(sample_chunks) < 2:
            raise ValueError("Not enough words in input data to generate sample chunks")

        # Generate the info extraction prompt
        info_extraction_prompt = self.generate_info_extraction_prompt(
            split_result["subprompt"], sample_chunks[0], sample_chunks[1]
        )

        # Print the info extraction prompt
        self.console.log(
            "[bold]Info Extraction Prompt (Used to Summarize Peripheral Chunks):[/bold]"
        )
        self.console.log(info_extraction_prompt)

        sample_output = copy.deepcopy(input_data)
        max_plan = copy.deepcopy(base_operations)

        split_op = self._create_split_operation(
            op_config,
            {"chunk_size": max_chunk_size},
            peripheral_configs[-1],
            split_result["split_key"],
            info_extraction_prompt,
            "gpt-4o-mini",
        )
        map_op = self._create_map_operation(
            op_config, split_result["subprompt"] + " Only process the main chunk."
        )

        unnest_ops = self._create_unnest_operations(op_config)
        max_plan.extend([split_op, map_op] + unnest_ops)

        for op in max_plan:
            sample_output = self._run_operation(op, sample_output)

        # Generate the combine prompt using the sample output
        combine_prompt, is_commutative = self._get_combine_prompt(
            op_config, sample_output
        )

        # Print the combine prompt
        self.console.log("[bold]Combine Prompt:[/bold]")
        self.console.log(combine_prompt)

        # Create the reduce operation
        reduce_op = self._create_reduce_operation(
            op_config, combine_prompt, is_commutative
        )

        # Create plans for each chunk size
        plans = {}

        for chunk_size in chunk_sizes:
            peripheral_configs = self._generate_peripheral_configs(
                chunk_size, avg_doc_size
            )

            for peripheral_config in peripheral_configs:
                plan = copy.deepcopy(base_operations)

                split_op = self._create_split_operation(
                    op_config,
                    {"chunk_size": chunk_size},
                    peripheral_config,
                    split_result["split_key"],
                    info_extraction_prompt,
                    "gpt-4o-mini",
                )
                map_op = self._create_map_operation(
                    op_config,
                    split_result["subprompt"] + " Only process the main chunk.",
                )
                unnest_ops = self._create_unnest_operations(op_config)

                plan.extend([split_op, map_op] + unnest_ops + [reduce_op])
                plan_name = f"chunk_size_{chunk_size}_peripheral_"
                if peripheral_config:
                    for direction in ["previous", "next"]:
                        if direction in peripheral_config:
                            for part, details in peripheral_config[direction].items():
                                plan_name += f"{direction}_{part}_{details.get('count', '')}_{details.get('type', 'full')}_"
                else:
                    plan_name += "none"
                plan_name = plan_name.rstrip("_")
                plans[plan_name] = plan

        return plans

    # Generate info extraction prompt for chunk context
    def generate_info_extraction_prompt(
        self, subprompt: str, sample_chunk_1: str, sample_chunk_2: str
    ) -> str:
        """
        Generate an information extraction prompt based on a given subprompt and sample chunk.

        This method creates a prompt that can be used to extract key information from chunks of text.
        The extracted information will serve as context when applying the subprompt to subsequent chunks.

        Args:
            subprompt (str): The original subprompt used for processing chunks.
            sample_chunk_1 (str): A sample chunk of text to base the extraction prompt on.
            sample_chunk_2 (str): A sample chunk of text to base the extraction prompt on.

        Returns:
            str: A prompt string designed to extract relevant information from text chunks.
        """
        system_prompt = (
            "You are an AI assistant helping to process a super long document."
        )

        user_prompt = f"""Given the following subprompt and two consecutive sample chunks, create an info_extraction_prompt that will summarize each chunk and extract key information from it. This extracted information will be used as context when applying the subprompt to subsequent chunks.

        Subprompt:
        {subprompt}

        Sample Chunk 1:
        {sample_chunk_1}

        Sample Chunk 2:
        {sample_chunk_2}
        
        For example, the summary and information extracted from sample chunk 1 will be used as context in the subprompt, along with sample chunk 2's text, when processing sample chunk 2.

        The info_extraction_prompt should:
        1. Identify and extract the most relevant information from the chunk that might be useful for running the subprompt on subsequent chunks.
        2. Be concise and focused on key details that provide context.
        3. Be a Jinja2 template, using the variable {{ chunk_content }} to represent the chunk content to extract information from.

        Provide your info_extraction_prompt as a string.
        """

        parameters = {
            "type": "object",
            "properties": {"info_extraction_prompt": {"type": "string"}},
            "required": ["info_extraction_prompt"],
        }

        response = self.llm_client.generate(
            [{"role": "user", "content": user_prompt}], system_prompt, parameters
        )

        result = json.loads(response.choices[0].message.content)
        return result["info_extraction_prompt"]

    def _generate_gleaning_plans(
        self,
        op_config: Dict[str, Any],
        validation_prompt: str,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Generate plans that use gleaning for the given operation.

        Gleaning involves iteratively refining the output of an operation
        based on validation feedback. This method creates plans with different
        numbers of gleaning rounds.

        Args:
            op_config (Dict[str, Any]): The configuration of the operation.
            validation_prompt (str): The prompt used for validating the operation's output.

        Returns:
            Dict[str, List[Dict[str, Any]]]: A dictionary of gleaning plans, where each key
            is a plan name and each value is a list containing a single operation configuration
            with gleaning parameters.

        """
        # Generate an op with gleaning num_rounds and validation_prompt
        plans = {}
        gleaning_rounds = [1]
        for gleaning_round in gleaning_rounds:
            op_config_copy = copy.deepcopy(op_config)
            op_config_copy["gleaning"] = {
                "num_rounds": gleaning_round,
                "validation_prompt": validation_prompt,
            }
            plans[f"gleaning_{gleaning_round}_rounds"] = [op_config_copy]
        return plans

    def _generate_parallel_plans(
        self, op_config: Dict[str, Any], input_data: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Generate plans that use parallel execution for the given operation.

        This method analyzes the operation's output schema and attempts to decompose
        the task into subtasks that can be executed in parallel. It then creates a
        plan that includes these parallel subtasks.

        Args:
            op_config (Dict[str, Any]): The configuration of the operation.
            input_data (List[Dict[str, Any]]): The input data for the operation.

        Returns:
            Dict[str, List[Dict[str, Any]]]: A dictionary containing a single key
        """
        output_schema = op_config["output"]["schema"]
        if len(output_schema) <= 1:
            return (
                {}
            )  # No need for parallel decomposition if there's only one output key

        system_prompt = "You are an AI assistant tasked with decomposing a complex data processing task into parallel subtasks."

        variables_in_prompt = extract_jinja_variables(op_config["prompt"])
        variables_in_prompt = [v.replace("input.", "") for v in variables_in_prompt]

        prompt = f"""
        Original task prompt:
        {op_config['prompt']}

        Output schema:
        {json.dumps(output_schema, indent=2)}

        Input data sample:
        {json.dumps({k: v for k, v in (input_data[0] if input_data else {}).items() if k in variables_in_prompt}, indent=2)}

        Decompose the original task into parallel subtasks, where each subtask produces one or more keys of the output schema.
        Assume that the subtasks can be executed independently. You cannot rely on the output of one subtask to complete another subtask. Make sure you include the same input variables as in the original task prompt. Each prompt should be a Jinja2 template.

        Provide your response in the following format:
        {{
            "subtasks": [
                {{
                    "name": "subtask_name",
                    "prompt": "subtask_prompt",
                    "output_keys": ["key1", "key2"]
                }},
                ...
            ]
        }}
        """

        parameters = {
            "type": "object",
            "properties": {
                "subtasks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "prompt": {"type": "string"},
                            "output_keys": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "additionalProperties": False,
                        "required": ["name", "prompt", "output_keys"],
                    },
                }
            },
            "required": ["subtasks"],
        }

        response = self.llm_client.generate(
            [{"role": "user", "content": prompt}],
            system_prompt,
            parameters,
        )
        result = json.loads(response.choices[0].message.content)

        # Verify that all keys in the output schema are covered by the subtask output keys
        output_schema_keys = set(op_config["output"]["schema"].keys())
        covered_keys = set()
        for subtask in result["subtasks"]:
            covered_keys.update(subtask["output_keys"])

        missing_keys = output_schema_keys - covered_keys
        if missing_keys:
            raise ValueError(
                f"Trying to create a parallel map decomposition. The following output schema keys are not covered by any subtask: {missing_keys}"
            )

        parallel_map_operation = self._create_parallel_map_operation(
            op_config, result["subtasks"]
        )

        # Print the parallel decomposition plan
        self.console.log("[bold]Parallel Decomposition Plan:[/bold ]")
        self.console.log(f"Operation: {op_config['name']}")
        self.console.log(f"Number of subtasks: {len(result['subtasks'])}")
        for idx, subtask in enumerate(result["subtasks"], 1):
            self.console.log(f"\n[bold]Subtask {idx}: {subtask['name']}[/bold]")
            self.console.log(f"Output keys: {', '.join(subtask['output_keys'])}")
            if len(subtask["prompt"]) > 500:
                self.console.log(
                    f"Prompt: {subtask['prompt'][:500]}..."
                )  # Truncate long prompts
            else:
                self.console.log(f"Prompt: {subtask['prompt']}")

        return {"parallel_map": [parallel_map_operation]}

    def _generate_chain_plans(
        self, op_config: Dict[str, Any], input_data: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Generate chain decomposition plans for the given operation.

        This method analyzes the operation configuration and input data to create a
        chain of subtasks that collectively accomplish the original task. It's particularly
        useful for complex operations that can be broken down into simpler, sequential steps.

        Args:
            op_config (Dict[str, Any]): The configuration of the original operation.
            input_data (List[Dict[str, Any]]): A sample of the input data.

        Returns:
            Dict[str, List[Dict[str, Any]]]: A dictionary containing the chain decomposition plan.
            The key is 'chain_decomposition' and the value is a list of operation configurations
            for each subtask in the chain.

        Note:
            - This method is most effective when the original task has multiple output keys
              with dependencies between them.
            - If the output schema has only one key, an empty dictionary is returned as
              chain decomposition is not necessary.
            - The method uses the LLM to generate the chain of subtasks, ensuring that
              all output keys from the original task are covered.
        """

        output_schema = op_config["output"]["schema"]
        if len(output_schema) <= 1:
            return {}  # No need for chain decomposition if there's only one output key

        variables_in_prompt = extract_jinja_variables(op_config["prompt"])
        variables_in_prompt = [v.replace("input.", "") for v in variables_in_prompt]

        system_prompt = "You are an AI assistant tasked with decomposing a complex data processing task into a chain of simpler tasks."

        prompt = f"""
        Original task prompt:
        {op_config['prompt']}

        Output schema:
        {json.dumps(output_schema, indent=2)}

        Input data sample:
        {json.dumps({k: v for k, v in input_data[0].items() if k in variables_in_prompt} if input_data else {}, indent=2)}

        Decompose the original task into a chain of subtasks, where each subtask produces one or more keys of the output schema.
        Analyze dependencies between output keys and arrange subtasks in a logical order. To access the output of a previous subtask, use the syntax {{ input.key }}. Each prompt should be a Jinja2 template.

        Provide your response in the following format:
        {{
            "subtasks": [
                {{
                    "name": "subtask_name",
                    "prompt": "subtask_prompt",
                    "output_keys": ["key1", "key2"],
                }},
                ...
            ]
        }}
        """

        parameters = {
            "type": "object",
            "properties": {
                "subtasks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "prompt": {"type": "string"},
                            "output_keys": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "additionalProperties": False,
                        "required": ["name", "prompt", "output_keys"],
                    },
                }
            },
            "required": ["subtasks"],
        }

        response = self.llm_client.generate(
            [{"role": "user", "content": prompt}],
            system_prompt,
            parameters,
        )
        result = json.loads(response.choices[0].message.content)

        # Verify that all output schema keys are covered by the subtasks' output keys
        output_schema_keys = set(output_schema.keys())
        subtask_output_keys = set()
        for subtask in result["subtasks"]:
            subtask_output_keys.update(subtask["output_keys"])

        missing_keys = output_schema_keys - subtask_output_keys
        if missing_keys:
            self.console.log(
                "[bold red]Warning:[/bold red] Some output schema keys are not covered by subtasks:"
            )
            for key in missing_keys:
                self.console.log(f"  - {key}")

            # Attempt to add missing keys to the most appropriate subtask
            for key in missing_keys:
                # Find the subtask with the most similar existing output keys
                best_subtask = max(
                    result["subtasks"],
                    key=lambda s: len(set(s["output_keys"]) & output_schema_keys),
                )
                best_subtask["output_keys"].append(key)
                self.console.log(
                    f"[yellow]Added missing key '{key}' to subtask '{best_subtask['name']}'[/yellow]"
                )

        # Verify again after attempting to add missing keys
        subtask_output_keys = set()
        for subtask in result["subtasks"]:
            subtask_output_keys.update(subtask["output_keys"])

        if output_schema_keys != subtask_output_keys:
            raise ValueError(
                "Not all output schema keys are covered by subtasks after correction attempt."
            )

        chain_plan = []
        for idx, subtask in enumerate(result["subtasks"]):
            subtask_config = copy.deepcopy(op_config)
            subtask_config["name"] = f"{op_config['name']}_subtask_{idx+1}"
            subtask_config["prompt"] = subtask["prompt"]
            subtask_config["output"]["schema"] = {
                key: output_schema[key] for key in subtask["output_keys"]
            }
            chain_plan.append(subtask_config)

        # Log the chain decomposition
        self.console.log("[bold]Chain Decomposition Plan:[/bold]")
        for idx, subtask in enumerate(chain_plan):
            self.console.log(f"[cyan]Subtask {idx+1}:[/cyan]")
            self.console.log(f"  [yellow]Name:[/yellow] {subtask['name']}")
            self.console.log(
                f"  [yellow]Prompt:[/yellow] {subtask['prompt'][:500]}..."
                if len(subtask["prompt"]) > 500
                else f"  [yellow]Prompt:[/yellow] {subtask['prompt']}"
            )
            self.console.log(
                f"  [yellow]Output Keys:[/yellow] {', '.join(subtask['output']['schema'].keys())}"
            )
            self.console.log("")  # Add a blank line between subtasks
        self.console.log("\n")  # Add a newline for better readability

        return {"chain_decomposition": chain_plan}

    # Evaluation and assessment methods

    def _pairwise_compare_plans(
        self,
        filtered_results: Dict[str, Tuple[float, float, List[Dict[str, Any]]]],
        validator_prompt: str,
        op_config: Dict[str, Any],
        input_data: List[Dict[str, Any]],
    ) -> Dict[str, int]:
        plan_names = list(filtered_results.keys())
        rankings = {plan: 0 for plan in plan_names}
        overall_prompt = op_config["prompt"]

        with ThreadPoolExecutor(
            max_workers=self._num_plans_to_evaluate_in_parallel
        ) as executor:
            futures = {}
            for i in range(len(plan_names)):
                for j in range(i + 1, len(plan_names)):
                    plan1, plan2 = plan_names[i], plan_names[j]
                    future = executor.submit(
                        self._compare_two_plans,
                        overall_prompt,
                        plan1,
                        filtered_results[plan1][2],
                        plan2,
                        filtered_results[plan2][2],
                        validator_prompt,
                        op_config,
                        input_data,
                    )
                    futures[(plan1, plan2)] = future

            for (plan1, plan2), future in futures.items():
                try:
                    comparison_result = future.result()
                    if comparison_result == plan1:
                        rankings[plan1] += 1
                    elif comparison_result == plan2:
                        rankings[plan2] += 1
                    # If comparison_result is None, it's a tie, so we don't update rankings
                except Exception as e:
                    self.console.log(
                        f"[red]Error comparing {plan1} and {plan2}: {str(e)}[/red]"
                    )

        return rankings

    def _compare_two_plans(
        self,
        overall_prompt: str,
        plan1_name: str,
        plan1_output: List[Dict[str, Any]],
        plan2_name: str,
        plan2_output: List[Dict[str, Any]],
        validator_prompt: str,
        op_config: Dict[str, Any],
        input_data: List[Dict[str, Any]],
    ) -> Optional[str]:
        system_prompt = "You are an AI assistant tasked with comparing the outputs of two ways to complete a task."

        comparisons = []
        for i in range(min(len(plan1_output), len(plan2_output), len(input_data))):
            # Extract variables from the overall prompt template using extract_jinja_variables
            variables_in_prompt = extract_jinja_variables(overall_prompt)
            variables_in_prompt = [v.replace("input.", "") for v in variables_in_prompt]

            # Filter input_data to only include relevant variables
            filtered_input = {
                k: v for k, v in input_data[i].items() if k in variables_in_prompt
            }

            output_vars = op_config.get("output", {}).get("schema", {}).keys()

            filtered_plan1_output = {
                k: v for k, v in plan1_output[i].items() if k in output_vars
            }
            filtered_plan2_output = {
                k: v for k, v in plan2_output[i].items() if k in output_vars
            }

            prompt = f"""
            Overall Task Prompt:
            {overall_prompt}

            Validator Prompt:
            {validator_prompt}

            Input:
            {json.dumps(filtered_input, indent=2)}

            Compare the outputs of two plans for this input:

            Plan 1 output:
            {json.dumps(filtered_plan1_output, indent=2)}

            Plan 2 output:
            {json.dumps(filtered_plan2_output, indent=2)}

            Based on the overall task prompt, validator prompt, input, and these outputs, which plan performed better for this specific input?
            If one plan is clearly superior, return either "plan_1" or "plan_2". If they are roughly equivalent, return "tie".

            Provide your response in the following format:
            """

            parameters = {
                "type": "object",
                "properties": {
                    "better_plan": {
                        "type": "string",
                        "enum": ["plan_1", "plan_2", "tie"],
                    },
                    "reason": {"type": "string"},
                },
                "required": ["better_plan", "reason"],
            }

            response = self.llm_client.generate(
                [{"role": "user", "content": prompt}],
                system_prompt,
                parameters,
            )
            result = json.loads(response.choices[0].message.content)
            comparisons.append(result)

        # Aggregate results
        plan1_wins = sum(1 for comp in comparisons if comp["better_plan"] == "plan_1")
        plan2_wins = sum(1 for comp in comparisons if comp["better_plan"] == "plan_2")
        ties = sum(1 for comp in comparisons if comp["better_plan"] == "tie")

        comparison_details = "\n".join(
            [
                f"Input {i+1}: \"{comp['better_plan']}\" because {comp['reason']}"
                for i, comp in enumerate(comparisons)
            ]
        )

        self.console.log(
            f"[bold magenta]Pairwise Comparison: {plan1_name} vs {plan2_name}[/bold magenta]\n"
            f"[cyan]{plan1_name} wins: {plan1_wins}[/cyan]\n"
            f"[green]{plan2_name} wins: {plan2_wins}[/green]\n"
            f"[yellow]Ties: {ties}[/yellow]\n\n"
            f"Comparison Details:\n{comparison_details}"
        )

        if plan1_wins > plan2_wins:
            return plan1_name
        elif plan2_wins > plan1_wins:
            return plan2_name
        else:
            return None  # Tie

    def _evaluate_plan(
        self,
        plan_name: str,
        op_config: Dict[str, Any],
        plan: Union[Dict[str, Any], List[Dict[str, Any]]],
        input_data: List[Dict[str, Any]],
        validator_prompt: str,
    ) -> Tuple[float, float, List[Dict[str, Any]]]:
        """
        Evaluate a single optimization plan.

        This method executes the given plan on the input data, measures its runtime,
        and assesses the quality of the output using a custom validator prompt.

        Args:
            plan_name (str): The name of the plan being evaluated.
            op_config (Dict[str, Any]): The original operation configuration.
            plan (Union[Dict[str, Any], List[Dict[str, Any]]]): The plan to be evaluated,
                which can be a single operation or a list of operations.
            input_data (List[Dict[str, Any]]): The input data to run the plan on.
            validator_prompt (str): The prompt used to assess the quality of the output.

        Returns:
            Tuple[float, float, List[Dict[str, Any]]]: A tuple containing:
                - The average quality score of the plan's output (float)
                - The runtime of the plan (float)
                - The output data produced by the plan (List[Dict[str, Any]])

        Note:
            The quality score is calculated based on the assessment of each output
            item using the validator prompt. The scoring is as follows:
            - Satisfactory: 4 points
            - Mostly Satisfactory: 3 points
            - Partially Satisfactory: 2 points
            - Unsatisfactory: 1 point
            The final score is the average of all individual scores.
        """

        if isinstance(plan, dict):
            plan = [plan]

        output_data = input_data
        start_time = time.time()
        for op in plan:
            output_data = self._run_operation(op, output_data)
        runtime = time.time() - start_time

        # Reorder output_data to match input_data
        output_data = [
            next(
                output
                for output in output_data
                if output["_map_opt_id"] == inp["_map_opt_id"]
            )
            for inp in input_data
        ]

        scores = []

        for idx in range(len(input_data)):
            # Evaluate the quality of the output using the custom validator prompt
            quality = self._assess_output_quality(
                op_config, input_data, output_data, idx, validator_prompt
            )
            score_map = {
                "Satisfactory": 4,
                "Mostly Satisfactory": 3,
                "Partially Satisfactory": 2,
                "Unsatisfactory": 1,
            }
            scores.append(score_map.get(quality["quality_category"], 0))

        average_score = sum(scores) / len(scores)

        return average_score, runtime, output_data

    def _assess_operation(
        self,
        op_config: Dict[str, Any],
        input_data: List[Dict[str, Any]],
        output_data: List[Dict[str, Any]],
        validator_prompt: str,
    ) -> Dict[str, Any]:
        system_prompt = "You are an AI assistant tasked with assessing the performance of data processing operations. Use the provided validator prompt to evaluate the operation's output."
        prompt = f"""
        {validator_prompt}

        Operation Name: {op_config['name']}
        Operation Type: {op_config['type']}
        Input Data (sample): {json.dumps(input_data[:2] if input_data else {}, indent=2)}
        Output Data (sample): {json.dumps([output for output in output_data[:2] if output['_map_opt_id'] in [input['_map_opt_id'] for input in input_data[:2]]] if output_data else {}, indent=2)}
        Current Prompt: {op_config.get('prompt', 'N/A')}

        Based on this information and the validator prompt, assess the operation's performance. Provide your assessment in the following format:
        """

        parameters = {
            "type": "object",
            "properties": {
                "needs_improvement": {"type": "boolean"},
                "reasons": {"type": "array", "items": {"type": "string"}},
                "improvements": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["needs_improvement", "reasons", "improvements"],
        }

        response = self.llm_client.generate(
            [
                {"role": "user", "content": prompt},
            ],
            system_prompt,
            parameters,
        )
        return json.loads(response.choices[0].message.content)

    def _assess_output_quality(
        self,
        op_config: Dict[str, Any],
        input_data: List[Dict[str, Any]],
        output_data: List[Dict[str, Any]],
        element_idx: int,
        validator_prompt: str,
    ) -> str:
        """
        Assess the quality of a single output element against its corresponding input.

        This method evaluates the quality of a specific output element by comparing it
        to its corresponding input element, using the provided validator prompt as a
        guideline for assessment.

        Args:
            op_config (Dict[str, Any]): The configuration of the operation being assessed.
            input_data (List[Dict[str, Any]]): The list of input data elements.
            output_data (List[Dict[str, Any]]): The list of output data elements.
            element_idx (int): The index of the specific input/output pair to assess.
            validator_prompt (str): The prompt used to guide the quality assessment.

        Returns:
            str: A JSON string containing the quality assessment, including a quality
                 category and a reason for the assessment.

        The quality assessment is categorized into four levels:
        1. "Unsatisfactory": The output failed to meet any validator prompt requirements.
        2. "Partially Satisfactory": The output met some, but not all, requirements.
        3. "Mostly Satisfactory": The output met most requirements with room for improvement.
        4. "Satisfactory": The output fully met all validator prompt requirements.

        This method uses the LLM client to generate the quality assessment based on
        the input-output pair and the validator prompt.
        """

        system_prompt = "You are an AI assistant tasked with evaluating the quality of data processing outputs."
        output_schema_keys = op_config["output"]["schema"].keys()
        document_id = input_data[element_idx]["document_id"]
        input_elem = input_data[element_idx]
        output_elem = [
            item for item in output_data if item["document_id"] == document_id
        ][0]
        output_elem = {key: output_elem[key] for key in output_schema_keys}

        prompt = f"""
        {validator_prompt}

        Input and Output Data Sample:
        {json.dumps({"input": input_elem, "output": output_elem}, indent=2)}

        Based on the validator prompt and the input-output data samples, assess the quality of the output.
        Categorize the quality into one of these four categories:
        1. "Unsatisfactory": The output failed to meet any of the validator prompt requirements.
        2. "Partially Satisfactory": The output met some of the validator prompt requirements but not all.
        3. "Mostly Satisfactory": The output met most of the validator prompt requirements but has some room for improvement.
        4. "Satisfactory": The output fully met the validator prompt requirements.

        Provide your response in the following format:
        """

        parameters = {
            "type": "object",
            "properties": {
                "quality_category": {
                    "type": "string",
                    "enum": [
                        "Unsatisfactory",
                        "Partially Satisfactory",
                        "Mostly Satisfactory",
                        "Satisfactory",
                    ],
                },
                "reason": {"type": "string"},
            },
            "required": ["quality_category", "reason"],
        }

        response = self.llm_client.generate(
            [
                {"role": "user", "content": prompt},
            ],
            system_prompt,
            parameters,
        )
        result = json.loads(response.choices[0].message.content)

        return result

    # Prompt generation and modification methods

    def _generate_validator_prompt(
        self,
        op_config: Dict[str, Any],
        input_data: List[Dict[str, Any]],
        output_data: List[Dict[str, Any]],
    ) -> str:
        system_prompt = "You are an AI assistant tasked with creating custom validation prompts for data processing operations. Your goal is to create a prompt that will assess how well the operation performed its intended task."

        prompt = f"""
        Analyze the following operation and its input/output:

        Operation Name: {op_config['name']}
        Operation Type: {op_config['type']}
        Sample Input: {json.dumps(input_data[0] if input_data else {}, indent=2)}
        Sample Output: {json.dumps(output_data[0] if output_data else {}, indent=2)}
        Current Prompt: {op_config.get('prompt', 'N/A')}

        Based on this information, create a custom validator prompt that will assess how well the original task was performed. The prompt should ask specific questions about the quality and completeness of the output, such as:
        1. Are there any instances of the target information missed?
        2. Would the output improve if the input was analyzed more carefully?
        3. Is the output format correct and consistent?
        4. Are there any errors or inconsistencies in the extracted information?

        Provide your response as a single string containing the custom validator prompt.
        """

        parameters = {
            "type": "object",
            "properties": {"validator_prompt": {"type": "string"}},
            "required": ["validator_prompt"],
        }

        response = self.llm_client.generate(
            [
                {"role": "user", "content": prompt},
            ],
            system_prompt,
            parameters,
        )
        return json.loads(response.choices[0].message.content)["validator_prompt"]

    def _get_improved_prompt(
        self,
        op_config: Dict[str, Any],
        assessment: Dict[str, Any],
        input_data_sample: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        system_prompt = "You are an AI assistant tasked with improving prompts for data processing operations."

        random_sample = random.choice(input_data_sample) if input_data_sample else {}

        prompt = f"""
        Operation Name: {op_config['name']}
        Operation Type: {op_config['type']}
        Current Prompt: {op_config.get('prompt', 'N/A')}

        Input Data Sample:
        {json.dumps(random_sample, indent=2)}

        Use the following feedback to improve the current prompt:
        {json.dumps(assessment['improvements'], indent=2)}

        Improve the current prompt to better handle the input data and produce more accurate results.     
        Note: The new prompt should only include the variables present in the current prompt verbatim. Do not introduce any new variables.

        Provide your response in the following format:
        """

        parameters = {
            "type": "object",
            "properties": {
                "new_prompt": {"type": "string"},
            },
            "required": ["new_prompt"],
        }

        response = self.llm_client.generate(
            [
                {"role": "user", "content": prompt},
            ],
            system_prompt,
            parameters,
        )
        result = json.loads(response.choices[0].message.content)

        improved_op_config = op_config.copy()
        improved_op_config["prompt"] = result["new_prompt"]
        return [improved_op_config]

    def _get_combine_prompt(
        self,
        op_config: Dict[str, Any],
        sample_output: List[Dict[str, Any]],
    ) -> Tuple[str, bool]:
        """
        Generate a combine prompt for merging chunk results in a map-reduce operation.

        This method creates a prompt that will be used to combine the results from
        processing individual chunks of data in a map-reduce operation. The combine
        prompt is designed to accomplish the original task by merging the outputs
        from various chunks.

        Args:
            op_config (Dict[str, Any]): The configuration of the original operation,
                including the original prompt and output schema.
            sample_output (List[Dict[str, Any]]): A list of sample outputs from
                processing various chunks. Each item in the list represents the
                output from a single chunk.

        Returns:
            Tuple[str, bool]: A tuple containing:
                - A Jinja2 template string that serves as the combine prompt.
                  This prompt will be used to merge the results from individual
                  chunks to produce the final output of the map-reduce operation.
                - A boolean indicating whether the combine operation is commutative.

        The method performs the following steps:
        1. Extracts relevant information from the op_config, including the original
           prompt and output schema.
        2. Prepares sample inputs based on the sample_output and the output schema.
        3. Constructs a base prompt that includes the original prompt, output schema,
           and sample inputs.
        4. Uses the LLM to generate a combine prompt based on the base prompt and
           specific guidelines.
        5. Validates the generated prompt to ensure it meets the required format
           and uses the correct variables.
        6. Determines whether the combine operation is commutative.

        Note:
            The generated combine prompt is constrained to use only the 'values'
            variable, which contains all chunk results. It must be a valid Jinja2
            template and avoid using complex logic or filters.

        Raises:
            Any exceptions raised by the underlying _generate_and_validate_prompt
            method, which may include validation errors or LLM-related issues.
        """
        system_prompt = "You are an expert data processing assistant, decomposing a task into subtasks and joining the reults."

        # Prepare sample inputs for the combine prompt
        schema = op_config["output"]["schema"]
        schema_keys = schema.keys()
        sample_inputs = json.dumps(
            [{sk: item[sk] for sk in schema_keys} for item in sample_output[:3]],
            indent=2,
        )  # Limit to 3 samples for brevity

        base_prompt = f"""Original prompt (that operates on the full input, not the individual chunks):
        {op_config['prompt']}
        
        Output schema:
        {json.dumps(op_config['output']['schema'], indent=2)}

        Sample inputs from processing various chunks:
        {sample_inputs}

        Modify the original prompt to be a prompt that will combine these chunk results to accomplish the original task. 

        Guidelines for your prompt template:
        - The only variable you are allowed to use is the values variable, which contains all chunk results. Each value is a dictionary with the keys {', '.join(op_config['output']['schema'].keys())}
        - Avoid using filters or complex logic, even though Jinja technically supports it
        - The prompt template must be a valid Jinja2 template
        - You must use the {{ values }} variable somehow (you can access specific schema keys if you'ld like)

        Provide your prompt template as a single string.
        """
        parameters = {
            "type": "object",
            "properties": {"combine_prompt": {"type": "string"}},
            "required": ["combine_prompt"],
        }

        result = self._generate_and_validate_prompt(
            base_prompt, system_prompt, parameters, op_config, is_metadata=False
        )
        combine_prompt = result["combine_prompt"]

        # Determine if the combine operation is commutative
        system_prompt_commutative = (
            "You are an AI assistant analyzing data processing tasks."
        )
        commutative_prompt = f"""
        Given the original task prompt and the combine prompt, determine if the order of combining chunk results matters.

        Original task prompt:
        {op_config['prompt']}
        
        Output schema:
        {json.dumps(op_config['output']['schema'], indent=2)}

        Sample inputs from processing various chunks:
        {sample_inputs}

        Prompt to combine results of subtasks:
        {combine_prompt}

        Does the order of combining chunk results matter? Answer with 'yes' if order matters (non-commutative) or 'no' if order doesn't matter (commutative). 
        Explain your reasoning briefly.

        For example:
        - Merging extracted key-value pairs from documents is commutative: combining {{"name": "John", "age": 30}} with {{"city": "New York", "job": "Engineer"}} yields the same result regardless of order
        - Generating a timeline of events is non-commutative: the order of events matters for maintaining chronological accuracy.

        Consider these examples when determining if the combining operation is commutative or not.
        """

        parameters_commutative = {
            "type": "object",
            "properties": {
                "is_commutative": {"type": "string", "enum": ["yes", "no"]},
                "explanation": {"type": "string"},
            },
            "required": ["is_commutative", "explanation"],
        }

        commutative_result = self._generate_and_validate_prompt(
            commutative_prompt,
            system_prompt_commutative,
            parameters_commutative,
            op_config,
            is_metadata=False,
        )

        is_commutative = commutative_result["is_commutative"] == "no"
        commutative_explanation = commutative_result["explanation"]

        self.console.log(f"[bold]Commutativity Analysis:[/bold]")
        self.console.log(f"Is commutative: {'Yes' if is_commutative else 'No'}")
        self.console.log(f"Explanation: {commutative_explanation}")

        return combine_prompt, is_commutative

    def _edit_subprompt_to_reflect_metadata(
        self,
        subprompt: str,
        metadata_schema: Dict[str, Any],
        sample_output: List[Dict[str, Any]],
    ) -> str:
        # Select only metadata_schema keys from sample_output
        filtered_sample_output = []
        for item in sample_output:
            filtered_item = {key: item[key] for key in metadata_schema if key in item}
            filtered_sample_output.append(filtered_item)

        system_prompt = "You are an AI data processing agent. We have some metadata we can add to every document, and your job is to modify the data processing task prompt to reflect the new metadata."

        prompt = f"""
        Original task prompt:
        {subprompt}

        Metadata schema:
        {json.dumps(metadata_schema, indent=2)}

        Sample metadata output (from some docs):
        {json.dumps(filtered_sample_output[:3], indent=2)}

        Edit the original subprompt to incorporate the metadata. The new subprompt should:
        1. Reference the metadata fields where relevant
        2. Provide guidance on how to use the metadata in the context of the original task
        3. Maintain the original intent and requirements of the subprompt

        Provide the edited subprompt as a single string.
        """

        parameters = {
            "type": "object",
            "properties": {"edited_subprompt": {"type": "string"}},
            "required": ["edited_subprompt"],
        }

        response = self.llm_client.generate(
            [
                {"role": "user", "content": prompt},
            ],
            system_prompt,
            parameters,
        )
        result = json.loads(response.choices[0].message.content)

        return result["edited_subprompt"]

    # Configuration and analysis methods

    def _get_split_config(
        self,
        op_config: Dict[str, Any],
        input_data_sample: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Generate a configuration for splitting the input data and processing chunks.

        This method analyzes the operation configuration and a sample of the input data
        to determine an appropriate split key and subprompt for processing chunks of the
        input data. It uses the LLM to generate a suitable configuration based on the
        operation's requirements and the structure of the input data.

        Args:
            op_config (Dict[str, Any]): The configuration of the operation.
            input_data_sample (List[Dict[str, Any]]): A sample of the input data.

        Returns:
            Dict[str, Any]: A dictionary containing the split configuration, including:
                - split_key (str): The key in the input data to be used for splitting.
                - subprompt (str): A Jinja template prompt to be applied to each chunk.

        Note:
            - The split_key is determined based on the structure of the input data.
            - The subprompt is designed to process individual chunks of the split data.
            - The subprompt's output schema matches the original operation's output schema.
            - In the subprompt, we've replace all variables of 'input.{split_key}' with 'input.chunk_content'.
        """

        system_prompt = "You are an AI assistant tasked with configuring split operations for data processing."

        random_sample = random.choice(input_data_sample) if input_data_sample else {}

        # Extract Jinja variables from the prompt
        variables_in_prompt = extract_jinja_variables(op_config.get("prompt", ""))

        # Remove 'input.' prefix from variable names
        variables_in_prompt = [v.replace("input.", "") for v in variables_in_prompt]

        # Subselect variables in random_sample based on Jinja variables in the prompt
        random_sample = {
            k: v for k, v in random_sample.items() if k in variables_in_prompt
        }

        output_schema = op_config["output"]["schema"]

        prompt = f"""
        Operation Name: {op_config['name']}
        Operation Type: {op_config['type']}
        Current Prompt: {op_config.get('prompt', 'N/A')}

        Input Data Sample:
        {json.dumps(random_sample, indent=2)}

        Determine the split key and subprompt for processing chunks of the input data.
        The split key should be a key in the input data that contains a string to be split.
        The subprompt should be designed to process individual chunks of the split data. 
        Note that the subprompt's output schema will be: {json.dumps(output_schema, indent=2)}.

        Important:
        - The subprompt should be a Jinja template.
        - The subprompt should use the variable 'input.chunk_content' instead of 'input.{{ split_key }}'.

        Provide your response in the following format:
        - split_key: The key in the input data to be used for splitting
        - subprompt: The Jinja template prompt to be applied to each chunk
        """

        parameters = {
            "type": "object",
            "properties": {
                "split_key": {"type": "string"},
                "subprompt": {"type": "string"},
            },
            "required": ["split_key", "subprompt"],
        }

        response = self.llm_client.generate(
            [
                {"role": "user", "content": prompt},
            ],
            system_prompt,
            parameters,
        )
        result = json.loads(response.choices[0].message.content)

        # Strip out "input." from split_key if it exists
        result["split_key"] = result["split_key"].replace("input.", "")

        # Validate that the split_key exists in the input data sample
        if result["split_key"] not in input_data_sample[0]:
            raise ValueError(
                f"Split key '{result['split_key']}' not found in the input data sample."
            )

        variables_in_subprompt = extract_jinja_variables(result["subprompt"])
        # Replace variables in subprompt with f"input.chunk_{split_key}"
        for variable in variables_in_subprompt:
            inp_split_key = "input.chunk_content"
            result["subprompt"] = result["subprompt"].replace(
                f"{{{{ {variable} }}}}", f"{{{{ {inp_split_key} }}}}"
            )

        self.console.log(
            f"[yellow]Breaking down operation {op_config['name']}[/yellow]"
        )
        self.console.log(f"[cyan]Subprompt:[/cyan] {result['subprompt']}")
        return result

    def _determine_metadata_needs(
        self,
        op_config: Dict[str, Any],
        subprompt: str,
        chunk_size: int,
        split_key: str,
        input_data_sample: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        needs_metadata = self._check_metadata_necessity(
            op_config, subprompt, chunk_size, split_key, input_data_sample
        )

        if needs_metadata["needs_metadata"]:
            return self._get_metadata_config(
                op_config, subprompt, chunk_size, split_key, input_data_sample
            )
        else:
            return needs_metadata

    def _check_metadata_necessity(
        self,
        op_config: Dict[str, Any],
        subprompt: str,
        chunk_size: int,
        split_key: str,
        input_data_sample: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Determine if metadata is necessary for processing document chunks.

        This method analyzes the given operation configuration, subprompt, chunk size,
        split key, and a sample of input data to decide whether additional metadata
        is required for accurate processing of document chunks.

        Args:
            op_config (Dict[str, Any]): The configuration of the original operation.
            subprompt (str): The prompt to be used for processing individual chunks.
            chunk_size (int): The size of each chunk in words.
            split_key (str): The key used to split the input data into chunks.
            input_data_sample (List[Dict[str, Any]]): A sample of the input data.

        Returns:
            Dict[str, Any]: A dictionary containing:
                - 'needs_metadata' (bool): True if metadata is needed, False otherwise.
                - 'reason' (str): An explanation for why metadata is or isn't needed.

        The method considers several factors to make this determination:
        1. The nature of the subtask as described in the subprompt.
        2. The size and content of the chunks.
        3. The structure and completeness of the input data.
        4. The potential presence of crucial information in metadata (e.g., headers,
           document structure) that might not be present in the chunks themselves.
        """
        system_prompt = "You are an AI assistant tasked with determining if metadata is needed for document processing."

        random_sample = random.choice(input_data_sample)[split_key]

        # Get the total number of words in the sample
        total_words = len(random_sample.split())

        # Ensure we don't start beyond the possible range
        max_start = max(0, total_words - chunk_size)

        # Choose a random starting point, ensuring a valid range
        if max_start > chunk_size:
            start = random.randint(chunk_size, max_start)
        else:
            start = 0

        # Extract the chunk
        words = random_sample.split()[start : start + chunk_size]
        random_chunk = " ".join(words)

        # Calculate the number of words before and after the chunk
        num_words_before = start
        num_words_after = total_words - (start + chunk_size)

        prompt = f"""
        Given the following subtask prompt:
        {subprompt}

        And a chunk size of {chunk_size} words, analyze if metadata (e.g., headers) is needed to perform the subtask.

        Here's a random sample chunk of {chunk_size} words from the input:
        "{random_chunk}"

        There are {num_words_before} words before this chunk and {num_words_after} words after this chunk in the full text.

        Full input sample:
        {json.dumps(random.choice(input_data_sample), indent=2)}

        Determine if metadata is needed to perform the subtask.

        Consider:
        1. Does the subtask require information that might be present in metadata?
        2. Is the sample chunk or full input missing any crucial information that could be in metadata?
        3. Would having metadata significantly improve the performance or accuracy of the subtask?

        Provide your response in the following format:
        """

        parameters = {
            "type": "object",
            "properties": {
                "needs_metadata": {"type": "boolean"},
                "reason": {"type": "string"},
            },
            "required": ["needs_metadata", "reason"],
        }

        response = self.llm_client.generate(
            [
                {"role": "user", "content": prompt},
            ],
            system_prompt,
            parameters,
        )
        return json.loads(response.choices[0].message.content)

    def _get_metadata_config(
        self,
        op_config: Dict[str, Any],
        subprompt: str,
        chunk_size: int,
        split_key: str,
        input_data_sample: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        system_prompt = "You are an AI assistant tasked with creating metadata extraction prompts for document processing."

        random_sample = random.choice(input_data_sample)[split_key]

        metadata_var = "input." + split_key

        base_prompt = f"""
        Given the following subtask prompt:
        {subprompt}

        And a chunk size of {chunk_size} words, create a prompt to extract metadata from each document/input.

        Full input sample:
        {random_sample}

        Provide a prompt to extract this metadata from each document/input.

        Note: The metadata prompt should be a Jinja template that is only allowed to use the split_key variable like {{ {{ metadata_var }} }} and nothing else.

        Also, provide an output schema for the metadata, which should be a dictionary mapping keys to their respective types.

        Provide your response in the following format:
        """

        parameters = {
            "type": "object",
            "properties": {
                "metadata_prompt": {"type": "string"},
                "output_schema": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "string",
                        "enum": ["string", "integer", "number", "boolean", "array"],
                    },
                },
            },
            "required": ["metadata_prompt", "output_schema"],
        }

        result = self._generate_and_validate_prompt(
            base_prompt, system_prompt, parameters, op_config, is_metadata=True
        )
        result["needs_metadata"] = True
        return result

    def _determine_context_needs(
        self,
        op_config: Dict[str, Any],
        subprompt: str,
        chunk_size: int,
        split_key: str,
        input_data_sample: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        system_prompt = "You are an AI assistant tasked with determining context needs for document chunk processing."

        # Select a random element from input_data_sample
        sample_input = random.choice(input_data_sample)

        # Extract the content to be chunked
        content = sample_input[split_key]

        # Split the content into words
        words = content.split()

        # Calculate the start index for the random chunk
        start_index = max(0, random.randint(0, len(words) - chunk_size))

        # Extract the random chunk
        random_chunk = " ".join(words[int(start_index) : int(start_index + chunk_size)])

        # Calculate number of words before and after
        num_words_before = start_index
        num_words_after = max(0, len(words) - (start_index + chunk_size))

        prompt = f"""
        Given the following subtask prompt:
        {subprompt}

        And a chunk size of {chunk_size} words, analyze if peripheral chunks or context is necessary.

        Here's a random chunk of {chunk_size} words from the input:
        "{random_chunk}"

        Number of words before the chunk: {num_words_before}
        Number of words after the chunk: {num_words_after}

        Consider:
        1. Is this chunk sufficient to perform the specific subtask, or are there ambiguous pronouns/phrases that are relevant to the subtask and require peripheral chunks/context for clarity?
        2. If peripherals are necessary, do you need previous context, next context, or both?
        3. Do you need the head/tail of the entire document as well?

        Provide your response in the following format:
        """
        # TODO: get the right peripheral chunk sizes here (or experimentally find them)

        parameters = {
            "type": "object",
            "properties": {
                "needs_peripherals": {"type": "boolean"},
                "previous_context": {"type": "boolean"},
                "next_context": {"type": "boolean"},
                "needs_document_head": {"type": "boolean"},
                "needs_document_tail": {"type": "boolean"},
                "reason": {"type": "string"},
            },
            "required": [
                "needs_peripherals",
                "previous_context",
                "next_context",
                "needs_document_head",
                "needs_document_tail",
                "reason",
            ],
        }

        response = self.llm_client.generate(
            [
                {"role": "user", "content": prompt},
            ],
            system_prompt,
            parameters,
        )
        return json.loads(response.choices[0].message.content)

    # Helper methods for generating configurations

    def _generate_chunk_sizes(
        self,
        split_key: str,
        input_data_sample: List[Dict[str, Any]],
        num_chunks: int = 5,
    ) -> List[int]:
        # Get the average document length
        avg_doc_length = sum(
            len(doc[split_key].split()) for doc in input_data_sample
        ) / len(input_data_sample)

        # Create a linspace of chunk sizes from 100 to half the average document length
        max_chunk_size = int(avg_doc_length / 2)
        return [
            int(100 + i * (max_chunk_size - 100) / (num_chunks - 1))
            for i in range(num_chunks)
        ]

    def _generate_peripheral_configs(
        self, chunk_size: int, avg_doc_size: int
    ) -> List[Dict[str, Any]]:
        """
        Generate a list of peripheral chunk configurations, considering:
        * Adaptive scaling: this scales the config based on the ratio of document to chunk size
        * Extensive context: this adds a config for when the chunk size is small relative to the document size

        This method works as follows:
        1. It starts with an empty configuration (no peripheral context) as a baseline.
        2. It calculates the maximum number of chunks based on the average document size and chunk size.
        3. It defines base configurations with minimal context (1 previous chunk, 1 previous and 1 next chunk).
        4. It applies adaptive scaling to these base configurations:
           - It scales the number of chunks based on the ratio of document size to chunk size.
           - The scaling uses a square root function to provide a balanced increase.
           - It ensures the scaled count doesn't exceed the maximum number of chunks.
        5. It adds an extensive context configuration when the chunk size is small relative to the document size:
           - This provides more context (up to 5 previous chunks and 2 next chunks) for small chunk sizes.
        6. It adds configurations with summary for small-ish chunk sizes:
           - When the chunk size is less than 1/5 of the average document size, it creates summary configurations.
        7. Finally, it deduplicates the configurations to ensure uniqueness.

        This approach allows for a range of configurations that adapt to different document and chunk sizes,
        providing more context when necessary and less when the chunks are already large relative to the document.
        """

        configs = [{}]  # Always include no peripheral context as an option

        max_chunks = max(1, avg_doc_size // chunk_size)

        # Define base configurations
        base_configs = [
            {"previous": {"tail": {"count": 1}}},
            {"previous": {"tail": {"count": 1}}, "next": {"head": {"count": 1}}},
        ]

        # Scale configurations based on document and chunk size
        scaled_configs = []
        for config in base_configs:
            scaled_config = copy.deepcopy(config)
            for direction in ["previous", "next"]:
                if direction in scaled_config:
                    for part in scaled_config[direction]:
                        count = scaled_config[direction][part]["count"]
                        scaled_count = min(
                            max_chunks,
                            max(1, int(count * (avg_doc_size / chunk_size) ** 0.5)),
                        )
                        scaled_config[direction][part]["count"] = scaled_count
            scaled_configs.append(scaled_config)

        final_configs = configs + scaled_configs

        # Add a configuration with more extensive context if the chunk size is small relative to the document size
        if chunk_size < avg_doc_size / 10:
            extensive_config = {
                "previous": {"tail": {"count": min(5, max_chunks)}},
                "next": {"head": {"count": min(2, max_chunks)}},
            }
            final_configs.append(extensive_config)

        # Add configurations with summary for small-ish chunk sizes
        if chunk_size < avg_doc_size / 5:
            summary_configs = []
            for config in final_configs:
                summary_config = copy.deepcopy(config)
                if "previous" not in summary_config:
                    summary_config["previous"] = {
                        "tail": {"count": 1, "type": "summary"}
                    }
                summary_config["previous"]["middle"] = {"type": "summary"}
                summary_configs.append(summary_config)
            final_configs.extend(summary_configs)

        # Deduplicate configs
        unique_configs = []
        for config in final_configs:
            if config not in unique_configs:
                unique_configs.append(config)
        final_configs = unique_configs

        return final_configs

    # Operation creation methods

    def _create_parallel_map_operation(
        self, op_config: Dict[str, Any], subtasks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        parallel_map_op = {
            "type": "parallel_map",
            "name": f"{op_config['name']}_parallel_map",
            "prompts": [],
            "output": op_config["output"],
            "model": op_config.get("model", self.config["default_model"]),
        }

        for subtask in subtasks:
            parallel_map_op["prompts"].append(
                {
                    "name": subtask["name"],
                    "prompt": subtask["prompt"],
                    "output_keys": subtask["output_keys"],
                }
            )

        return parallel_map_op

    def _create_metadata_operation(
        self,
        op_config: Dict[str, Any],
        metadata_prompt: str,
        output_schema: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "type": "map",
            "name": f"extract_metadata_{op_config['name']}",
            "prompt": metadata_prompt,
            "model": self.config["default_model"],
            "output": {"schema": output_schema},
        }

    def _create_split_operation(
        self,
        op_config: Dict[str, Any],
        chunk_info: Dict[str, Any],
        context_info: Dict[str, Any],
        split_key: str,
        summary_prompt: str,
        summary_model: str,
    ) -> Dict[str, Any]:
        chunk_size = int(chunk_info["chunk_size"] * 1.5)
        name = f"split_{op_config['name']}"
        split_config = {
            "type": "split",
            "name": name,
            "split_key": split_key,
            "chunk_size": chunk_size,
            "peripheral_chunks": {},
            "summary_prompt": summary_prompt,
            "summary_model": summary_model,
        }

        if "previous" in context_info:
            split_config["peripheral_chunks"]["previous"] = context_info["previous"]

        if "next" in context_info:
            split_config["peripheral_chunks"]["next"] = context_info["next"]

        # Remove peripheral_chunks if it's empty
        if not split_config["peripheral_chunks"]:
            del split_config["peripheral_chunks"]

        return split_config

    def _create_map_operation(
        self, op_config: Dict[str, Any], subprompt: str
    ) -> Dict[str, Any]:
        name = f"submap_{op_config['name']}"
        return {
            "type": "map",
            "name": name,
            "prompt": subprompt,
            "model": (
                op_config["model"]
                if "model" in op_config
                else self.config["default_model"]
            ),
            "output": op_config["output"],
        }

    def _create_unnest_operations(
        self, op_config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        # Check if the output schema has a list type key and create an unnest operation for it
        output_list_keys = [
            key
            for key, value in op_config.get("output", {}).get("schema", {}).items()
            if isinstance(value, str) and value.startswith("list[")
        ]

        # Create an unnest operation for each list type key
        unnest_ops = []
        for unnest_key in output_list_keys:
            unnest_ops.append(
                {
                    "type": "unnest",
                    "name": f"unnest_{unnest_key}_{op_config['name']}",
                    "unnest_key": unnest_key,
                    "keep_empty": True,
                }
            )

        return unnest_ops

    def _create_reduce_operation(
        self, op_config: Dict[str, Any], combine_prompt: str, is_commutative: bool
    ) -> Dict[str, Any]:
        name = f"subreduce_{op_config['name']}"
        return {
            "type": "reduce",
            "name": name,
            "reduce_key": "document_id",
            "input": op_config["output"],  # subselect keys
            "prompt": combine_prompt,
            "model": (
                op_config["model"]
                if "model" in op_config
                else self.config["default_model"]
            ),
            "output": op_config["output"],
            "pass_through": True,
            "commutative": is_commutative,
        }

    # Utility methods

    def _select_evaluation_samples(
        self, input_data: List[Dict[str, Any]], num_samples: int
    ) -> List[Dict[str, Any]]:
        if len(input_data) <= num_samples:
            return input_data
        sample = random.sample(input_data, num_samples)

        return sample

    def _generate_and_validate_prompt(
        self,
        base_prompt: str,
        system_prompt: str,
        parameters: Dict[str, Any],
        op_config: Dict[str, Any],
        is_metadata: bool,
    ) -> Dict[str, Any]:
        max_retries = 3
        attempt = 0
        chat_history = [
            {"role": "user", "content": base_prompt},
        ]

        while attempt < max_retries:
            try:
                response = self.llm_client.generate(
                    chat_history,
                    system_prompt,
                    parameters,
                )
                result = json.loads(response.choices[0].message.content)
                chat_history += [
                    {"role": "assistant", "content": result},
                ]

                # Create a dummy operation to test the prompt
                dummy_op_config = {**op_config}  # Create a deep copy
                if is_metadata:
                    dummy_op_config.update(
                        {
                            "type": "map",
                            "prompt": result["metadata_prompt"],
                            "output": {"schema": result["output_schema"]},
                        }
                    )
                else:
                    dummy_op_config.update(
                        {"type": "reduce", "prompt": result["combine_prompt"]}
                    )

                operation_class = get_operation(dummy_op_config["type"])
                operation_class(
                    dummy_op_config,
                    self.config["default_model"],
                    self.max_threads,
                    self.console,
                )

                # If we reach here, the prompt is valid
                return result

            except jinja2.exceptions.TemplateError as e:
                error_message = f"Invalid Jinja2 template: {str(e)}"
            except Exception as e:
                # We only care about jinja errors
                return result

            # Print the error message to the console
            self.console.log(f"[bold red]Error:[/bold red] {error_message}")

            chat_history.append(
                {
                    "role": "user",
                    "content": f"The previous attempt failed. Error: {error_message}\n\nPlease try again, ensuring the prompt is a valid Jinja2 template and meets all requirements.",
                }
            )
            attempt += 1

        raise Exception(
            f"Failed to generate a valid prompt after {max_retries} attempts."
        )
