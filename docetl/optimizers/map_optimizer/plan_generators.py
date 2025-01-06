import copy
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple

from rich.console import Console

from docetl.optimizers.map_optimizer.config_generators import ConfigGenerator
from docetl.optimizers.map_optimizer.operation_creators import OperationCreator
from docetl.optimizers.map_optimizer.prompt_generators import PromptGenerator
from docetl.optimizers.reduce_optimizer import ReduceOptimizer
from docetl.optimizers.utils import LLMClient
from docetl.utils import extract_jinja_variables


class PlanGenerator:
    def __init__(
        self,
        runner,
        llm_client: LLMClient,
        console: Console,
        config: Dict[str, Any],
        run_operation: Callable[
            [Dict[str, Any], List[Dict[str, Any]]], List[Dict[str, Any]]
        ],
        max_threads: int,
        is_filter: bool = False,
        depth: int = 1,
    ):
        self.llm_client = llm_client
        self.console = console
        self.operation_creator = OperationCreator(config)
        self.config_generator = ConfigGenerator(
            llm_client, console, config, max_threads
        )
        self._run_operation = run_operation
        self.prompt_generator = PromptGenerator(
            runner, llm_client, console, config, max_threads, is_filter
        )
        self.max_threads = max_threads
        self.config = config
        self.subplan_optimizer_cost = 0.0
        self.is_filter = is_filter
        self.depth = depth
        self.max_depth = 2
        self.runner = runner

    def _generate_chunk_size_plans(
        self,
        op_config: Dict[str, Any],
        input_data: List[Dict[str, Any]],
        validator_prompt: str,
        token_limit: int,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Generate plans with different chunk sizes for the given operation.

        This method analyzes the input data and operation configuration to create
        multiple plans with varying chunk sizes. It also determines if metadata
        extraction is necessary and includes it in the plans if needed.

        Args:
            op_config (Dict[str, Any]): The configuration of the operation.
            input_data (List[Dict[str, Any]]): The input data for the operation.
            validator_prompt (str): The prompt used for validating the operation's output.
            token_limit (int): The maximum number of tokens allowed in the operation's input.

        Returns:
            Dict[str, List[Dict[str, Any]]]: A dictionary of plans, where each key
            is a plan name and each value is a list of operation configurations
            that make up the plan.

        """
        split_result = self.config_generator._get_split_config(op_config, input_data)
        # Generate split keys
        split_key = split_result["split_key"]
        content_key = f"{split_key}_chunk"
        summary_key = f"{split_key}_summary"
        doc_id_key = f"split_{op_config['name']}_id"
        subprompt_output_schema = split_result.get("subprompt_output_schema", {})
        if not subprompt_output_schema:
            subprompt_output_schema = op_config["output"]["schema"]
        split_subprompt = split_result["subprompt"]

        chunk_sizes = self.config_generator._generate_chunk_sizes(
            split_key, input_data, token_limit
        )

        self.console.log("[bold]Chunk Sizes to Evaluate:[/bold]")
        self.console.log(f"{chunk_sizes}")

        avg_doc_size = sum(len(doc[split_key].split()) for doc in input_data) // len(
            input_data
        )
        avg_chunk_size = sum(chunk_sizes) // len(chunk_sizes)

        def determine_metadata_with_retry():
            try:
                metadata_info = self.config_generator._determine_metadata_needs(
                    op_config,
                    split_subprompt,
                    avg_chunk_size,
                    split_key,
                    input_data,
                )
                return metadata_info
            except Exception as e:
                self.console.log(
                    f"[yellow]Error determining metadata needs: {e}. Retrying...[/yellow]"
                )
                try:
                    # Retry once
                    return self.config_generator._determine_metadata_needs(
                        op_config,
                        split_subprompt,
                        avg_chunk_size,
                        split_key,
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
            split_subprompt = "Given the following metadata about the document:\n{{ input.metadata }}\n\n" + split_subprompt

        # Create header extraction prompt
        header_extraction_prompt, header_output_schema = (
            self.prompt_generator._get_header_extraction_prompt(
                op_config, input_data, split_key
            )
        )
        if header_extraction_prompt:
            self.console.log(
                f"Inferring headers from the documents. Will apply this prompt to find headers in chunks: {header_extraction_prompt}"
            )
        else:
            self.console.log(
                "Not inferring headers from the documents. Will not apply any header extraction prompt."
            )

        # Create base operations
        # TODO: try with and without metadata
        base_operations = []
        if metadata_info["needs_metadata"]:
            base_operations.append(
                self.operation_creator.create_metadata_operation(
                    op_config,
                    metadata_info["metadata_prompt"],
                    metadata_info["output_schema"],
                )
            )

        # Generate sample output for the max chunk size to create the combine prompt
        max_chunk_size = max(chunk_sizes)
        peripheral_configs = self.config_generator._generate_peripheral_configs(
            summary_key, max_chunk_size, avg_doc_size
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
            split_subprompt,
            split_result["split_key"],
            sample_chunks[0],
            sample_chunks[1],
        )

        # Print the info extraction prompt
        self.console.log(
            "[bold]Info Extraction Prompt (Used to Summarize Peripheral Chunks):[/bold]"
        )
        self.console.log(info_extraction_prompt)

        # Synthesize the reduce operation

        sample_output = copy.deepcopy(input_data)
        max_plan = copy.deepcopy(base_operations)

        smg_ops = self.operation_creator.create_split_map_gather_operations(
            op_config,
            {"chunk_size": max_chunk_size},
            peripheral_configs[-1][0],
            split_key,
            content_key,
            info_extraction_prompt if peripheral_configs[-1][1] else None,
            "gpt-4o-mini",
            header_extraction_prompt,
            header_output_schema,
        )
        map_op = self.operation_creator.create_map_operation(
            op_config,
            subprompt_output_schema,
            split_subprompt,
        )

        # unnest_ops = self.operation_creator.create_unnest_operations(op_config)
        max_plan.extend(smg_ops)

        sample_map_input = copy.deepcopy(input_data)
        for smg_op in max_plan:
            sample_map_input = self._run_operation(smg_op, sample_map_input)

        sample_output = self._run_operation(map_op, sample_map_input, is_build=True)
        max_plan.append(map_op)

        # Generate the combine prompt using the sample output
        combine_prompt, is_associative = self.prompt_generator._get_combine_prompt(
            op_config, sample_output
        )

        # Print the combine prompt
        self.console.log("[bold]Combine Prompt:[/bold]")
        self.console.log(combine_prompt)

        # Create the reduce operation
        reduce_op = self.operation_creator.create_reduce_operation(
            op_config, combine_prompt, is_associative, doc_id_key
        )

        # First optimize the map operation once
        optimized_map_ops = [map_op]  # Default to original map op
        if not self.is_filter and op_config.get("recursively_optimize", False):
            try:
                optimized_map_ops, cost = self._recursively_optimize_subtask(
                    map_op,
                    sample_map_input,
                    "shared_submap",
                    plan_types=["proj_synthesis", "glean"]
                )
                self.subplan_optimizer_cost += cost
            except Exception as e:
                self.console.log(
                    f"[yellow]Warning: Failed to recursively optimize map operation: {e}. Using original map operation.[/yellow]"
                )

        # Then optimize the reduce operation once
        optimized_reduce_ops = [reduce_op]  # Default to original reduce op
        if not self.is_filter and op_config.get("recursively_optimize", False):
            try:
                optimized_reduce_ops, _, cost = ReduceOptimizer(
                    self.runner,
                    self.config,
                    self.console,
                    self.llm_client,
                    self.max_threads,
                    self._run_operation,
                ).optimize(reduce_op, sample_output)
                self.subplan_optimizer_cost += cost
            except Exception as e:
                import traceback    
                self.console.log(
                    f"[yellow]Warning: Failed to recursively optimize reduce operation: {e}. Using original reduce operation.[/yellow]"
                )
                self.console.log(f"[yellow]Traceback:[/yellow]\n{traceback.format_exc()}")

        # Create plans for each chunk size
        plans = {}

        for chunk_size in chunk_sizes:
            peripheral_configs = self.config_generator._generate_peripheral_configs(
                summary_key, chunk_size, avg_doc_size
            )

            # Define the _create_plan_task method outside of this loop
            def _create_plan_task(
                op_config,
                chunk_size,
                peripheral_config,
                split_result,
                info_extraction_prompt,
                base_operations,
                plan_name,
                optimized_map_ops,
                optimized_reduce_ops,
            ):
                def task():
                    smg_ops = self.operation_creator.create_split_map_gather_operations(
                        op_config,
                        {"chunk_size": chunk_size},
                        peripheral_config[0],
                        split_key,
                        content_key,
                        info_extraction_prompt if peripheral_config[1] else None,
                        self.config.get("default_model", "gpt-4o-mini"),
                        header_extraction_prompt,
                        header_output_schema,
                    )
                    
                    # Create the plan by combining all operations
                    plan = copy.deepcopy(base_operations)
                    plan.extend(smg_ops + optimized_map_ops + optimized_reduce_ops)
                    return plan_name, plan

                return task

            # Create all peripheral_config plans concurrently
            plan_tasks = []
            for peripheral_config_tuple in peripheral_configs:
                # Create plan name
                peripheral_config, _ = peripheral_config_tuple
                multiplied_chunk_size = int(chunk_size * 1.5)
                plan_name = f"chunk_size_{multiplied_chunk_size}_peripheral_"
                if peripheral_config:
                    for direction in ["previous", "next"]:
                        if direction in peripheral_config:
                            for part, details in peripheral_config[direction].items():
                                plan_name += f"{direction}_{part}_{details.get('count', '')}_{details.get('type', 'full')}_"
                else:
                    plan_name += "none"
                plan_name = plan_name.rstrip("_")

                plan_tasks.append(
                    _create_plan_task(
                        op_config,
                        chunk_size,
                        peripheral_config_tuple,
                        split_result,
                        info_extraction_prompt,
                        base_operations,
                        plan_name,
                        optimized_map_ops,
                        optimized_reduce_ops,
                    )
                )

            # Execute all plan tasks concurrently
            with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
                plan_results = list(executor.map(lambda f: f(), plan_tasks))

            # Add all plans to the candidates
            for plan_name, plan in plan_results:
                plans[plan_name] = plan

        return plans

    # Generate info extraction prompt for chunk context
    def generate_info_extraction_prompt(
        self, subprompt: str, split_key: str, sample_chunk_1: str, sample_chunk_2: str
    ) -> str:
        """
        Generate an information extraction prompt based on a given subprompt and sample chunk.

        This method creates a prompt that can be used to extract key information from chunks of text.
        The extracted information will serve as context when applying the subprompt to subsequent chunks.

        Args:
            subprompt (str): The original subprompt used for processing chunks.
            split_key (str): The key that is getting turned into a chunk.
            sample_chunk_1 (str): A sample chunk of text to base the extraction prompt on.
            sample_chunk_2 (str): A sample chunk of text to base the extraction prompt on.

        Returns:
            str: A prompt string designed to extract relevant information from text chunks.
        """
        system_prompt = (
            "You are an AI assistant helping to process a super long document."
        )
        chunk_content_key = f"input.{split_key}_chunk"

        user_prompt = f"""Given the following task prompt and two example consecutive chunks for context, create a sentence that will guide the summarization of each chunk to be more relevant to the task. The chunks will then be summarized and appended to the task prompt when performing the task, to maintain as much context as possible.

        Task prompt:
        {subprompt}

        Sample Chunk 1:
        {sample_chunk_1}

        Sample Chunk 2:
        {sample_chunk_2}

        Your task is to create a single sentence that will be appended to the following base prompt:
        f"Summarize the following chunk: {{{{ {chunk_content_key} }}}}\n\n"

        This sentence should:
        1. Guide the summarization to focus on information relevant to the task prompt.
        2. Be concise and specific to the task at hand.
        3. Ensure the summary will be useful for providing context when processing subsequent chunks.

        Provide your guiding sentence as a string.
        """

        parameters = {
            "type": "object",
            "properties": {"guiding_sentence": {"type": "string"}},
            "required": ["guiding_sentence"],
        }

        response = self.llm_client.generate(
            [{"role": "user", "content": user_prompt}], system_prompt, parameters
        )

        result = json.loads(response.choices[0].message.content)
        info_extraction_prompt = f"Summarize the following chunk: {{{{ {chunk_content_key} }}}}\n\n{result['guiding_sentence']}"

        return info_extraction_prompt

    def _evaluate_partial_plan_output(
        self,
        plan_name: str,
        op_config: Dict[str, Any],
        subprompt_output_schema: Dict[str, Any],
        split_op_output: List[Dict[str, Any]],
        map_op_output: List[Dict[str, Any]],
        task_prompt: str,
        validator_prompt: str,
    ) -> float:
        total_score = 0
        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            futures = []
            for i in range(len(split_op_output)):
                future = executor.submit(
                    self._assess_output_quality,
                    plan_name,
                    op_config,
                    subprompt_output_schema,
                    split_op_output,
                    map_op_output,
                    i,
                    task_prompt,
                    validator_prompt,
                )
                futures.append(future)

            score_map = {
                "Satisfactory": 1.0,
                "Mostly Satisfactory": 0.7,
                "Partially Satisfactory": 0.4,
                "Unsatisfactory": 0.0,
            }

            total_score = sum(
                score_map.get(
                    future.result().get("quality_category", "Unsatisfactory"), 0
                )
                for future in as_completed(futures)
                if not isinstance(future.exception(), Exception)
            )

        return total_score / len(split_op_output)

    def _assess_output_quality(
        self,
        plan_name: str,
        op_config: Dict[str, Any],
        subprompt_output_schema,
        split_op_output: List[Dict[str, Any]],
        map_op_output: List[Dict[str, Any]],
        element_idx: int,
        task_prompt: str,
        validator_prompt: str,
    ) -> Dict[str, Any]:
        system_prompt = "You are an AI assistant tasked with evaluating the quality of data processing outputs."
        output_schema_keys = subprompt_output_schema.keys()
        input_elem = split_op_output[element_idx]
        output_elem = map_op_output[element_idx]
        output_elem = {key: output_elem[key] for key in output_schema_keys}

        variables_in_prompt = extract_jinja_variables(task_prompt)
        variables_in_prompt = [v.replace("input.", "") for v in variables_in_prompt]
        input_elem = {key: input_elem[key] for key in variables_in_prompt}

        prompt = f"""Task Prompt:
        {task_prompt}

        Validation Prompt:
        {validator_prompt}

        Input and Output Data Sample:
        {json.dumps({"input": input_elem, "output": output_elem}, indent=2)}

        Important Note:
        The input provided is a chunk of the entire input document, and the task prompt was applied specifically to this chunk. The input may contain some contextual information around the chunk. Your task is to evaluate whether the output meets all the validation requirements pertaining to the context provided in this chunk, not the contextual information or the full document.

        Based on the validation prompt and the input-output data sample, assess the quality of the output for this specific chunk.
        Categorize the quality into one of these four categories:
        1. "Unsatisfactory": The output failed to meet any of the validation prompt requirements for the given chunk.
        2. "Partially Satisfactory": The output met some of the validation prompt requirements but not all for the given chunk.
        3. "Mostly Satisfactory": The output met most of the validation prompt requirements but has some room for improvement for the given chunk.
        4. "Satisfactory": The output fully met the validation prompt requirements for the given chunk.

        Remember, only consider the main chunk content when evaluating the output, not any information surrounding the chunk.
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
        return json.loads(response.choices[0].message.content)

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

        Output schema the operation will produce:
        {json.dumps(output_schema, indent=2)}

        Input data keys:
        {json.dumps(variables_in_prompt, indent=2)}

        Input data sample:
        {json.dumps({k: v for k, v in (input_data[0] if input_data else {}).items() if k in variables_in_prompt}, indent=2)}

        Decompose the original task into parallel subtasks, where each subtask produces one or more keys of the output schema.
        Assume that the subtasks can be executed independently. You cannot rely on the output of one subtask to complete another subtask. Make sure you include the same input variables as in the original task prompt. Each prompt should be a Jinja2 template. You can reference the keys of the input data using the syntax {{ input.key }}.

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
        op_output_schema = op_config["output"]["schema"]
        output_schema_keys = set(op_output_schema.keys())
        covered_keys = set()
        for subtask in result["subtasks"]:
            covered_keys.update(subtask["output_keys"])

        missing_keys = output_schema_keys - covered_keys
        # Attempt to add missing keys to the most appropriate subtask
        if missing_keys:
            self.console.log(
                "[bold yellow]Warning:[/bold yellow] Some output schema keys are not covered by subtasks. Attempting to add them to the most appropriate subtask."
            )
            for key in missing_keys:
                # Find the subtask with the most similar existing output keys
                best_subtask = max(
                    result["subtasks"],
                    key=lambda s: len(set(s["output_keys"]) & output_schema_keys),
                )
                best_subtask["output_keys"].append(key)
                covered_keys.add(key)
                self.console.log(
                    f"[yellow]Added missing key '{key}' to subtask '{best_subtask['name']}'[/yellow]"
                )

        # Check again for any remaining missing keys
        missing_keys = output_schema_keys - covered_keys

        if missing_keys:
            self.console.log(
                f"[bold red]Error in parallel map decomposition:[/bold red] Some output schema keys are not covered by subtasks: {missing_keys}"
            )
            return {}

        # Update op_output_schema if there are keys in covered_keys that are not in the output schema
        new_keys = covered_keys - output_schema_keys
        if new_keys:
            self.console.log(
                "[bold yellow]Warning:[/bold yellow] Some keys in subtasks are not in the original output schema. Adding them to the output schema."
            )
            for key in new_keys:
                op_output_schema[key] = "string"  # Default to string type for new keys
                self.console.log(
                    f"[yellow]Added new key '{key}' to output schema[/yellow]"
                )

        parallel_map_operation = self.operation_creator.create_parallel_map_operation(
            op_config, op_output_schema, result["subtasks"]
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
        
        If recursively_optimize is True in the op_config, each subtask in the chain
        will be recursively optimized using a new MapOptimizer instance.
        """
        output_schema = op_config["output"]["schema"]
        variables_in_prompt = extract_jinja_variables(op_config["prompt"])
        variables_in_prompt = [v.replace("input.", "") for v in variables_in_prompt]

        system_prompt = "You are an AI assistant tasked with decomposing a complex data processing task into a chain of simpler tasks."

        prompt = f"""
        Original task prompt:
        {op_config['prompt']}

        Output schema the operation will produce:
        {json.dumps(output_schema, indent=2)}

        Input data sample:
        {json.dumps({k: v for k, v in input_data[0].items() if k in variables_in_prompt} if input_data else {}, indent=2)}
        Think step by step to decompose the original task into a chain of subtasks, even if the steps are not explicitly outlined in the original task prompt. Break down the task into logical steps by:
        1. Analyzing the original task and output schema carefully to identify the key components and dependencies
        2. Breaking down the task into logical steps, where each step produces one or more output keys or helpful intermediate results
        3. Considering what information each step needs from previous steps
        4. Arranging the steps in a sequence that satisfies all dependencies

        Each subtask should produce one or more keys of the output schema or synthesize new intermediate keys. You can create new intermediate keys that don't exist in the given output schema if they help break down the task into simpler steps. For example, the first chain step can generate a helpful intermediate result that subsequent steps can build upon.

        To access the output of a previous subtask, use the syntax {{{{ input.key }}}}. Each prompt should be a Jinja2 template.

        Every variable you use in the prompt must be defined in the input data or the output of a previous subtask, and should be accessed like this: {{{{ input.key }}}}. You may need to reference the data for all the subtasks in the chain.

        Ensure that all keys in the original output schema are produced by the end of the chain, even if some subtasks create additional intermediate keys.

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

        if len(output_schema_keys - subtask_output_keys) > 0:
            self.console.log(
                f"[bold red]Error in chain decomposition:[/bold red] Some output schema keys are not covered by subtasks: {output_schema_keys - subtask_output_keys}"
            )
            return {}

        chain_plan = []
        for idx, subtask in enumerate(result["subtasks"]):
            subtask_config = copy.deepcopy(op_config)
            subtask_config["name"] = f"{op_config['name']}_subtask_{idx+1}"
            subtask_config["prompt"] = subtask["prompt"]
            subtask_config["output"]["schema"] = {
                key: output_schema.get(key, "string") for key in subtask["output_keys"]
            }

            # If recursive optimization is enabled, optimize each subtask
            if op_config.get("recursively_optimize", False):
                try:
                    optimized_subtask_plan, cost = self._recursively_optimize_subtask(
                        subtask_config, 
                        input_data,
                        f"chain_subtask_{idx+1}",
                        plan_types=["proj_synthesis", "glean"]
                    )
                    self.subplan_optimizer_cost += cost
                    chain_plan.extend(optimized_subtask_plan)
                except Exception as e:
                    self.console.log(
                        f"[yellow]Warning: Failed to recursively optimize subtask {idx+1}: {str(e)}. Using original subtask.[/yellow]"
                    )
                    chain_plan.append(subtask_config)
            else:
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

    def _recursively_optimize_subtask(
        self,
        subtask_config: Dict[str, Any],
        input_data: List[Dict[str, Any]],
        subtask_name: str,
        plan_types: List[str]
    ) -> Tuple[List[Dict[str, Any]], float]:
        """
        Recursively optimize a subtask using a new MapOptimizer instance.
        """
        if self.depth >= self.max_depth:
            self.console.log(
                f"[yellow]Reached maximum recursion depth ({self.max_depth}) for {subtask_name}. Using original configuration.[/yellow]"
            )
            return [subtask_config], 0

        from docetl.optimizers.map_optimizer.optimizer import MapOptimizer

        self.console.log(f"[cyan]Recursively optimizing {subtask_name} (depth {self.depth})...[/cyan]")

        subtask_optimizer = MapOptimizer(
            self.runner,
            self.config,
            self.console,
            self.llm_client,
            self.max_threads,
            self._run_operation,
            is_filter=self.is_filter,
            depth=self.depth + 1
        )

        try:
            optimized_plan, _, cost = subtask_optimizer.optimize(
                subtask_config,
                input_data,
                plan_types
            )
            return optimized_plan, cost

        except Exception as e:
            self.console.log(
                f"[yellow]Warning: Failed to recursively optimize {subtask_name}: {str(e)}. Using original configuration.[/yellow]"
            )
            return [subtask_config], 0
