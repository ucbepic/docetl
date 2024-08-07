import json
from typing import Any, Dict, List, Callable, Tuple, Union
from motion.optimizers.utils import LLMClient, extract_jinja_variables
import random
from motion.operations import get_operation
from rich.console import Console
import jinja2
import copy
from rich.table import Table
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


class MapOptimizer:
    def __init__(
        self,
        config: Dict[str, Any],
        console: Console,
        llm_client: LLMClient,
        max_threads: int,
        run_operation: Callable,
        timeout: int = 60,
    ):
        self.config = config
        self.console = console
        self.llm_client = llm_client
        self._run_operation = run_operation
        self.max_threads = max_threads
        self.timeout = timeout

    def optimize(
        self, op_config: Dict[str, Any], input_data: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        # Execute the original operation on the sample data
        output_data = self._run_operation(op_config, input_data)

        # Generate custom validator prompt
        validator_prompt = self._generate_validator_prompt(
            op_config, input_data, output_data
        )

        # Step 2: Use the validator prompt to assess the operation's performance
        assessment = self._assess_operation(
            op_config, input_data, output_data, validator_prompt
        )

        # Print out the assessment
        self.console.print(
            f"[bold]Assessment for whether we should improve operation {op_config['name']}:[/bold]"
        )
        self.console.print(json.dumps(assessment, indent=2))
        self.console.print("\n")  # Add a newline for better readability

        # Check if improvement is needed based on the assessment
        if assessment.get("needs_improvement", True) == False:
            self.console.print(
                f"[green]No improvement needed for operation {op_config['name']}[/green]"
            )
            return [op_config], output_data

        # Generate improved prompt plan
        improved_prompt_plan = self._get_improved_prompt(
            op_config, assessment, input_data
        )

        # Generate chunk size plans
        chunk_size_plans = self._generate_chunk_size_plans(op_config, input_data)

        # Evaluate both plans
        plans_to_evaluate = {
            "improved_prompt_plan": improved_prompt_plan,
            **chunk_size_plans,
        }
        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            futures = {
                executor.submit(
                    self._evaluate_plan,
                    plan_name,
                    op_config,
                    plan,
                    copy.deepcopy(input_data),
                    validator_prompt,
                    num_evaluations=min(5, len(input_data)),
                ): plan_name
                for plan_name, plan in plans_to_evaluate.items()
            }
            results = {}
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Evaluating plans",
            ):
                plan_name = futures[future]
                score, output = future.result(timeout=self.timeout)
                results[plan_name] = (score, output)

        # Create a table of scores sorted in descending order
        scores = sorted(
            [(score, plan) for plan, (score, _) in results.items()], reverse=True
        )

        self.console.print(
            f"\n[bold]Score Distribution for {op_config['name']} ({op_config['type']}, {len(scores)} plans):[/bold]"
        )
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Plan", style="dim", width=30)
        table.add_column("Score", justify="right")
        for score, plan in scores:
            table.add_row(plan, f"{score:.2f}")

        self.console.print(table)
        self.console.print("\n")

        # Choose the best plan
        best_plan_name = max(results, key=lambda x: results[x][0])
        _, best_output = results[best_plan_name]
        self.console.print(
            f"[green]Choosing {best_plan_name} for operation {op_config['name']}[/green]"
        )
        return plans_to_evaluate[best_plan_name], best_output

    def _evaluate_plan(
        self,
        plan_name: str,
        op_config: Dict[str, Any],
        plan: Union[Dict[str, Any], List[Dict[str, Any]]],
        input_data: List[Dict[str, Any]],
        validator_prompt: str,
        num_evaluations: int,
    ) -> Tuple[float, List[Dict[str, Any]]]:
        if isinstance(plan, dict):
            plan = [plan]

        output_data = input_data
        for op in plan:
            output_data = self._run_operation(op, output_data)

        scores = []

        for _ in range(num_evaluations):
            # Evaluate the quality of the output using the custom validator prompt
            quality = self._assess_output_quality(
                op_config, input_data, output_data, validator_prompt
            )
            score_map = {
                "Satisfactory": 4,
                "Mostly Satisfactory": 3,
                "Partially Satisfactory": 2,
                "Unsatisfactory": 1,
            }
            scores.append(score_map.get(quality["quality_category"], 0))

        average_score = sum(scores) / num_evaluations
        # Print the quality assessment for the last evaluation
        # self.console.print(f"[bold]Quality assessment for plan {plan_name}:[/bold]")
        # self.console.print(f"\tCategory: {quality['quality_category']}")
        # self.console.print(f"\tReason: {quality['reason']}")
        # self.console.print(f"\tAverage Score: {average_score:.2f}")
        # self.console.print("\n")  # Add a newline for better readability

        return average_score, output_data

    def _assess_output_quality(
        self,
        op_config: Dict[str, Any],
        input_data: List[Dict[str, Any]],
        output_data: List[Dict[str, Any]],
        validator_prompt: str,
    ) -> str:
        system_prompt = "You are an AI assistant tasked with evaluating the quality of data processing outputs."
        output_schema_keys = op_config["output"]["schema"].keys()
        random_idx = random.randint(0, len(input_data) - 1)
        document_id = input_data[random_idx]["document_id"]
        input_elem = input_data[random_idx]
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
        1. "Unsatisfactory": The output failed to meet the validator prompt requirements.
        2. "Partially Satisfactory": The output met some of the validator prompt requirements but not all.
        3. "Mostly Satisfactory": The output mostly met the validator prompt requirements but has some room for improvement.
        3. "Satisfactory": The output fully met the validator prompt requirements.

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
        Output Data (sample): {json.dumps(output_data[:2] if output_data else {}, indent=2)}
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

    def _generate_chunk_size_plans(
        self, op_config: Dict[str, Any], input_data: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        split_result = self._get_split_config(op_config, input_data)

        chunk_sizes = self._generate_chunk_sizes(split_result["split_key"], input_data)

        self.console.print("[bold]Chunk Sizes to Evaluate:[/bold]")
        self.console.print(f"{chunk_sizes}")

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
                self.console.print(
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
        self.console.print("[bold]Metadata Information:[/bold]")
        self.console.print(f"Needs metadata: {metadata_info['needs_metadata']}")
        if metadata_info["needs_metadata"]:
            self.console.print(
                f"Metadata prompt and output schema: {metadata_info.get('metadata_prompt', 'N/A')}; {metadata_info.get('output_schema', 'N/A')}"
            )
            self.console.print(f"Reason: {metadata_info.get('reason', 'N/A')}")

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
        sample_output = copy.deepcopy(input_data)
        max_plan = copy.deepcopy(base_operations)

        split_op = self._create_split_operation(
            op_config,
            {"chunk_size": max_chunk_size},
            peripheral_configs[-1],
            split_result["split_key"],
        )
        map_op = self._create_map_operation(
            op_config, split_result["subprompt"] + " Only process the main chunk."
        )

        max_plan.extend([split_op, map_op])

        for op in max_plan:
            sample_output = self._run_operation(op, sample_output)

        # Generate the combine prompt using the sample output
        combine_prompt = self._get_combine_prompt(op_config, sample_output)

        # Print the combine prompt
        self.console.print("[bold]Combine Prompt:[/bold]")
        self.console.print(combine_prompt)

        # Create the reduce operation
        reduce_op = self._create_reduce_operation(op_config, combine_prompt)

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
                )
                map_op = self._create_map_operation(
                    op_config,
                    split_result["subprompt"] + " Only process the main chunk.",
                )

                plan.extend([split_op, map_op, reduce_op])
                plan_name = (
                    f"chunk_size_{chunk_size}_peripheral_{hash(str(peripheral_config))}"
                )
                plans[plan_name] = plan

        return plans

    def _get_split_config(
        self,
        op_config: Dict[str, Any],
        input_data_sample: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        system_prompt = "You are an AI assistant tasked with configuring split operations for data processing."

        random_sample = random.choice(input_data_sample) if input_data_sample else {}
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
        - The only variable in the subprompt should be `input.chunk_content`.

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

        variables_in_subprompt = extract_jinja_variables(result["subprompt"])
        # Replace variables in subprompt with f"input.chunk_{split_key}"
        for variable in variables_in_subprompt:
            inp_split_key = "input.chunk_content"
            result["subprompt"] = result["subprompt"].replace(
                f"{{{{ {variable} }}}}", f"{{{{ {inp_split_key} }}}}"
            )

        self.console.print(
            f"[yellow]Breaking down operation {op_config['name']}[/yellow]"
        )
        self.console.print(f"[cyan]Subprompt:[/cyan] {result['subprompt']}")
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
            self.console.print(f"[bold red]Error:[/bold red] {error_message}")

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

    def _generate_peripheral_configs(
        self, chunk_size: int, avg_doc_size: int
    ) -> List[Dict[str, Any]]:
        """
        Generate a list of peripheral chunk configurations, considering:
        * Adaptive scaling: this scales the config based on the ratio of document to chunk size
        * Extensive context: this adds a config for when the chunk size is small relative to the document size
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

        # Deduplicate configs
        unique_configs = []
        for config in final_configs:
            if config not in unique_configs:
                unique_configs.append(config)
        final_configs = unique_configs

        return final_configs

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

    def _get_combine_prompt(
        self,
        op_config: Dict[str, Any],
        sample_output: List[Dict[str, Any]],
    ) -> str:
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
        - The only variable you are allowed to use is the `values` variable, which contains all chunk results. Each value is a dictionary with the keys {', '.join(op_config['output']['schema'].keys())}
        - Avoid using filters or complex logic, even though Jinja technically supports it

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
        return result["combine_prompt"]

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
    ) -> Dict[str, Any]:
        chunk_size = int(chunk_info["chunk_size"] * 1.5)
        name = f"split_{op_config['name']}"
        split_config = {
            "type": "split",
            "name": name,
            "split_key": split_key,
            "chunk_size": chunk_size,
            "peripheral_chunks": {},
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

    def _create_reduce_operation(
        self, op_config: Dict[str, Any], combine_prompt: str
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
        }
