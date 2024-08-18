import copy
import json
from typing import Callable, Dict, Any, List

from rich.console import Console

from motion.optimizers.utils import LLMClient
from motion.optimizers.map_optimizer.operation_creators import OperationCreator
from motion.optimizers.utils import extract_jinja_variables
from motion.optimizers.map_optimizer.config_generators import ConfigGenerator
from motion.optimizers.map_optimizer.prompt_generators import PromptGenerator


class PlanGenerator:
    def __init__(
        self,
        llm_client: LLMClient,
        console: Console,
        config: Dict[str, Any],
        run_operation: Callable[
            [Dict[str, Any], List[Dict[str, Any]]], List[Dict[str, Any]]
        ],
        max_threads: int,
    ):
        self.llm_client = llm_client
        self.console = console
        self.operation_creator = OperationCreator(config)
        self.config_generator = ConfigGenerator(llm_client, console)
        self._run_operation = run_operation
        self.prompt_generator = PromptGenerator(
            llm_client, console, config, max_threads
        )

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
        split_result = self.config_generator._get_split_config(op_config, input_data)

        chunk_sizes = self.config_generator._generate_chunk_sizes(
            split_result["split_key"], input_data
        )

        self.console.log("[bold]Chunk Sizes to Evaluate:[/bold]")
        self.console.log(f"{chunk_sizes}")

        avg_doc_size = sum(
            len(doc[split_result["split_key"]].split()) for doc in input_data
        ) // len(input_data)
        avg_chunk_size = sum(chunk_sizes) // len(chunk_sizes)

        def determine_metadata_with_retry():
            try:
                metadata_info = self.config_generator._determine_metadata_needs(
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
                    return self.config_generator._determine_metadata_needs(
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
                self.operation_creator.create_metadata_operation(
                    op_config,
                    metadata_info["metadata_prompt"],
                    metadata_info["output_schema"],
                )
            )

        # Generate sample output for the max chunk size to create the combine prompt
        max_chunk_size = max(chunk_sizes)
        peripheral_configs = self.config_generator._generate_peripheral_configs(
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

        split_op = self.operation_creator.create_split_operation(
            op_config,
            {"chunk_size": max_chunk_size},
            peripheral_configs[-1],
            split_result["split_key"],
            info_extraction_prompt,
            "gpt-4o-mini",
        )
        map_op = self.operation_creator.create_map_operation(
            op_config, split_result["subprompt"] + " Only process the main chunk."
        )

        unnest_ops = self.operation_creator.create_unnest_operations(op_config)
        max_plan.extend([split_op, map_op] + unnest_ops)

        for op in max_plan:
            sample_output = self._run_operation(op, sample_output)

        # Generate the combine prompt using the sample output
        combine_prompt, is_commutative = self.prompt_generator._get_combine_prompt(
            op_config, sample_output
        )

        # Print the combine prompt
        self.console.log("[bold]Combine Prompt:[/bold]")
        self.console.log(combine_prompt)

        # Create the reduce operation
        reduce_op = self.operation_creator.create_reduce_operation(
            op_config, combine_prompt, is_commutative
        )

        # Create plans for each chunk size
        plans = {}

        for chunk_size in chunk_sizes:
            peripheral_configs = self.config_generator._generate_peripheral_configs(
                chunk_size, avg_doc_size
            )

            for peripheral_config in peripheral_configs:
                plan = copy.deepcopy(base_operations)

                split_op = self.operation_creator.create_split_operation(
                    op_config,
                    {"chunk_size": chunk_size},
                    peripheral_config,
                    split_result["split_key"],
                    info_extraction_prompt,
                    "gpt-4o-mini",
                )
                map_op = self.operation_creator.create_map_operation(
                    op_config,
                    split_result["subprompt"] + " Only process the main chunk.",
                )
                unnest_ops = self.operation_creator.create_unnest_operations(op_config)

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

        parallel_map_operation = self.operation_creator.create_parallel_map_operation(
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
