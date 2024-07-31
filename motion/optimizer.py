from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import yaml
from typing import Dict, List, Any, Optional, Tuple, Union
from motion.operations import get_operation
from motion.utils import load_config
from rich.console import Console
import random
import json
from litellm import completion, completion_cost
import os
import jinja2
from jinja2 import Environment, meta
import re


def extract_jinja_variables(template_string):
    # Create a Jinja2 environment
    env = Environment()

    # Parse the template
    ast = env.parse(template_string)

    # Find all the variables referenced in the template
    variables = meta.find_undeclared_variables(ast)

    # Use regex to find any additional variables that might be missed
    # This regex looks for {{ variable }} patterns
    regex_variables = set(re.findall(r"{{\s*(\w+)\s*}}", template_string))

    # Combine both sets of variables
    all_variables = variables.union(regex_variables)

    return list(all_variables)


class LLMClient:
    def __init__(self, model="gpt-4o"):
        self.model = model
        self.total_cost = 0

    def generate(self, messages, system_prompt, parameters):
        response = completion(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                *messages,
            ],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "write_output",
                        "description": "Write output to a database",
                        "parameters": parameters,
                    },
                }
            ],
            parallel_tool_calls=False,
            tool_choice={"type": "function", "function": {"name": "write_output"}},
        )
        cost = completion_cost(response)
        self.total_cost += cost
        return response


class Optimizer:
    def __init__(
        self,
        yaml_file: str,
        max_threads: Optional[int] = None,
        sample_size: int = 5,
        model: str = "gpt-4o",
    ):
        self.yaml_file_path = yaml_file
        self.config = load_config(yaml_file)
        self.sample_size = sample_size
        self.console = Console()
        self.optimized_config = self.config.copy()
        self.llm_client = LLMClient(model)
        self.max_threads = max_threads or (os.cpu_count() or 1) * 4
        self.operations_cost = 0

    def optimize(self):
        optimized_steps = []
        optimized_operations = {}
        for step in self.config["pipeline"]["steps"]:
            optimized_step, optimized_operations = self._optimize_step(step)
            optimized_steps.append(optimized_step)
            optimized_operations.update(optimized_operations)

        self.optimized_config["operations"] = optimized_operations
        self.optimized_config["pipeline"]["steps"] = optimized_steps
        self._save_optimized_config()

        self.console.print(
            f"[bold]Total agent cost: ${self.llm_client.total_cost:.2f}[/bold]"
        )
        self.console.print(
            f"[bold]Total operations cost: ${self.operations_cost:.2f}[/bold]"
        )
        self.console.print(
            f"[bold]Total cost: ${self.llm_client.total_cost + self.operations_cost:.2f}[/bold]"
        )

    def _optimize_step(
        self, step: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        input_data = self._get_sample_data(step.get("input"))

        optimized_operations = {}
        for operation in step["operations"]:
            if isinstance(operation, dict):
                operation_name = list(operation.keys())[0]
                operation_config = operation[operation_name]
            else:
                operation_name = operation
                operation_config = {}

            op_object = self.config["operations"][operation_name].copy()
            op_object.update(operation_config)
            op_object["name"] = operation_name

            optimized_ops, input_data = self._optimize_operation(op_object, input_data)
            for op in optimized_ops:
                op_name = op.pop("name")
                optimized_operations[op_name] = op

        optimized_step = step.copy()
        optimized_step["operations"] = list(optimized_operations.keys())
        return optimized_step, optimized_operations

    def _get_sample_data(self, dataset_name: str) -> List[Dict[str, Any]]:
        if dataset_name is None:
            return []

        dataset = self.config["datasets"][dataset_name]
        if dataset["type"] == "file":
            with open(dataset["path"], "r") as f:
                data = json.load(f)
            return random.sample(data, min(self.sample_size, len(data)))
        else:
            raise ValueError(f"Unsupported dataset type: {dataset['type']}")

    def _optimize_operation(
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

        # Evaluate both plans
        plans_to_evaluate = {
            "improved_prompt_plan": self._get_improved_prompt(
                op_config, assessment, input_data
            ),
            "breakdown_plan": self._get_split_config(op_config, input_data),
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
                ): plan_name
                for plan_name, plan in plans_to_evaluate.items()
            }
            results = {}
            for future in as_completed(futures):
                plan_name = futures[future]
                score, output = future.result()
                results[plan_name] = (score, output)

        # Choose the best plan
        best_plan_name = max(results, key=lambda x: results[x][0])
        _, best_output = results[best_plan_name]
        self.console.print(
            f"[green]Choosing {best_plan_name} for operation {op_config['name']}[/green]"
        )
        return plans_to_evaluate[best_plan_name], best_output

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
        return json.loads(response.choices[0].message.tool_calls[0].function.arguments)[
            "validator_prompt"
        ]

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
        return json.loads(response.choices[0].message.tool_calls[0].function.arguments)

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
        result = json.loads(
            response.choices[0].message.tool_calls[0].function.arguments
        )

        improved_op_config = op_config.copy()
        improved_op_config["prompt"] = result["new_prompt"]
        return [improved_op_config]

    def _get_split_config(
        self,
        op_config: Dict[str, Any],
        input_data_sample: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
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
        result = json.loads(
            response.choices[0].message.tool_calls[0].function.arguments
        )

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
        return self._handle_split_operation(
            op_config,
            result["subprompt"],
            result["split_key"],
            input_data_sample,
        )

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
        return json.loads(response.choices[0].message.tool_calls[0].function.arguments)

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
                result = json.loads(
                    response.choices[0].message.tool_calls[0].function.arguments
                )
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

    def _handle_split_operation(
        self,
        op_config: Dict[str, Any],
        subprompt: str,
        split_key: str,
        input_data_sample: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        chunk_info = self._determine_chunk_size(
            op_config, subprompt, split_key, input_data_sample
        )

        # Print the chunk info
        self.console.print("[bold]Chunk Information:[/bold]")
        self.console.print(f"Minimum chunk size: {chunk_info['min_chunk_size']} words")
        self.console.print(f"Maximum chunk size: {chunk_info['max_chunk_size']} words")
        self.console.print(f"Reason: {chunk_info['reason']}")

        # Set chunk_size to the average of the min and max
        # TODO: explore relevant chunk sizes here
        chunk_info["chunk_size"] = int(
            (chunk_info["min_chunk_size"] + chunk_info["max_chunk_size"]) / 2
        )

        context_info = self._determine_context_needs(
            op_config, subprompt, chunk_info["chunk_size"], split_key, input_data_sample
        )

        # Print the context info
        self.console.print("[bold]Context Information:[/bold]")
        self.console.print(
            f"Needs peripherals: {context_info['needs_peripherals']} because {context_info['reason']}"
        )
        self.console.print(f"Previous context: {context_info['previous_context']}")
        self.console.print(f"Next context: {context_info['next_context']}")

        metadata_info = self._determine_metadata_needs(
            op_config, subprompt, chunk_info["chunk_size"], split_key, input_data_sample
        )
        # Print the metadata info
        self.console.print("[bold]Metadata Information:[/bold]")
        self.console.print(f"Needs metadata: {metadata_info['needs_metadata']}")
        self.console.print(
            f"Metadata prompt and output schema: {metadata_info.get('metadata_prompt', 'N/A')}; {metadata_info.get('output_schema', 'N/A')}"
        )
        self.console.print(f"Reason: {metadata_info['reason']}")

        operations = []
        sample_output = copy.deepcopy(input_data_sample)

        if metadata_info["needs_metadata"]:
            operations.append(
                self._create_metadata_operation(
                    op_config,
                    metadata_info["metadata_prompt"],
                    metadata_info["output_schema"],
                )
            )

            # Execute the operations to get sample output
            sample_output = self._run_operation(operations[0], sample_output)

            # edit the subprompt to reflect the metadata
            subprompt = self._edit_subprompt_to_reflect_metadata(
                subprompt, metadata_info["output_schema"], sample_output
            )

            # Print the updated subprompt
            self.console.print("[bold]Updated Subprompt:[/bold]")
            self.console.print(subprompt)

        split_op = self._create_split_operation(
            op_config, chunk_info, context_info, split_key
        )
        map_op = self._create_map_operation(
            op_config, subprompt + " Only process the main chunk."
        )

        operations.append(split_op)
        operations.append(map_op)

        # Execute the operations to get sample output
        for op in [split_op, map_op]:
            sample_output = self._run_operation(op, sample_output)

        # Generate the combine prompt using the sample output
        combine_prompt = self._get_combine_prompt(op_config, sample_output)

        # Print the combine prompt
        self.console.print("[bold]Combine Prompt:[/bold]")
        self.console.print(combine_prompt)

        # Create the reduce operation
        reduce_op = self._create_reduce_operation(op_config, combine_prompt)
        operations.append(reduce_op)

        return operations

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
        result = json.loads(
            response.choices[0].message.tool_calls[0].function.arguments
        )

        return result["edited_subprompt"]

    def _determine_chunk_size(
        self,
        op_config: Dict[str, Any],
        subprompt: str,
        split_key: str,
        input_data_sample: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        # TODO: posibly just binary search for the right chunk size

        system_prompt = "You are an AI assistant helping with processing documents, identifying how to split documents into smaller chunks that can be processed one at a time."

        chunk_sizes = []

        def process_sample(sample_input):
            sample_input_length = len(sample_input.split())

            prompt = f"""
            Given the following subtask prompt:
            {subprompt}

            And a sample input (of {sample_input_length} words):
            {sample_input}

            Identify a small, cohesive chunk of text that forms a logical unit and can be understood independently for this task.
            Provide the first few words and last few words of this chunk; preserving the exact formatting/punctuation/etc. so we can programmatically extract them. Also provide an estimate for the number of words in this chunk.

            Provide your response in the following format:
            """

            parameters = {
                "type": "object",
                "properties": {
                    "start_words": {"type": "string"},
                    "end_words": {"type": "string"},
                    "num_words": {"type": "integer"},
                },
                "required": ["start_words", "end_words", "num_words"],
            }

            response = self.llm_client.generate(
                [
                    {"role": "user", "content": prompt},
                ],
                system_prompt,
                parameters,
            )
            result = json.loads(
                response.choices[0].message.tool_calls[0].function.arguments
            )

            # Extract the chunk and calculate its size
            start_index = sample_input.index(result["start_words"])
            end_index = sample_input.index(result["end_words"]) + len(
                result["end_words"]
            )
            chunk = sample_input[start_index:end_index]
            chunk_size = len(chunk.split())

            return chunk_size

        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            futures = [
                executor.submit(
                    process_sample, random.choice(input_data_sample)[split_key]
                )
                for _ in range(8)
            ]
            for future in as_completed(futures):
                try:
                    chunk_size = future.result()
                    chunk_sizes.append(chunk_size)
                except Exception as e:
                    pass
            self.console.print(
                f"Identified chunk sizes: {', '.join(map(str, chunk_sizes))} words"
            )

        # Calculate min, max, and average chunk sizes
        min_chunk_size = min(chunk_sizes)
        max_chunk_size = max(chunk_sizes)
        remainders = [
            chunk_size
            for chunk_size in chunk_sizes
            if chunk_size not in [min_chunk_size, max_chunk_size]
        ]
        avg_chunk_size = sum(remainders) / len(remainders)

        self.console.print(f"Minimum chunk size: {min_chunk_size} words")
        self.console.print(f"Maximum chunk size: {max_chunk_size} words")
        self.console.print(f"Average chunk size: {avg_chunk_size} words")

        return {
            "min_chunk_size": min_chunk_size,
            "max_chunk_size": max_chunk_size,
            "avg_chunk_size": avg_chunk_size,
            "reason": f"Based on {len(chunk_sizes)} sample chunks, sizes ranging from {min_chunk_size} to {max_chunk_size} words, with an average of {avg_chunk_size:.2f} words.",
        }

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
        return json.loads(response.choices[0].message.tool_calls[0].function.arguments)

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

    def _evaluate_plan(
        self,
        plan_name: str,
        op_config: Dict[str, Any],
        plan: Union[Dict[str, Any], List[Dict[str, Any]]],
        input_data: List[Dict[str, Any]],
        validator_prompt: str,
    ) -> Tuple[float, List[Dict[str, Any]]]:
        if isinstance(plan, dict):
            plan = [plan]

        output_data = input_data
        for op in plan:
            output_data = self._run_operation(op, output_data)

        # Evaluate the quality of the output using the custom validator prompt
        quality = self._assess_output_quality(
            op_config, input_data, output_data, validator_prompt
        )
        # Print the quality assessment
        self.console.print(f"[bold]Quality assessment for plan {plan_name}:[/bold]")
        self.console.print(f"\tCategory: {quality['quality_category']}")
        self.console.print(f"\tReason: {quality['reason']}")
        self.console.print("\n")  # Add a newline for better readability
        score_map = {
            "Satisfactory": 3,
            "Partially Satisfactory": 2,
            "Unsatisfactory": 1,
        }
        return score_map.get(quality["quality_category"], 0), output_data

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
        prompt = f"""
        {validator_prompt}

        Input and Output Data Sample:
        {json.dumps({"input": input_data[random_idx], "output": {key: output_data[random_idx][key] for key in output_schema_keys}}, indent=2)}

        Based on the validator prompt and the input-output data samples, assess the quality of the output.
        Categorize the quality into one of these three categories:
        1. "Unsatisfactory": The output failed to meet the validator prompt requirements.
        2. "Partially Satisfactory": The output partially met the validator prompt requirements but has significant room for improvement.
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
        result = json.loads(
            response.choices[0].message.tool_calls[0].function.arguments
        )

        return result

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
        chunk_size = int(
            chunk_info["max_chunk_size"] * 1.5
        )  # Using max_chunk_size * 1.5 as suggested
        name = f"split_{op_config['name']}"
        split_config = {
            "type": "split",
            "name": name,
            "split_key": split_key,
            "chunk_size": chunk_size,
            "peripheral_chunks": {},
        }

        if context_info["previous_context"]:
            split_config["peripheral_chunks"]["previous"] = {
                "head": {"count": 2},
                "tail": {"count": 1.5},
            }

        if context_info["next_context"]:
            split_config["peripheral_chunks"]["next"] = {"head": {"count": 1}}

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

    def _run_operation(
        self, op_config: Dict[str, Any], input_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        operation_class = get_operation(op_config["type"])
        operation_instance = operation_class(
            op_config, self.config["default_model"], self.max_threads, self.console
        )
        output_data, cost = operation_instance.execute(input_data)
        self.operations_cost += cost
        return output_data

    def _save_optimized_config(self):
        # Create a copy of the optimized config to modify
        config_to_save = self.optimized_config.copy()

        # Recursively resolve all anchors and aliases
        def resolve_anchors(data):
            if isinstance(data, dict):
                return {k: resolve_anchors(v) for k, v in data.items()}
            elif isinstance(data, list):
                return [resolve_anchors(item) for item in data]
            else:
                return data

        resolved_config = resolve_anchors(config_to_save)

        # Use safe_dump to avoid creating anchors and aliases
        # Get the base filename without extension
        base_filename = os.path.splitext(self.yaml_file_path)[0]

        # Append '_opt' to the base filename
        optimized_filename = f"{base_filename}_opt.yaml"

        with open(optimized_filename, "w") as f:
            yaml.safe_dump(resolved_config, f, default_flow_style=False)

        self.console.print(
            "[green]Optimized config saved to optimized_config.yaml[/green]"
        )


if __name__ == "__main__":
    optimizer = Optimizer("workloads/medical/map.yaml", model="gpt-4o")
    optimizer.optimize()
