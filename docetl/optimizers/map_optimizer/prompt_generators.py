import json
import random
from typing import Any, Dict, List, Tuple

from rich.console import Console
from litellm import model_cost

from docetl.optimizers.map_optimizer.utils import generate_and_validate_prompt
from docetl.optimizers.utils import LLMClient, extract_jinja_variables
from docetl.utils import count_tokens, truncate_sample_data


class PromptGenerator:
    def __init__(
        self,
        llm_client: LLMClient,
        console: Console,
        config: Dict[str, Any],
        max_threads: int,
        is_filter: bool = False,
    ):
        self.llm_client = llm_client
        self.console = console
        self.config = config
        self.max_threads = max_threads
        self.is_filter = is_filter

    def _generate_validator_prompt(
        self,
        op_config: Dict[str, Any],
        input_data: List[Dict[str, Any]],
        output_data: List[Dict[str, Any]],
    ) -> str:
        system_prompt = "You are an AI assistant tasked with creating custom validation prompts for data processing operations. Your goal is to create a prompt that will assess how well the operation performed its intended task."

        variables_in_prompt = extract_jinja_variables(op_config["prompt"])
        variables_in_prompt = [v.replace("input.", "") for v in variables_in_prompt]
        output_keys_not_in_input = [
            k for k in output_data[0].keys() if k not in input_data[0].keys()
        ]

        # Get the model's input context length
        model_input_context_length = model_cost.get(self.llm_client.model, {}).get(
            "max_input_tokens", 8192
        )
        # Count tokens in the prompt
        prompt_tokens = count_tokens(
            op_config.get("prompt", "N/A"), self.llm_client.model
        )

        # Calculate available tokens for sample data
        available_tokens = (
            model_input_context_length - prompt_tokens - 100
        )  # 100 token buffer

        truncated_output = truncate_sample_data(
            output_data[0],
            available_tokens,
            output_keys_not_in_input + variables_in_prompt,
            self.llm_client.model,
        )

        prompt = f"""
        Analyze the following operation and its input/output:

        Operation Name: {op_config['name']}
        Operation Type: {op_config['type']}
        Sample Input & Output: {json.dumps(truncated_output, indent=2)}
        Task Prompt: {op_config.get('prompt', 'N/A')}

        Based on this information, create a custom validator prompt that will assess how well the original task was performed. The prompt should ask 2 or 3 specific questions about the quality and completeness of the output, such as:
        1. Are there any instances of the target information missed?
        2. Would the output improve if the input was analyzed more carefully?
        3. Is the output format correct and consistent?
        4. Are there any errors or inconsistencies in the extracted information?

        Provide your response as a single string containing the custom validator prompt. The prompt should be tailored to the task and avoid generic criteria.
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

    def _get_header_extraction_prompt(
        self,
        op_config: Dict[str, Any],
        input_data: List[Dict[str, Any]],
        split_key: str,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate a header extraction prompt for a split operation. This prompt will be used to extract the headers from the input data in each chunk.

        Args:
            op_config (Dict[str, Any]): The operation configuration.
            input_data (List[Dict[str, Any]]): A list of input data samples.
            split_key (str): The key used to split the data.

        Returns:
            str: The header extraction prompt.
        """
        system_prompt = (
            "You are an AI assistant tasked with extracting metadata from documents."
        )
        document = random.choice(input_data)[split_key]
        prompt = f"""Analyze the following document and extract examples of headers along with their levels. The document structure is as follows:

        {document}

        Your task:
        1. Identify different header levels in the document.
        2. Extract at least 1 example of headers for each level you identify.
        3. Describe any patterns you notice in the header formatting (e.g., numbering, indentation, font size, etc.).

        Provide your analysis in the following JSON format:

        {{
        \"header_levels\": [
            {{
            \"level\": 1,
            \"examples\": [\"Example 1\", \"Example 2\", \"Example 3\"],
            \"pattern\": \"Description of the pattern for level 1 headers\"
            }},
            {{
            \"level\": 2,
            \"examples\": [\"Example 1\", \"Example 2\", \"Example 3\"],
            \"pattern\": \"Description of the pattern for level 2 headers\"
            }},
            // Add more levels as needed
        ],
        \"overall_structure\": \"Brief description of the overall header structure and any notable observations\"
        }}

        Ensure that your analysis captures the hierarchical structure of the headers (if any) and any consistent formatting patterns that can be used to identify headers in similar documents. If there are no headers, return an empty list of header levels.
        """

        parameters = {
            "type": "object",
            "properties": {
                "header_levels": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "level": {"type": "integer"},
                            "examples": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "pattern": {"type": "string"},
                        },
                        "required": ["level", "examples", "pattern"],
                        "additionalProperties": False,
                    },
                },
                "overall_structure": {"type": "string"},
            },
            "required": ["header_levels", "overall_structure"],
            "additionalProperties": False,
        }

        response = self.llm_client.generate(
            [
                {"role": "user", "content": prompt},
            ],
            system_prompt,
            parameters,
        )
        result = json.loads(response.choices[0].message.content)

        # Check if there are any header levels identified
        if not result["header_levels"]:
            return "", {}

        header_examples = []
        for level in result["header_levels"]:
            for example in level["examples"]:
                header_examples.append(f"- \"{example}\" (level {level['level']})")

        header_extraction_prompt = f"""Analyze the following chunk of a document and extract any headers you see.

        {{ input.{split_key}_chunk }}

        Examples of headers and their levels based on the document structure:
        {chr(10).join(header_examples)}

        Overall structure: {result["overall_structure"]}

        Provide your analysis as a list of dictionaries, where each dictionary contains a 'header' (string) and 'level' (integer). For example:

        [
            {{"header": "{result['header_levels'][0]['examples'][0]}", "level": {result['header_levels'][0]['level']}}},
            {{"header": "{result['header_levels'][1]['examples'][0] if len(result['header_levels']) > 1 else ''}", "level": {result['header_levels'][1]['level'] if len(result['header_levels']) > 1 else 2}}}
        ]

        Only include headers you find in the text, do not add any that are not present. Use the patterns described for each level to identify headers:
        {chr(10).join([f"Level {level['level']}: {level['pattern']}" for level in result['header_levels']])}
        """
        output_schema = {"headers": "list[{header: string, level: integer}]"}
        return header_extraction_prompt, output_schema

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
                - A boolean indicating whether the combine operation is associative.

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
        6. Determines whether the combine operation is associative.

        Note:
            The generated combine prompt is constrained to use only the 'inputs'
            variable, which contains all chunk results. It must be a valid Jinja2
            template and avoid using complex logic or filters.

        Raises:
            Any exceptions raised by the underlying generate_and_validate_prompt
            method, which may include validation errors or LLM-related issues.
        """
        system_prompt = "You are an expert data processing assistant, decomposing a task into subtasks and joining the reults."

        # Prepare sample inputs for the combine prompt
        schema = op_config["output"]["schema"]
        schema_keys = list(schema.keys())
        if self.is_filter:
            schema_keys.append("_short_explanation")

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
        - The only variable you are allowed to use is the inputs variable, which contains all chunk results. Each value is a dictionary with the keys {', '.join(schema_keys)}
        - Avoid using filters or complex logic, even though Jinja technically supports it
        - The prompt template must be a valid Jinja2 template
        - You must use the {{ inputs }} variable somehow (you can access specific schema keys if you'ld like)

        Provide your prompt template as a single string.
        """
        parameters = {
            "type": "object",
            "properties": {"combine_prompt": {"type": "string"}},
            "required": ["combine_prompt"],
        }

        result = generate_and_validate_prompt(
            self.llm_client,
            base_prompt,
            system_prompt,
            parameters,
            op_config,
            is_metadata=False,
            config=self.config,
            max_threads=self.max_threads,
            console=self.console,
        )
        combine_prompt = result["combine_prompt"]

        # Determine if the combine operation is associative
        system_prompt_associative = (
            "You are an AI assistant analyzing data processing tasks."
        )
        associative_prompt = f"""
        Given the original task prompt and the combine prompt, determine if the order of combining chunk results matters.

        Original task prompt:
        {op_config['prompt']}

        Output schema:
        {json.dumps(op_config['output']['schema'], indent=2)}

        Sample inputs from processing various chunks:
        {sample_inputs}

        Prompt to combine results of subtasks:
        {combine_prompt}

        Does the order in which we process data matter when combining chunk results? Answer with 'yes' if order matters or 'no' if order doesn't matter.
        Explain your reasoning briefly.

        For example:
        - Merging extracted key-value pairs from documents doesn't require order: combining {{"name": "John", "age": 30}} with {{"city": "New York", "job": "Engineer"}} yields the same result regardless of order
        - Generating a timeline of events requires order: the order of events matters for maintaining chronological accuracy.

        Consider these examples when determining whether the order in which we process data matters.
        """

        parameters_order_matters = {
            "type": "object",
            "properties": {
                "order_matters": {"type": "string", "enum": ["yes", "no"]},
                "explanation": {"type": "string"},
            },
            "required": ["order_matters", "explanation"],
        }

        order_matters_result = generate_and_validate_prompt(
            self.llm_client,
            associative_prompt,
            system_prompt_associative,
            parameters_order_matters,
            op_config,
            is_metadata=False,
            config=self.config,
            max_threads=self.max_threads,
            console=self.console,
        )

        is_associative = order_matters_result["order_matters"] == "no"
        associative_explanation = order_matters_result["explanation"]

        self.console.log("[bold]Associativity Analysis:[/bold]")
        self.console.log(f"Is associative: {'Yes' if is_associative else 'No'}")
        self.console.log(f"Explanation: {associative_explanation}")

        return combine_prompt, is_associative

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

        Sample metadata output (from some docs):
        {json.dumps(filtered_sample_output[:2], indent=2)}

        Edit the original subprompt to incorporate the metadata. The new subprompt should:
        1. Reference the metadata field as `input.metadata` where relevant
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
