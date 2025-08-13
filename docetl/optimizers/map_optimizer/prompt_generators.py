import json
import random
from typing import Any

from litellm import model_cost
from rich.console import Console

from docetl.optimizers.map_optimizer.utils import generate_and_validate_prompt
from docetl.optimizers.utils import LLMClient
from docetl.utils import count_tokens, extract_jinja_variables, truncate_sample_data


class PromptGenerator:
    def __init__(
        self,
        runner,
        llm_client: LLMClient,
        console: Console,
        config: dict[str, Any],
        max_threads: int,
        is_filter: bool = False,
    ):
        self.llm_client = llm_client
        self.console = console
        self.config = config
        self.max_threads = max_threads
        self.is_filter = is_filter
        self.runner = runner

    def _generate_validator_prompt(
        self,
        op_config: dict[str, Any],
        input_data: list[dict[str, Any]],
        output_data: list[dict[str, Any]],
    ) -> str:
        system_prompt = "You are an AI assistant tasked with creating custom validation prompts for data processing operations. Your goal is to create a prompt that will assess how well the operation performed its intended task."

        variables_in_prompt = extract_jinja_variables(op_config["prompt"])
        variables_in_prompt = [v.replace("input.", "") for v in variables_in_prompt]
        output_keys_not_in_input = [
            k for k in output_data[0].keys() if k not in input_data[0].keys()
        ]

        # Get the model's input context length
        model_input_context_length = model_cost.get(
            self.llm_client.rewrite_agent_model, {}
        ).get("max_input_tokens", 8192)
        # Count tokens in the prompt
        prompt_tokens = count_tokens(
            op_config.get("prompt", "N/A"), self.llm_client.rewrite_agent_model
        )

        # Calculate available tokens for sample data
        available_tokens = (
            model_input_context_length - prompt_tokens - 100
        )  # 100 token buffer

        truncated_output = truncate_sample_data(
            output_data[0],
            available_tokens,
            output_keys_not_in_input + variables_in_prompt,
            self.llm_client.rewrite_agent_model,
        )

        prompt = f"""
        Analyze the following operation and its input/output:

        Operation Name: {op_config['name']}
        Operation Type: {op_config['type']}
        Sample Input & Output: {json.dumps(truncated_output, indent=2)}
        Task Prompt: {op_config.get('prompt', 'N/A')}

        Based on this information, create a custom validator prompt that will assess how well the original task was performed. The prompt should ask 2 specific questions about the quality and completeness (i.e., precision and recall) of the output, such as:
        1. Recall-oriented; if the prompt asks for all instances of a target information, the validator prompt should ask if all instances were found?
        2. Would the output significantly improve if the input was analyzed more carefully?
        3. Is the output format correct and consistent?
        4. Are there any errors or inconsistencies in the extracted information?

        Provide your response as a single string containing the custom validator prompt. The prompt should be tailored to the task and avoid generic criteria.
        """

        parameters = {
            "type": "object",
            "properties": {"validator_prompt": {"type": "string"}},
            "required": ["validator_prompt"],
        }

        response = self.llm_client.generate_rewrite(
            [
                {"role": "user", "content": prompt},
            ],
            system_prompt,
            parameters,
        )
        return json.loads(response.choices[0].message.content)["validator_prompt"]

    def _get_header_extraction_prompt(
        self,
        op_config: dict[str, Any],
        input_data: list[dict[str, Any]],
        split_key: str,
    ) -> tuple[str, dict[str, Any]]:
        """
        Generate a header extraction prompt for a split operation. This prompt will be used to extract the headers from the input data in each chunk.

        Args:
            op_config (dict[str, Any]): The operation configuration.
            input_data (list[dict[str, Any]]): A list of input data samples.
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

        response = self.llm_client.generate_rewrite(
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

        {{{{ input.{split_key}_chunk }}}}

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
        op_config: dict[str, Any],
        assessment: dict[str, Any],
        input_data_sample: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
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

        response = self.llm_client.generate_rewrite(
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
        op_config: dict[str, Any],
        sample_output: list[dict[str, Any]],
    ) -> tuple[str, bool]:
        """
        Generate a combine prompt for merging chunk results in a map-reduce operation.

        This method creates a prompt that will be used to combine the results from
        processing individual chunks of data in a map-reduce operation. The combine
        prompt is designed to accomplish the original task by merging the outputs
        from various chunks.

        Args:
            op_config (dict[str, Any]): The configuration of the original operation,
                including the original prompt and output schema.
            sample_output (list[dict[str, Any]]): A list of sample outputs from
                processing various chunks. Each item in the list represents the
                output from a single chunk.

        Returns:
            tuple[str, bool]: A tuple containing:
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
        This prompt will be submitted to an LLM, so it must be a valid Jinja2 template, with natural language instructions.

        Guidelines for your prompt template:
        - The only variable you are allowed to use is the `inputs` variable, which contains all chunk results. Each value is a dictionary with the keys {', '.join(schema_keys)}
        - Avoid using filters or complex logic like `do` statements, even though Jinja technically supports it
        - The prompt template must be a valid Jinja2 template
        - You must use the {{{{ inputs }}}} variable somehow, in a for loop. You must access specific keys in each item in the loop.
        - The prompt template must also contain natural language instructions so the LLM knows what to do with the data

        Provide your prompt template as a single string.
        """
        # Add example for combining themes
        base_prompt += """
        Example of a good combine prompt for combining themes:
        ```
        You are tasked with combining themes extracted from different chunks of text.

        Here are the themes extracted from each chunk:
        {% for item in inputs %}
        Themes for chunk {loop.index}:
        {{ item.themes }}
        {% endfor %}

        Analyze all the themes above and create a consolidated list that:
        1. Combines similar or related themes
        2. Preserves unique themes that appear in only one chunk
        3. Prioritizes themes that appear multiple times across chunks
        4. Maintains the original wording where possible

        Provide the final consolidated list of themes, ensuring each theme is distinct and meaningful.
        ```

        Now generate a combine prompt for the current task.
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
            inclusion_strings=["inputs"],
        )
        combine_prompt = result["combine_prompt"]

        # Confirm with the user that this prompt is good & ask them to edit
        # if self.runner.status:
        #     self.runner.status.stop()

        # combine_prompt = Prompt.ask(
        #     f"Here is the prompt generated for the reduce operation:\n```\n{combine_prompt}\n```\n\nPress enter to confirm, or type in the prompt you would like to use instead.",
        #     default=combine_prompt,
        #     console=self.console,
        # )

        # if self.runner.status:
        #     self.runner.status.start()

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
        metadata_schema: dict[str, Any],
        sample_output: list[dict[str, Any]],
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

        response = self.llm_client.generate_rewrite(
            [
                {"role": "user", "content": prompt},
            ],
            system_prompt,
            parameters,
        )
        result = json.loads(response.choices[0].message.content)

        return result["edited_subprompt"]

    def _get_schema_transform_prompt(
        self,
        op_config: dict[str, Any],
        parallel_output_schema: dict[str, Any],
        target_schema: dict[str, Any],
        sample_output: list[dict[str, Any]],
    ) -> str:
        """
        Generate a prompt for transforming parallel map output into the target schema.

        Args:
            op_config: Original operation configuration
            parallel_output_schema: Schema produced by parallel map operations
            target_schema: Desired final output schema
            sample_output: Sample output from parallel map operations

        Returns:
            str: Prompt for the reduce operation that transforms schemas
        """
        system_prompt = (
            "You are an AI assistant tasked with transforming data between schemas."
        )

        # Filter sample output to only include parallel output schema keys
        filtered_sample = [
            {k: d[k] for k in parallel_output_schema.keys()} for d in sample_output[:2]
        ]  # Limit to 2 samples for brevity

        missing_keys = set(target_schema.keys()) - set(parallel_output_schema.keys())

        prompt = f"""Original task prompt that operated on the full input:
        {op_config['prompt']}

        Current schema from parallel operations:
        {json.dumps(parallel_output_schema, indent=2)}

        Target schema we need to transform to:
        {json.dumps(target_schema, indent=2)}

        Sample output from parallel operations/input to this transform operation:
        {json.dumps(filtered_sample, indent=2)}

        Keys that need to be synthesized: {list(missing_keys)}

        For example, in a legal document analysis task (where the document key is 'document'), the parallel operations may have extracted specific clauses into separate lists:

        Input schema:
        ```json
        {{
            "indemnification_clauses": ["Company A shall indemnify..."],
            "liability_clauses": ["Neither party shall be liable..."],
            "confidentiality_clauses": ["All information shared..."],
            "term_clauses": ["This agreement shall remain in effect..."]
        }}
        ```

        Target schema:
        ```json
        {{
            "clauses": [
                {{
                    "type": "indemnification",
                    "text": "Company A shall indemnify...",
                    "risk_level": "high"
                }},
                {{
                    "type": "liability",
                    "text": "Neither party shall be liable...",
                    "risk_level": "medium"
                }}
            ]
        }}
        ```

        Create a prompt that will transform the parallel operation output into the target schema.
        The prompt should:
        1. Use the existing data to synthesize the missing keys
        2. Maintain the values of keys that already exist
        3. Follow the original task's intent when synthesizing new values
        4. Include the original prompt as context, so the LLM has context on how to fill out the remaining keys if the parallel operations didn't cover them

        Guidelines:
        - The only variable you can use is 'input', which contains a single result from the parallel operations
        - To reference a key created by the parallel operations, use the 'input.key_name' syntax
        - Also reference the original document like the original prompt did, so the LLM has context on how to fill out the remaining keys if the parallel operations didn't cover them
        - The prompt must be a valid Jinja2 template
        - Explain how to use the existing data to inform the synthesis

        Example of a good transform prompt for the above example:
        ```
        [Insert original task prompt here]

        Here is the document we are working with:
        {{ {{ input.document }} }}

        Here are the extracted clauses:
        - Indemnification clauses: {{ {{ input.indemnification_clauses }} }}
        - Liability clauses: {{ {{ input.liability_clauses }} }}
        - Confidentiality clauses: {{ {{ input.confidentiality_clauses }} }}
        - Term clauses: {{ {{ input.term_clauses }} }}

        Format the output as a list of clause objects with type, text and risk_level fields.
        ```

        Provide your prompt template as a single string.
        """

        parameters = {
            "type": "object",
            "properties": {"transform_prompt": {"type": "string"}},
            "required": ["transform_prompt"],
        }

        result = generate_and_validate_prompt(
            self.llm_client,
            prompt,
            system_prompt,
            parameters,
            op_config,
            is_metadata=True,
            config=self.config,
            max_threads=self.max_threads,
            console=self.console,
            inclusion_strings=["input"],
        )

        return result["transform_prompt"]

    def _get_missing_keys_prompt(
        self,
        op_config: dict[str, Any],
        missing_keys: set[str],
        existing_keys: set[str],
    ) -> str:
        """
        Generate a prompt for synthesizing missing keys in chain plans.

        Args:
            op_config: Original operation configuration
            missing_keys: Set of keys that need to be synthesized
            existing_keys: Set of keys that were already generated by previous chain steps
            sample_output: Unused, kept for interface compatibility

        Returns:
            str: Prompt focused on generating just the missing keys
        """
        system_prompt = "You are an AI assistant tasked with completing missing fields in a data processing task."

        prompt = f"""Original task prompt:
        {op_config['prompt']}

        The chain steps so far will generate these keys:
        {json.dumps({k: op_config['output']['schema'][k] for k in existing_keys}, indent=2)}

        We need to generate these remaining keys:
        {json.dumps({k: op_config['output']['schema'][k] for k in missing_keys}, indent=2)}

        Create a prompt that focuses specifically on generating the missing keys.
        The prompt should:
        1. Use the context from the original task
        2. Reference the existing keys (using input.key_name syntax) to inform the generation of missing keys
        3. Only generate the missing keys, as the existing keys will be merged in later
        4. Maintain consistency with the already generated data

        Guidelines:
        - Reference existing keys using the 'input.key_name' syntax
        - Include any relevant parts of the original prompt that help with generating these specific keys
        - The prompt must be a valid Jinja2 template
        - Focus only on generating the missing keys: {', '.join(missing_keys)}

        Provide your prompt template as a single string.
        """

        parameters = {
            "type": "object",
            "properties": {"synthesis_prompt": {"type": "string"}},
            "required": ["synthesis_prompt"],
        }

        result = generate_and_validate_prompt(
            self.llm_client,
            prompt,
            system_prompt,
            parameters,
            op_config,
            is_metadata=True,
            config=self.config,
            max_threads=self.max_threads,
            console=self.console,
            inclusion_strings=["input"],
        )

        return result["synthesis_prompt"]
