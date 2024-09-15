import copy
import json
import random
from typing import Any, Dict, List, Tuple

from rich.console import Console

from docetl.optimizers.map_optimizer.utils import generate_and_validate_prompt
from docetl.optimizers.utils import LLMClient, extract_jinja_variables


class ConfigGenerator:
    def __init__(
        self,
        llm_client: LLMClient,
        console: Console,
        config: Dict[str, Any],
        max_threads: int,
    ):
        self.llm_client = llm_client
        self.console = console
        self.config = config
        self.max_threads = max_threads

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
            - In the subprompt, we've replace all variables of 'input.{split_key}' with 'input.{split_key}_chunk'.
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
        Current Output Schema: {json.dumps(output_schema, indent=2)}
        
        Input keys: {input_data_sample[0].keys()}

        Input Data Sample:
        {json.dumps(random_sample, indent=2)[:5000]}

        Determine the split key and subprompt for processing chunks of the input data.
        The split key should be a key in the input data that contains a string to be split.
        The subprompt should be designed to process individual chunks of the split data.
        Note that the subprompt's output schema might be different from the original operation's output schema, since you may want to extract more information or make the information less structured/more free text. The original output schema will be preserved when combining the chunks' processed results.

        Important:
        - The subprompt should be a Jinja template.
        - The subprompt should use the variable 'input.{{ split_key }}_chunk_rendered' instead of 'input.{{ split_key }}'.

        Provide your response in the following format:
        - split_key: The key in the input data to be used for splitting
        - subprompt: The Jinja template prompt to be applied to each chunk
        - subprompt_output_schema: The output schema for the subprompt
        """

        parameters = {
            "type": "object",
            "properties": {
                "split_key": {"type": "string"},
                "subprompt": {"type": "string"},
                "subprompt_output_schema": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": {
                        "type": "string",
                        "enum": ["string", "integer", "number", "boolean", "array"],
                    },
                },
            },
            "required": ["split_key", "subprompt", "subprompt_output_schema"],
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

        # Strip out "input." from split_key if it exists
        result["split_key"] = result["split_key"].replace("input.", "")

        # Validate that the split_key exists in the input data sample
        if result["split_key"] not in input_data_sample[0]:
            raise ValueError(
                f"Split key '{result['split_key']}' not found in the input data sample."
            )

        variables_in_subprompt = extract_jinja_variables(result["subprompt"])
        # Replace variables in subprompt with f"input.{split_key}_chunk"
        for variable in variables_in_subprompt:
            inp_split_key = f"input.{result['split_key']}_chunk_rendered"
            result["subprompt"] = result["subprompt"].replace(
                f"{{{{ {variable} }}}}", f"{{{{ {inp_split_key} }}}}"
            )

        # Fix output schema array keys to be list[string]
        for key, value in result["subprompt_output_schema"].items():
            if value == "array" or value == "list":
                result["subprompt_output_schema"][key] = "list[string]"

        result["subprompt_output_schema"].update(op_config["output"]["schema"])

        self.console.log(
            f"[yellow]Breaking down operation {op_config['name']}[/yellow]"
        )
        self.console.log(f"[cyan]Subprompt:[/cyan] {result['subprompt']}")
        self.console.log(
            f"[cyan]Subprompt Output Schema:[/cyan] {result['subprompt_output_schema']}"
        )

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
        {json.dumps(random.choice(input_data_sample), indent=2)[:1000]}

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

        metadata_var = "input." + split_key  # noqa: F841

        base_prompt = f"""
        Given the following subtask prompt:
        {subprompt}

        And a chunk size of {chunk_size} words, create a prompt to extract metadata from each document/input.

        Full input sample:
        {random_sample}

        Provide a prompt to extract this metadata from each document/input. The extracted metadata should be a string, and your prompt should be a Jinja template that is only allowed to reference the variable `{metadata_var}` and nothing else.

        Provide your response in the following format:
        """

        parameters = {
            "type": "object",
            "properties": {
                "metadata_prompt": {"type": "string"},
            },
            "required": ["metadata_prompt"],
            "additionalProperties": False,
        }

        result = generate_and_validate_prompt(
            self.llm_client,
            base_prompt,
            system_prompt,
            parameters,
            op_config,
            is_metadata=True,
            config=self.config,
            max_threads=self.max_threads,
            console=self.console,
        )
        result["output_schema"] = {"metadata": "str"}

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

    def _generate_chunk_sizes(
        self,
        split_key: str,
        input_data_sample: List[Dict[str, Any]],
        token_limit: int,
        num_chunks: int = 8,
    ) -> List[int]:
        # Get the average document length
        avg_doc_length = sum(
            len(doc[split_key].split()) for doc in input_data_sample
        ) / len(input_data_sample)

        # Calculate the word limit based on the token limit
        word_limit = min(int(token_limit * 0.75), int(avg_doc_length))

        # Create chunk sizes based on word_limit
        min_chunk_size_word_limit = max(20, int(0.15 * word_limit))
        word_limit_chunks = [
            int(
                min_chunk_size_word_limit
                + i * (word_limit - min_chunk_size_word_limit) / (num_chunks // 2 - 1)
            )
            for i in range(num_chunks // 2)
        ]

        # Create chunk sizes based on avg_doc_length
        min_chunk_size_doc_length = max(20, int(0.15 * avg_doc_length))
        doc_length_chunks = [
            min(
                int(
                    min_chunk_size_doc_length
                    + i
                    * (avg_doc_length - min_chunk_size_doc_length)
                    / (num_chunks // 2 - 1)
                ),
                word_limit,
            )
            for i in range(num_chunks // 2)
        ]

        # Combine both lists and remove duplicates
        all_chunks = sorted(set(word_limit_chunks + doc_length_chunks))

        return all_chunks

    def _generate_peripheral_configs(
        self, summary_key: str, chunk_size: int, avg_doc_size: int
    ) -> List[Tuple[Dict[str, Any], bool]]:
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

        final_configs = configs + base_configs + scaled_configs

        # Add a configuration with more extensive context if the chunk size is small relative to the document size
        if chunk_size < avg_doc_size / 10:
            extensive_config = {
                "previous": {"tail": {"count": min(5, max_chunks)}},
                "next": {"head": {"count": min(2, max_chunks)}},
            }
            final_configs.append(extensive_config)

        # Add false to each config because there's no summarization needed
        final_configs = [(config, False) for config in final_configs]

        # Add configurations with summary for small-ish chunk sizes
        if chunk_size < avg_doc_size / 5:
            summary_configs = []
            for config in final_configs:
                summary_config = copy.deepcopy(config)[0]
                if "previous" not in summary_config:
                    summary_config["previous"] = {
                        "tail": {"count": 1, "content_key": summary_key}
                    }
                summary_config["previous"]["middle"] = {"content_key": summary_key}
                summary_configs.append((summary_config, True))
            final_configs.extend(summary_configs)

        # Deduplicate configs
        unique_configs = []
        for config in final_configs:
            if config not in unique_configs:
                unique_configs.append(config)
        final_configs = unique_configs

        return final_configs
