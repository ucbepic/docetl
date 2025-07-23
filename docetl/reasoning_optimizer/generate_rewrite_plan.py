import argparse
import json
import os
import random
from typing import Any, Dict, List, Optional, Union

import litellm
from dotenv import load_dotenv
from pydantic import BaseModel
from pyexpat.errors import messages
from rich.console import Console

from docetl.utils import extract_jinja_variables, load_config

# Load environment variables from .env file
load_dotenv()


class ChunkSize(BaseModel):
    chunk_size: int
    reason: str


class SplitConfig(BaseModel):
    split_key: str
    subprompt: str
    subprompt_output_schema: str


class MetadataNeeds(BaseModel):
    needs_metadata: bool
    reason: str


class Generate_rewrite_plan:
    """ """

    def __init__(self, ai_response: Union[str, Dict[str, Any]], model="o3"):
        """
        Initialize with the AI response from optimize_plan.py.
        Parses the response to extract operations and saves the rest as records for history.
        Args:
            ai_response (str or dict): The AI response as a JSON string or dict.
        """
        if isinstance(ai_response, str):
            try:
                response = json.loads(ai_response)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON string provided: {e}")
        elif isinstance(ai_response, dict):
            response = ai_response
        else:
            raise TypeError("ai_response must be a JSON string or a dict")

        # Extract operations from 'stepwise_pipeline'
        self.operations: List[Dict[str, Any]] = response.get("stepwise_pipeline", [])

        # Save the rest of the response as records (excluding stepwise_pipeline)
        self.records: Dict[str, Any] = {
            k: v for k, v in response.items() if k != "stepwise_pipeline"
        }
        self.model = model
        self.console = Console()

    def get_plan_name(self) -> Optional[str]:
        """Return the plan name from the AI response, or None if not present."""
        return self.records.get("plan_name")

    def get_operations(self) -> List[Dict[str, Any]]:
        """Return the list of operations parsed from the AI response."""
        return self.operations

    def get_records(self) -> Dict[str, Any]:
        """Return the records/history part of the AI response (excluding operations)."""
        return self.records

    def choose_chunk_size(self, op: Dict[str, Any]) -> int:
        """
        Given a split operation description, generate an LLM prompt to ask for the chunk size.
        """

        prompt = f"""
        You have made the decision to use the split operation in the rewritten plan and your original thoughts on
        the description and parameter of it is here: {op}. Now, you need to generate the configuration for this split
        operation, which is the chunk size. Choose a chunk size that balances between context preservation and model performance.
        Smaller chunks may lose context, while larger chunks may degrade model performance. Output your choice of chunk size.
        """

        messages = [
            {
                "role": "system",
                "content": "You are an expert agent for document processing pipelines. Your role is to come up with the most suitable chunk size for the split opration. Your output must follow the structured output format.",
            },
            {"role": "user", "content": prompt},
        ]
        response = litellm.completion(
            model=self.model,  # Use a default model instead of self.model
            messages=messages,
            api_key=os.environ.get("AZURE_API_KEY"),
            api_base=os.environ.get("AZURE_API_BASE"),
            api_version=os.environ.get("AZURE_API_VERSION"),
            azure=True,
            response_format=ChunkSize,
        )
        content = response.choices[0].message.content
        parsed_content = json.loads(content)
        chunk_size = parsed_content.get("chunk_size")
        return int(chunk_size)

    def _get_split_config(
        self,
        op_config: Dict[str, Any],
        input_data_sample: List[Dict[str, Any]],
        random_sample: Dict[str, Any],
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
        The subprompt should be designed to process individual chunks of the split data, and only process the main chunk in within chunk delimiters if they are present.
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

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        response = litellm.completion(
            model=self.model,
            messages=messages,
            api_key=os.environ.get("AZURE_API_KEY"),
            api_base=os.environ.get("AZURE_API_BASE"),
            api_version=os.environ.get("AZURE_API_VERSION"),
            azure=True,
            response_format=SplitConfig,
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
        print("variables: ", variables_in_subprompt)
        # Replace variables in subprompt with f"input.{split_key}_chunk"
        for variable in variables_in_subprompt:
            inp_split_key = f"input.{result['split_key']}_chunk_rendered"
            result["subprompt"] = result["subprompt"].replace(
                f"{{{{ {variable} }}}}", f"{{{{ {inp_split_key} }}}}"
            )

        # # Fix output schema array keys to be list[string]
        # for key, value in result["subprompt_output_schema"].items():
        #     if value == "array" or value == "list":
        #         result["subprompt_output_schema"][key] = "list[string]"

        # result["subprompt_output_schema"].update(op_config["output"]["schema"])

        result["subprompt"] = (
            result["subprompt"]
            + " Only process the main chunk in --- Begin Main Chunk --- and --- End Main Chunk --- delimiters if they are present."
        )

        self.console.log(
            f"[yellow]Breaking down operation {op_config['name']}[/yellow]"
        )
        self.console.log(f"[cyan]Subprompt:[/cyan] {result['subprompt']}")
        self.console.log(
            f"[cyan]Subprompt Output Schema:[/cyan] {result['subprompt_output_schema']}"
        )

        return result

    def determine_metadata(
        self,
        op_config: Dict[str, Any],
        split_result: Dict[str, Any],
        chunk_size: int,
        input_data_sample: List[Dict[str, Any]],
    ):
        split_subprompt = split_result["subprompt"]
        split_key = split_result["split_key"]
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

    def prepare_gather(self, op: Dict[str, Any]):
        """
        Given a gather operation description, generate an LLM prompt to ask for the user_peripheral_config_tuple.
        """

        example_dict = {
            "previous": {
                "head": {"count": 1, "content_key": "full_content"},
                "middle": {"content_key": "summary_content"},
                "tail": {"count": 2, "content_key": "full_content"},
            },
            "next": {"head": {"count": 1, "content_key": "full_content"}},
        }

        prompt = f"""
        You have made the decision to use the gather operation in the rewritten plan and your original thoughts on the description and parameter of it is here: {op}. Now, you need to generate the configuration for this gather operation. This configuration should be a tuple:
        1. The first element "gather_config" is a configuration expressed in a dictionary describing how much context to include from surrounding chunks. The configuration is divided into two main sections:

        previous: Defines how chunks preceding the current chunk are included.
        next: Defines how chunks following the current chunk are included.
        Each of these sections can contain up to three subsections:

        head: The first chunk(s) in the section.
        middle: Chunks between the head and tail sections.
        tail: The last chunk(s) in the section.
        For each subsection, you can specify:

        count: The number of chunks to include (for head and tail only).
        content_key: The key in the chunk data that contains the content to use.
        An example dictionary is {example_dict}

        2. The second element "needs_summary" is a boolean indicating whether a summary is needed for the peripheral context (True if summaries should be included, False if only full content is needed).

        Best practices at making your design choice:
        - Use full content for immediate context (head/tail), and summaries for middle sections to save space.
        - You can use asymmetric configurations (e.g., more previous than next context).
        - Use 'content_key' to specify whether to include the full chunk or a summary.
        - Only include as much context as is necessary for the downstream map operation to work well.
        - If the task is highly local, you may set both previous and next to minimal values.

        """
        messages = [
            {
                "role": "system",
                "content": "You are an expert agent for document processing pipelines. Your role is to come up with the most suitable config for the gather opration. Your output must follow the structured output format.",
            },
            {"role": "user", "content": prompt},
        ]
        response = litellm.completion(
            model=self.model,  # Use a default model instead of self.model
            messages=messages,
            api_key=os.environ.get("AZURE_API_KEY"),
            api_base=os.environ.get("AZURE_API_BASE"),
            api_version=os.environ.get("AZURE_API_VERSION"),
            azure=True,
        )
        return response.choices[0].message.content


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("yaml_path", type=str, help="Path to the YAML file")
    args = parser.parse_args()

    # Sample AI response, should be the output of the first call to the reasoning model
    # Includes the chosen rewrite rule and a stepwise pipeline
    ai_response = """{
        "plan_name": "Chunk-Aware Contract Clause Extraction",
        "rewrites_used": ["Document Chunking Rewrite Directive"],
        "plan_description": "The original plan applies one large map operation to ~11K-token contracts, risking context overflow, hallucinations, and high token cost. By introducing chunk-based processing with contextual gathering and a per-document reduce, we fit within model limits, cut prompt+completion tokens per call, and preserve accuracy through context augmentation and deduplication.",
        "stepwise_pipeline": [
            {"operation": "split", "description": "Split each contract into non-overlapping chunks that comfortably fit the LLM context.", "parameter": ["chunk_size_tokens: 2000", "overlap_tokens: 0"]},
            {"operation": "gather", "description": "For every focal chunk, prepend summaries of the immediately preceding and following chunks to supply local context without blowing up the prompt length.", "parameter": ["neighbors_each_side: 1", "summary_tokens_per_neighbor: 200"]},
            {"operation": "map", "description": "Run the (slightly shortened) clause-extraction prompt on each context-augmented chunk, returning partial clause hits plus their text spans and in-document offsets.", "parameter": ["model: gemini/gemini-2.0-flash", "max_output_tokens: 600"]},
            {"operation": "reduce", "description": "Merge chunk-level results for the same document, deduplicate identical spans, and concatenate multiples within each clause field, yielding one consolidated record per contract.", "parameter": ["group_by: id", "deduplication_strategy: span_text_and_offsets"]}
        ],
        "reason": "Splitting contracts into ~2K-token pieces keeps prompts within context limits, lowering per-call latency and cost. Neighbor summaries supply just enough context to avoid losing cross-chunk clause information. Mapping at chunk level parallelizes work and reduces hallucination risk. The final reduce step reunifies partial results, ensuring correctness while preventing double charges for oversized prompts."
    }"""

    plan = Generate_rewrite_plan(ai_response)

    # load input_data_sample of CUAD
    data_dir = os.environ.get("EXPERIMENT_DATA_DIR", "./data/")
    with open(os.path.join(data_dir, "CUAD_input_data.json"), "r") as f:
        input_data_sample = json.load(f)
    with open(os.path.join(data_dir, "CUAD_random_sample.json"), "r") as f:
        random_sample = json.load(f)

    yaml_file = str(args.yaml_path)
    base_name = yaml_file.rsplit(".", 1)[0]
    suffix = yaml_file.split("/")[-1].split(".")[0]
    config = load_config(yaml_file)
    model = config.get("default_model")
    print(model)
    ops = config.get("operations")
    for op_config in ops:
        op_config["model"] = model
        result = plan._get_split_config(
            op_config,
            input_data_sample,
            random_sample,
        )

    print(result)
    return

    # For each op in the pipeline, prepare its config
    for op in plan.get_operations():
        op_type = op.get("operation")
        if op_type == "gather":
            print(op)
            # print("*******************")
            # output = plan.prepare_gather(op)
            # print(output)
        elif op_type == "split":
            print("*******************")
            output = plan.prepare_split_config(op, cuad_input_data_sample)
            print(output)


if __name__ == "__main__":
    main()
