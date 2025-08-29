import json
import os
import re
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


def extract_for_variable(template_content):
    """
    Extract variable name from for loop in template content
    Example: {% for item in inputs %} returns "item"
    """
    for_pattern = r"\{\%\s*for\s+(\w+)\s+in\s+\w+\s*\%\}"
    match = re.search(for_pattern, template_content)
    if match:
        return match.group(1)
    return None


def create_dynamic_pattern(new_key, template_content):
    """
    Create dynamic pattern based on for loop variable in template content
    """
    # Extract for loop variable
    loop_variable = extract_for_variable(template_content)

    if loop_variable:
        # Use loop variable instead of fixed "input"
        new_key_pattern = (
            r"\{\{\s*"
            + re.escape(loop_variable)
            + r"\."
            + re.escape(new_key)
            + r"\s*\}\}"
        )
    else:
        # If no for loop found, match new_key directly (without prefix)
        new_key_pattern = r"\{\{\s*" + re.escape(new_key) + r"\s*\}\}"

    return new_key_pattern, loop_variable


class MapOpConfig(BaseModel):
    """
    Configuration for a Map operator in a data processing chain.

    Attributes:
        name (str): The name of the Map operator.
        prompt (str): Jinja prompt template for the inserted Map operator. Must refer to the input document keys as {{ input.key }}.
        output_keys (List[str]): The keys of the output of the Map operator, to be referenced in the downstream operator's prompt.
        model (str): The model to use for the Map operator.
    """

    name: str = Field(..., description="The name of the Map operator")
    prompt: str = Field(
        ...,
        description="Jinja prompt template for the inserted Map operator. Must refer to the input document keys as {{ input.key }}.",
    )
    output_keys: List[str] = Field(
        ...,
        description="The keys of the output of the Map operator, to be referenced in the downstream operator's prompt. Can be a single key or a list of keys. Can be new keys or existing keys from the map operator we are rewriting.",
    )
    
    @classmethod
    def validate_prompt_contains_input_key(cls, value: str) -> str:
        """
        Validator to ensure the prompt contains at least one {{ input.key }} reference.

        Args:
            value (str): The prompt string.

        Returns:
            str: The validated prompt string.

        Raises:
            ValueError: If no {{ input.key }} pattern is found in the prompt.
        """
        # Matches {{ input.key }} for any key (non-whitespace, non-})
        if not re.search(r"\{\{\s*input\.[^}\s]+\s*\}\}", value):
            raise ValueError(
                "The prompt must contain at least one '{{ input.key }}' reference."
            )
        return value

    @field_validator("prompt")
    @classmethod
    def check_prompt(cls, v: str) -> str:
        return cls.validate_prompt_contains_input_key(v)


class ChainingInstantiateSchema(BaseModel):
    """
    Schema for chaining multiple Map operators in a data processing pipeline.

    Validates:
    - For each required input key, at least one MapOpConfig prompt must reference {{ input.key }}.
    - The output_keys of the final MapOpConfig must match the expected output_keys.
    """

    new_ops: List[MapOpConfig] = Field(
        ..., description="The new Map operators to insert in the chain."
    )

    @classmethod
    def validate_chain(
        cls,
        new_ops: List[MapOpConfig],
        required_input_keys: List[str],
        expected_output_keys: List[str],
    ) -> None:
        """
        Validates that for each required input key, at least one prompt in new_ops contains {{ input.key }},
        and that the final op's output_keys match expected_output_keys.

        Args:
            new_ops (List[MapOpConfig]): The list of MapOpConfig objects.
            required_input_keys (List[str]): The list of input keys that must be referenced in prompts.
            expected_output_keys (List[str]): The expected output keys for the final op.

        Raises:
            ValueError: If any required input key is not referenced in any prompt,
                        or if the final op's output_keys do not match expected_output_keys.
        """

        # Check each required input key is referenced in at least one prompt
        for key in required_input_keys:
            pattern = r"\{\{\s*input\." + re.escape(key) + r"\s*\}\}"
            if not any(re.search(pattern, op.prompt) for op in new_ops):
                raise ValueError(
                    f"Input key '{key}' must be referenced as '{{{{ input.{key} }}}}' in at least one prompt."
                )

        # Check that the final op's output_keys match expected_output_keys (order-insensitive)
        final_output_keys = list(new_ops[-1].output_keys) if new_ops else []
        if set(final_output_keys) != set(expected_output_keys):
            raise ValueError(
                f"The output_keys of the final op ({final_output_keys}) do not match the expected output_keys ({expected_output_keys})."
            )


class GleaningInstantiateSchema(BaseModel):
    """
    Schema for gleaning operations in a data processing pipeline.
    """

    validation_prompt: str = Field(
        ...,
        description="The prompt to evaluate and improve the output of the upstream operator.",
    )
    num_rounds: int = Field(
        ..., description="The maximum number of refinement iterations."
    )
    model: str = Field(default="gpt-4o-mini", description="The LLM model to use.")

    @field_validator("validation_prompt")
    @classmethod
    def check_no_jinja_variables(cls, v: str) -> str:
        """
        Validates that the validation_prompt contains no Jinja template variables.

        Args:
            v (str): The validation prompt string.

        Returns:
            str: The validated prompt string.

        Raises:
            ValueError: If Jinja template variables are found in the prompt.
        """
        # Check for Jinja variable patterns: {{ variable }}, {{ input.key }}, etc.
        jinja_patterns = [
            r"\{\{\s*[^}]*\}\}",  # {{ variable }}
            r"\{\%\s*[^%]*%\}",  # {% control_statement %}
        ]

        for pattern in jinja_patterns:
            if re.search(pattern, v):
                raise ValueError(
                    "The validation_prompt must not contain Jinja template variables. "
                    "Found pattern matching: " + pattern
                )
        return v


class ChangeModelInstantiateSchema(BaseModel):
    """
    Schema for changing model choice in a data processing pipeline.
    """

    model: str = Field(default="gpt-4o-mini", description="The new LLM model to use.")

    @classmethod
    def validate_diff_model_in_list(
        cls, orig_model, model: str, list_of_model: List[str]
    ) -> None:
        """
        Validates that the model is in the allowed list_of_model.

        Args:
            model (str): The model name to check.

        Raises:
            ValueError: If the model is not in the allowed list.
        """

        if model not in list_of_model:
            raise ValueError(
                f"Model '{model}' is not in the allowed list: {list_of_model}"
            )
        elif model == orig_model:
            raise ValueError(
                f"Model '{model}' is the same as the original model used: {orig_model}"
            )


class DocSummarizationInstantiateSchema(BaseModel):
    """
    Schema for document summarization operations in a data processing pipeline.
    """

    name: str = Field(..., description="The name of the Map summarization operator")
    document_key: str = Field(
        ...,
        description="The key in the input document that contains long content to be summarized",
    )
    prompt: str = Field(
        ...,
        description="Jinja prompt template for summarizing the document. Must reference {{ input.<document_key> }} and preserve information needed by downstream operators.",
    )
    model: str = Field(
        default="gpt-4o-mini", description="The model to use for summarization."
    )

    @field_validator("prompt")
    @classmethod
    def check_prompt_references_document_key(cls, v: str, info) -> str:
        # First check that it contains at least one input reference
        MapOpConfig.validate_prompt_contains_input_key(v)

        # Then check that it specifically references the document_key
        if hasattr(info, "data") and "document_key" in info.data:
            document_key = info.data["document_key"]
            pattern = r"\{\{\s*input\." + re.escape(document_key) + r"\s*\}\}"
            if not re.search(pattern, v):
                raise ValueError(
                    f"The prompt must reference the document_key as '{{{{ input.{document_key} }}}}'"
                )

        return v


class SubtaskConfig(BaseModel):
    """
    Configuration for a single subtask in the parallel map phase.
    """

    name: str = Field(..., description="The name of this subtask")
    prompt: str = Field(
        ...,
        description="Jinja template for this subtask. MUST use {{ input.ORIGINAL_KEY }} to reference the same input key as the original map operation. Example: 'Extract basic info from {{ input.document }}'",
    )
    output_keys: List[str] = Field(
        ..., description="The output keys this subtask generates"
    )

    @field_validator("prompt")
    @classmethod
    def check_prompt(cls, v: str) -> str:
        return MapOpConfig.validate_prompt_contains_input_key(v)


class IsolatingSubtasksInstantiateSchema(BaseModel):
    """
    Schema for isolating subtasks operations in a data processing pipeline.
    Rewrites a Map into Parallel Map -> Map pattern for better subtask isolation.
    """

    subtasks: List[SubtaskConfig] = Field(
        ..., description="List of subtasks for the parallel map"
    )
    aggregation_prompt: str = Field(
        default="",
        description="Jinja template to combine all subtask outputs into final result. MUST reference subtask outputs as {{ input.subtask_1_output }}, {{ input.subtask_2_output }}, etc. If empty, no aggregation step will be created. Example: 'Combine the basic info {{ input.subtask_1_output }} with details {{ input.subtask_2_output }}'",
    )

    @field_validator("aggregation_prompt")
    @classmethod
    def check_aggregation_prompt(cls, v: str) -> str:
        # Skip validation if empty (no aggregation needed)
        if not v.strip():
            return v
        return MapOpConfig.validate_prompt_contains_input_key(v)

    def validate_subtasks_coverage(self, original_output_keys: List[str]) -> None:
        """
        Validates that subtasks collectively cover all original output keys.
        """
        subtask_keys = set()
        for subtask in self.subtasks:
            subtask_keys.update(subtask.output_keys)

        original_keys_set = set(original_output_keys)
        if subtask_keys != original_keys_set:
            missing = original_keys_set - subtask_keys
            extra = subtask_keys - original_keys_set
            error_parts = []
            if missing:
                error_parts.append(f"Missing keys: {list(missing)}")
            if extra:
                error_parts.append(f"Extra keys: {list(extra)}")
            raise ValueError(
                f"Subtasks must cover exactly the original output keys. {'; '.join(error_parts)}"
            )

    def validate_aggregation_references_all_subtasks(self) -> None:
        """
        Validates that the aggregation prompt references outputs from all subtasks.
        Uses the actual output keys generated by the LLM for each subtask.
        Only validates if aggregation_prompt is not empty.
        """
        if not self.aggregation_prompt.strip():
            return  # Skip validation if no aggregation prompt

        missing_references = []
        for i, subtask in enumerate(self.subtasks):

            #  {{input.subtask_i_output}}
            pattern_subtask_output = (
                r"\{\{\s*input\.subtask_" + str(i + 1) + r"_output\s*\}\}"
            )
            has_subtask_output = bool(
                re.search(pattern_subtask_output, self.aggregation_prompt)
            )

            if has_subtask_output:
                continue
            subtask_has_reference = False

            for output_key in subtask.output_keys:
                pattern_output_key = (
                    r"\{\{\s*input\." + re.escape(output_key) + r"\s*\}\}"
                )

                pattern_subtask_output_key = (
                    r"\{\{\s*input\.subtask_"
                    + str(i + 1)
                    + r"_output\."
                    + re.escape(output_key)
                    + r"\s*\}\}"
                )

                if re.search(pattern_output_key, self.aggregation_prompt) or re.search(
                    pattern_subtask_output_key, self.aggregation_prompt
                ):
                    subtask_has_reference = True
                    break

            if not subtask_has_reference:
                missing_references.extend(subtask.output_keys)

        if missing_references:
            raise ValueError(
                f"Aggregation prompt must reference all subtask output keys. "
                f"Missing references: {missing_references}. "
                f"Expected patterns like: {{{{ input.{missing_references[0]} }}}}"
            )


class DocCompressionInstantiateSchema(BaseModel):
    """
    Schema for document compression operations in a data processing pipeline.
    Inserts an Extract operator before the target operation to compress long documents.
    """

    name: str = Field(..., description="The name of the Extract compression operator")
    document_key: str = Field(
        ...,
        description="The key in the input document that contains long content to be compressed",
    )
    prompt: str = Field(
        ...,
        description="Plain text instructions for what to extract from the document. NOT a Jinja template - the Extract operator will automatically assemble the document content.",
    )
    model: str = Field(
        default="gpt-4o-mini", description="The model to use for extraction."
    )

    def validate_document_keys_exists_in_input(self, input_file_path: str) -> None:
        """
        Validates that the split_key exists in the input JSON file items.

        Args:
            input_file_path (str): Path to the input JSON file

        Raises:
            ValueError: If document_keys is not found in any input items
        """

        if not os.path.exists(input_file_path):
            raise ValueError(f"Input file not found: {input_file_path}")

        try:
            with open(input_file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in input file: {e}")

        if not isinstance(data, list) or not data:
            raise ValueError("Input file must contain a non-empty list of items")

        # Check if document_keys exists in any of the input items
        available_keys = set()
        document_keys_found = False

        for item in data:
            if isinstance(item, dict):
                available_keys.update(item.keys())
                if self.document_key in available_keys:
                    document_keys_found = True

        if not document_keys_found:
            raise ValueError(
                f"document_keys '{self.document_key}' not found in any input items. "
                f"Available keys: {sorted(available_keys)}"
            )


class DeterministicDocCompressionInstantiateSchema(BaseModel):
    """
    Schema for deterministic document compression operations in a data processing pipeline.
    Inserts a Code Map operator before the target operation to compress long documents using deterministic logic.
    """

    name: str = Field(..., description="The name of the Code Map compression operator")
    code: str = Field(
        ...,
        description="Python code defining a 'transform' function that takes input_doc and returns a dictionary with compressed document field(s). Must include all necessary imports within the function.",
    )

    @field_validator("code")
    @classmethod
    def check_code_has_function(cls, v: str) -> str:
        if "def transform(" not in v:
            raise ValueError(
                "Code must define a function named 'transform' that takes input_doc as parameter"
            )
        if "return {" not in v and "return dict(" not in v:
            raise ValueError("Code must return a dictionary")
        return v

    def validate_code_returns_target_keys(self, target_ops_configs: List[Dict]) -> None:
        """
        Validates that the code returns dictionary keys that match document fields referenced in target operations.
        """
        import re

        # Extract all {{ input.key }} references from target operation prompts
        referenced_keys = set()
        for op_config in target_ops_configs:
            prompt = op_config.get("prompt", "")
            # Find all {{ input.key }} patterns
            matches = re.findall(r"\{\{\s*input\.([^}\s]+)\s*\}\}", prompt)
            referenced_keys.update(matches)

        if not referenced_keys:
            raise ValueError("No input document keys found in target operation prompts")

        # Check if the code appears to return the referenced keys
        # This is a basic check - we look for the keys in return statements
        for key in referenced_keys:
            if f"'{key}'" not in self.code and f'"{key}"' not in self.code:
                raise ValueError(
                    f"Code must return dictionary key '{key}' which is referenced in target operation prompts as '{{{{ input.{key} }}}}'"
                )

    def validate_against_target_ops(self, target_ops_configs: List[Dict]) -> None:
        """
        Validates that the configuration is appropriate for the target operations.
        """
        self.validate_code_returns_target_keys(target_ops_configs)


class OperatorFusionInstantiateSchema(BaseModel):
    """Schema for fusing two sequential operations."""

    fused_prompt: str = Field(
        ..., description="Combined prompt that performs both operations' tasks"
    )


class ChunkSubsectionConfig(BaseModel):
    """Configuration for head/tail subsections with count and optional content_key."""

    count: int = Field(..., description="Number of chunks to include")
    content_key: Optional[str] = Field(
        None, description="Key in chunk data containing content to use"
    )


class ChunkMiddleSubsectionConfig(BaseModel):
    """Configuration for middle subsection (no count field)."""

    content_key: Optional[str] = Field(
        None, description="Key in chunk data containing content to use"
    )


class PeripheralSectionConfig(BaseModel):
    """Configuration for previous/next sections in peripheral chunks."""

    head: Optional[ChunkSubsectionConfig] = None
    middle: Optional[ChunkMiddleSubsectionConfig] = None
    tail: Optional[ChunkSubsectionConfig] = None


class PeripheralChunksConfig(BaseModel):
    """Configuration for gather operation peripheral chunks."""

    previous: Optional[PeripheralSectionConfig] = None
    next: Optional[PeripheralSectionConfig] = None


class ChunkHeaderSummaryInstantiateSchema(BaseModel):
    """
    Schema for chunk header summary operations in a data processing pipeline.
    Transforms Split -> Gather => Split -> Map -> Gather pattern.
    """

    header_extraction_prompt: str = Field(
        ...,
        description="Jinja prompt template for extracting headers from each chunk. Must reference {{ input.<split_key>_chunk }} to access the chunk content. Should output headers with hierarchical level information.",
    )
    summary_prompt: str = Field(
        ...,
        description="Jinja prompt template for summarizing each chunk. Must reference {{ input.<split_key>_chunk }} to access the chunk content. Should output <split_key>_summary field with condensed chunk information.",
    )

    @field_validator("header_extraction_prompt")
    @classmethod
    def check_header_prompt_references_chunk(cls, v: str) -> str:
        # Check that it contains at least one input reference
        MapOpConfig.validate_prompt_contains_input_key(v)

        # Check that it references _chunk field
        if "_chunk" not in v:
            raise ValueError(
                "The header_extraction_prompt must reference the chunk content using '_chunk' suffix"
            )
        return v

    @field_validator("summary_prompt")
    @classmethod
    def check_summary_prompt_references_chunk(cls, v: str) -> str:
        # Check that it contains at least one input reference
        MapOpConfig.validate_prompt_contains_input_key(v)

        # Check that it references _chunk field
        if "_chunk" not in v:
            raise ValueError(
                "The summary_prompt must reference the chunk content using '_chunk' suffix"
            )
        return v



class SamplingConfig(BaseModel):
    """Configuration for optional sampling in document chunking."""

    method: str = Field(
        default="uniform",
        description="Sampling method to use (default: uniform for stratified sampling)",
    )
    samples: int = Field(
        ...,
        description="Number of chunks to sample (e.g., 1 for one chunk, 5 for five chunks)",
    )
    stratify_key: Optional[str] = Field(
        default=None,
        description="Optional key to stratify sampling by (in addition to split document ID)",
    )

    samples_per_group: Optional[bool] = Field(
        default=False,
        description="Whether to sample N items from each stratify group instead of dividing total samples across groups",
    )

    random_state: Optional[int] = Field(
        default=None,
        description="An integer to seed the random generator with",
    )

    method_kwargs: Optional[str] = Field(
        default=None,
        description="Additional parameters for the sampling method. Must be a valid JSON string.",
    )

    @field_validator("samples")
    @classmethod
    def validate_samples_count(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("samples must be a positive integer")
        return v


class DocumentChunkingInstantiateSchema(BaseModel):
    """
    Schema for document chunking operations in a data processing pipeline.
    Transforms Map => Split -> Gather -> [Sample] -> Map -> Reduce pattern.
    Sampling is applied by default unless the task requires processing all chunks.
    """

    chunk_size: int = Field(
        ..., description="Number of tokens per chunk for the split operation"
    )
    split_key: str = Field(
        ..., description="The key in the input document that contains the text to split"
    )
    sub_prompt: str = Field(
        ...,
        description="Jinja prompt template for the new map operation that processes chunks. Must reference {{ input.<split_key>_chunk_rendered }} to access the gathered chunk content.",
    )
    reduce_prompt: str = Field(
        ...,
        description="Jinja prompt template for the reduce operation that aggregates chunk results. Must use {% for input in inputs %} to iterate over chunk results and produce the same output as the original map operation.",
    )
    gather_config: PeripheralChunksConfig = Field(
        default_factory=lambda: PeripheralChunksConfig(
            previous=PeripheralSectionConfig(tail=ChunkSubsectionConfig(count=1))
        ),
        description="Configuration for the gather operation's peripheral_chunks. Specifies how much context to include from surrounding chunks. Default includes 1 previous chunk.",
    )
    sampling_config: Optional[SamplingConfig] = Field(
        default=None,
        description="Optional sampling configuration. If provided, inserts a Sample operation between Gather and Map. Use by default UNLESS task requires processing ALL chunks (like comprehensive extraction of all instances).",
    )
   
    def validate_split_key_exists_in_input(self, input_file_path: str) -> None:
        """
        Validates that the split_key exists in the input JSON file items.

        Args:
            input_file_path (str): Path to the input JSON file

        Raises:
            ValueError: If split_key is not found in any input items
        """

        if not os.path.exists(input_file_path):
            raise ValueError(f"Input file not found: {input_file_path}")

        try:
            with open(input_file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in input file: {e}")

        if not isinstance(data, list) or not data:
            raise ValueError("Input file must contain a non-empty list of items")

        # Check if split_key exists in any of the input items
        available_keys = set()
        split_key_found = False

        for item in data:
            if isinstance(item, dict):
                available_keys.update(item.keys())
                if self.split_key in item:
                    split_key_found = True

        if not split_key_found:
            raise ValueError(
                f"split_key '{self.split_key}' not found in any input items. "
                f"Available keys: {sorted(available_keys)}"
            )

    @field_validator("sub_prompt")
    @classmethod
    def check_sub_prompt_references_chunk_rendered(cls, v: str, info) -> str:
        # Check that it contains at least one input reference
        MapOpConfig.validate_prompt_contains_input_key(v)

        # Check that it references _chunk_rendered field
        if "_chunk_rendered" not in v:
            raise ValueError(
                "The sub_prompt must reference the rendered chunk content using '_chunk_rendered' suffix"
            )
        return v

    @field_validator("reduce_prompt")
    @classmethod
    def check_reduce_prompt_has_iteration(cls, v: str) -> str:
        # Check that it contains iteration pattern for reduce
        if "for input in inputs" not in v and "for item in inputs" not in v:
            raise ValueError(
                "The reduce_prompt must iterate over inputs using '{% for input in inputs %}' or '{% for item in inputs %}'"
            )
        return v

    @field_validator("gather_config")
    @classmethod
    def validate_gather_config(
        cls, v: PeripheralChunksConfig
    ) -> PeripheralChunksConfig:
        """
        Validates that the gather_config follows the correct structure for peripheral_chunks.
        """
        # The Pydantic model structure already enforces the basic validation
        # We can add additional business logic validation here if needed
        return v

    def validate_stratify_key_in_pipeline(
        self, pipeline_operations: List[Dict]
    ) -> None:
        """
        Validates that if sampling_config contains a stratify_key, it corresponds to
        a doc_id_key field mentioned somewhere in the pipeline operations.

        Args:
            pipeline_operations: List of operation configurations from the pipeline
        """
        if not self.sampling_config:
            return

        stratify_key = self.sampling_config.stratify_key
        if not stratify_key:
            return

        # Collect all doc_id_key values from pipeline operations
        doc_id_keys = set()
        for op in pipeline_operations:
            if "doc_id_key" in op:
                doc_id_keys.add(op["doc_id_key"])

        # Check if stratify_key is mentioned as a doc_id_key
        if stratify_key not in doc_id_keys:
            available_keys = sorted(doc_id_keys) if doc_id_keys else "None"
            raise ValueError(
                f"stratify_key '{stratify_key}' is not found as a doc_id_key in any pipeline operation. "
                f"Available doc_id_keys in pipeline: {available_keys}. "
                f"The stratify_key must correspond to a doc_id_key field that will be present in the data."
            )


class TopKConfig(BaseModel):
    """Configuration for topk operation in document chunking."""

    method: str = Field(
        ...,
        description="Method for topk selection: 'embedding' for semantic similarity or 'fts' for full-text search",
    )
    k: int = Field(
        ..., description="Number of chunks to retrieve (e.g., 10 for ten chunks)"
    )
    query: str = Field(
        ...,
        description="Query string for finding relevant chunks. For embedding: descriptive phrases. For fts: specific keywords.",
    )
    keys: List[str] = Field(
        ...,
        description="Keys to use for similarity matching, typically ['<split_key>_chunk']",
    )
    embedding_model: Optional[str] = Field(
        "text-embedding-3-small",
        description="Embedding model to use (only for embedding method)",
    )
    stratify_key: Optional[str] = Field(
        None, description="Optional key for stratified topk retrieval"
    )

    @field_validator("method")
    @classmethod
    def validate_method(cls, v: str) -> str:
        if v not in ["embedding", "fts"]:
            raise ValueError("method must be either 'embedding' or 'fts'")
        return v

    @field_validator("k")
    @classmethod
    def validate_k(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("k must be a positive integer")
        return v

    @field_validator("keys")
    @classmethod
    def validate_keys(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("keys cannot be empty")
        return v


class DocumentChunkingTopKInstantiateSchema(BaseModel):
    """
    Schema for document chunking with topk operations in a data processing pipeline.
    Transforms Map => Split -> TopK -> Reduce pattern.
    Uses topk to intelligently select relevant chunks based on a query.
    """

    chunk_size: int = Field(
        ..., description="Number of tokens per chunk for the split operation"
    )
    split_key: str = Field(
        ..., description="The key in the input document that contains the text to split"
    )
    reduce_prompt: str = Field(
        ...,
        description="Jinja prompt template for the reduce operation that processes selected chunks. Must use {% for input in inputs %} to iterate over chunks and extract/synthesize information to produce the same output as the original map operation.",
    )
    topk_config: TopKConfig = Field(
        ...,
        description="Configuration for the topk operation to select relevant chunks",
    )


    def validate_split_key_exists_in_input(self, input_file_path: str) -> None:
        """
        Validates that the split_key exists in the input JSON file items.

        Args:
            input_file_path (str): Path to the input JSON file

        Raises:
            ValueError: If split_key is not found in any input items
        """

        if not os.path.exists(input_file_path):
            raise ValueError(f"Input file not found: {input_file_path}")

        try:
            with open(input_file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in input file: {e}")

        if not isinstance(data, list) or not data:
            raise ValueError("Input file must contain a non-empty list of items")

        # Check if split_key exists in any of the input items
        available_keys = set()
        split_key_found = False

        for item in data:
            if isinstance(item, dict):
                available_keys.update(item.keys())
                if self.split_key in item:
                    split_key_found = True

        if not split_key_found:
            raise ValueError(
                f"split_key '{self.split_key}' not found in any input items. "
                f"Available keys: {sorted(available_keys)}"
            )

    @field_validator("reduce_prompt")
    @classmethod
    def check_reduce_prompt_has_iteration(cls, v: str) -> str:
        # Check that it contains iteration pattern for reduce
        if "for input in inputs" not in v and "for item in inputs" not in v:
            raise ValueError(
                "The reduce_prompt must iterate over inputs using '{% for input in inputs %}' or '{% for item in inputs %}'"
            )
        return v


class HierarchicalReduceInstantiateSchema(BaseModel):
    """
    Schema for hierarchical reduce operations in a data processing pipeline.
    Transforms Reduce => Reduce -> Reduce pattern where the first Reduce operation
    aggregates data at a finer granularity (reduce_key + additional_key), then
    the second Reduce combines these to the target granularity (reduce_key).
    Optionally includes a Map operation before the first Reduce to create synthetic keys.
    """

    map_config: Optional[MapOpConfig] = Field(
        None,
        description="Optional: Configuration for a Map operator to create synthetic keys for finer-grained aggregation",
    )
    additional_key: str = Field(
        ...,
        description="The additional key to use alongside the original reduce_key for finer granularity in the first reduce. Can be an existing key or the synthetic key created by the optional Map (use map_config.output_keys[0]).",
    )
    reduce_1_name: str = Field(
        ..., description="The name of the first Reduce operator (finer granularity)"
    )
    reduce_1_prompt: str = Field(
        ...,
        description="Jinja prompt template for the first Reduce that aggregates at finer granularity (reduce_key + additional_key). Should be adapted from the original reduce prompt.",
    )
    reduce_2_prompt: str = Field(
        ...,
        description="Jinja prompt template for the second Reduce that combines the outputs of the first reduce to the target granularity (reduce_key only). Should reference the output from the first reduce.",
    )

    @field_validator("reduce_1_prompt", "reduce_2_prompt")
    @classmethod
    def check_reduce_prompts(cls, v: str) -> str:
        # Check that it contains iteration pattern for reduce
        if "for input in inputs" not in v and "for item in inputs" not in v:
            raise ValueError(
                "The reduce prompts must iterate over inputs using '{% for input in inputs %}' or '{% for item in inputs %}'"
            )
        return v


class TakeHeadTailInstantiateSchema(BaseModel):
    """
    Schema for head/tail truncation operations in a data processing pipeline.
    Inserts a Code Map operator before the target operation to keep only first k and last l words.
    """

    name: str = Field(..., description="The name of the Code Map head/tail operator")
    document_key: str = Field(
        ...,
        description="The key in the input document that contains the longest text to be truncated",
    )
    head_words: int = Field(
        ...,
        description="Number of words to keep from the beginning of the document",
    )
    tail_words: int = Field(
        default=0,
        description="Number of words to keep from the end of the document. Default is 0.",
    )

    @field_validator("head_words")
    @classmethod
    def validate_head_words(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("head_words must be a positive integer")
        return v

    @field_validator("tail_words")
    @classmethod
    def validate_tail_words(cls, v: int) -> int:
        if v < 0:
            raise ValueError("tail_words must be a non-negative integer")
        return v


class ReduceChainingInstantiateSchema(BaseModel):
    """
    Schema for chaining a reduce operation with a map operation.
    Transforms Reduce => Map -> Reduce pattern where the Map operation
    preprocesses individual documents before the Reduce operation aggregates them.
    """

    map_name: str = Field(..., description="The name of the new Map operator")
    map_prompt: str = Field(
        ...,
        description="Jinja prompt template for the Map operator that processes individual documents. Must reference {{ input.document_key }} to access the document content.",
    )
    new_key: str = Field(
        ...,
        description="The new key name that the Map operation will output, which the Reduce operation will reference instead of the original document key.",
    )
    modified_reduce_prompt: str = Field(
        ...,
        description="The modified reduce prompt that references the new key ({{ input.new_key }}) instead of the original document content.",
    )

    @field_validator("map_prompt")
    @classmethod
    def check_map_prompt(cls, v: str) -> str:
        return MapOpConfig.validate_prompt_contains_input_key(v)

    @classmethod
    def validate_reduce_prompt_references_new_key(
        cls,
        modified_reduce_prompt: str,
        new_key: str,
        original_document_key: str,
    ) -> None:
        """
        Validates that the modified reduce prompt references the new key
        and doesn't reference the original document key.
        """
        # Check that it references the new key
        new_key_pattern, loop_variable = create_dynamic_pattern(
            new_key, modified_reduce_prompt
        )
        if not re.search(new_key_pattern, modified_reduce_prompt):
            raise ValueError("Modified reduce prompt must reference the new key")

        # Check that it doesn't reference the original document key
        old_key_pattern = (
            r"\{\{\s*input\." + re.escape(original_document_key) + r"\s*\}\}"
        )
        if re.search(old_key_pattern, modified_reduce_prompt):
            raise ValueError(
                "Modified reduce prompt should not reference the original document key '{{ input."
                + original_document_key
                + " }}'. Use '{{ input."
                + new_key
                + " }}' instead."
            )


class ClarifyInstructionsInstantiateSchema(BaseModel):
    """
    Schema for clarifying instructions in prompts using sample data.
    This directive rewrites a single operation's prompt by analyzing multiple samples
    from the input data to create a more specific and clear prompt.
    """

    clarified_prompt: str = Field(
        ...,
        description="The improved, more specific prompt based on analysis of sample data. Should be a Jinja template that references the same input fields as the original prompt.",
    )

    @classmethod
    def validate_input_variables_preserved(
        cls, clarified_prompt: str, original_prompt: str
    ) -> None:
        """
        Validates that all input variables from the original prompt are preserved in the clarified prompt.
        Only validates if the original prompt contains Jinja templates.
        """
        # Extract all {{ input.xxx }} patterns from original prompt
        original_vars = set(
            re.findall(r"\{\{\s*input\.([^}\s]+)\s*\}\}", original_prompt)
        )

        # If original prompt has no input variables, skip validation (e.g., extract, rank operators)
        if not original_vars:
            return

        # Extract all {{ input.xxx }} patterns from clarified prompt
        clarified_vars = set(
            re.findall(r"\{\{\s*input\.([^}\s]+)\s*\}\}", clarified_prompt)
        )

        # Check that all original variables are present in clarified prompt
        missing_vars = original_vars - clarified_vars
        if missing_vars:
            raise ValueError(
                f"Clarified prompt is missing input variables from original prompt: {sorted(missing_vars)}. "
                f"Original variables: {sorted(original_vars)}, Clarified variables: {sorted(clarified_vars)}"
            )


class SwapWithCodeInstantiateSchema(BaseModel):
    """
    Schema for swapping a Reduce operation with a Code Reduce + optional Map operation.
    Transforms Reduce => Code Reduce + Map pattern where the Code Reduce performs the core reduction logic
    and the optional Map operation converts the output to match the original reduce operation's schema.
    """

    code_reduce_name: str = Field(
        ..., description="The name of the new Code Reduce operator"
    )
    code: str = Field(
        ...,
        description="Python code defining a 'transform' function that takes a list of inputs and returns a dictionary with the reduced result fields. Must include all necessary imports within the function.",
    )
    map_prompt: Optional[str] = Field(
        default=None,
        description="Optional Jinja prompt template for the Map operator that converts the code reduce output to match the original reduce schema. Should reference {{ input.field_name }} to access fields returned by the code reduce operation. If None, no map operation will be added.",
    )

    @field_validator("code")
    @classmethod
    def check_code_has_function(cls, v: str) -> str:
        if "def transform(" not in v:
            raise ValueError(
                "Code must define a function named 'transform' that takes a list of inputs as parameter"
            )
        if "return {" not in v and "return dict(" not in v:
            raise ValueError("Code must return a dictionary")
        return v

    @field_validator("map_prompt")
    @classmethod
    def check_map_prompt_has_input_reference(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        if "{{ input." not in v:
            raise ValueError(
                "Map prompt must reference input fields using {{ input.field_name }} syntax"
            )
        return v


class CodePreFilter(BaseModel):
    """Configuration for a code-based pre-filter."""

    name: str = Field(..., description="Name for this filter operation")
    code: str = Field(
        ...,
        description="Python function def transform(input_doc): that returns True to keep, False to filter out",
    )
    reasoning: str = Field(
        ..., description="Explanation of what this filter checks and why it's effective"
    )


class LLMPreFilter(BaseModel):
    """Configuration for an LLM-based pre-filter."""

    name: str = Field(..., description="Name for this filter operation")
    prompt: str = Field(
        ...,
        description="Jinja2 prompt template that elicits a binary keep/drop decision. Must reference input fields using {{ input.fieldname }} syntax",
    )
    reasoning: str = Field(
        ..., description="Explanation of what this filter checks and why it's effective"
    )


class CascadeFilteringInstantiateSchema(BaseModel):
    """Schema for cascade filtering instantiation."""

    code_pre_filters: List[CodePreFilter] = Field(
        default_factory=list,
        description="List of code-based pre-filters to apply first (cheapest)",
    )

    llm_pre_filters: List[LLMPreFilter] = Field(
        default_factory=list,
        description="List of LLM-based pre-filters using gpt-5-nano (ordered by prompt length, shortest first)",
    )

    analysis_summary: str = Field(
        ...,
        description="Summary of patterns found in the data that informed the filter design",
    )

    @field_validator("llm_pre_filters")
    @classmethod
    def validate_llm_prompts(cls, v):
        """Validate that LLM prompts use proper Jinja2 syntax."""
        import re

        for filter_config in v:
            # Check for {{ input.something }} pattern
            if not re.search(r"\{\{\s*input\.\w+\s*\}\}", filter_config.prompt):
                raise ValueError(
                    f"LLM filter '{filter_config.name}' prompt must reference at least one input field using {{{{ input.fieldname }}}} syntax"
                )
        return v


class MapReduceFusionInstantiateSchema(BaseModel):
    """
    Schema for map-reduce fusion operations in a data processing pipeline.
    Transforms a Map -> Reduce pattern by having the Map pre-extract information
    that the Reduce operation needs, making the Reduce step more efficient.
    """

    new_map_name: str = Field(..., description="The name of the modified Map operator")
    new_map_prompt: str = Field(
        ..., description="Jinja template for the modified Map operator prompt"
    )
    new_key: str = Field(
        ...,
        description="The new key name that the Map operation will output, which the Reduce operation will reference instead of the original document key.",
    )
    new_reduce_prompt: str = Field(
        ..., description="Jinja template for the Reduce operator prompt"
    )

    @classmethod
    def validate_reduce_prompt_references_new_key(
        cls,
        new_reduce_prompt: str,
        new_key: str,
        original_document_key: str,
    ) -> None:
        """
        Validates that the modified reduce prompt references the new key
        and doesn't reference the original document key.
        """
        # Check that it references the new key
        new_key_pattern, loop_variable = create_dynamic_pattern(
            new_key, new_reduce_prompt
        )
        if not re.search(new_key_pattern, new_reduce_prompt):
            raise ValueError("Modified reduce prompt must reference the new key")

        # Check that it doesn't reference the original document key
        old_key_pattern = (
            r"\{\{\s*input\." + re.escape(original_document_key) + r"\s*\}\}"
        )
        if re.search(old_key_pattern, new_reduce_prompt):
            raise ValueError(
                "Modified reduce prompt should not reference the original document key '{{ input."
                + original_document_key
                + " }}'. Use '{{ input."
                + new_key
                + " }}' instead."
            )


class SearchReplaceEdit(BaseModel):
    """
    Represents a single search/replace edit to the pipeline operations JSON string.
    Works like a text editor's find-and-replace on the JSON representation.
    """

    search: str = Field(
        ...,
        description="The exact string to search for in the JSON representation of the pipeline. Must match exactly including whitespace.",
    )
    replace: str = Field(
        ...,
        description="The string to replace the search string with. Can be empty string to delete content.",
    )
    # reasoning: str = Field(
    #     ...,
    #     description="Explanation of why this edit improves the pipeline (cost, accuracy, or both)",
    # )


class ArbitraryRewriteInstantiateSchema(BaseModel):
    """
    Schema for arbitrary pipeline rewrites using search/replace edits.

    This directive allows the agent to make free-form edits to the pipeline
    by performing string search/replace operations on the JSON representation.
    The pipeline is converted to a JSON string, edits are applied, then parsed back.
    """

    search_replace_edits: List[SearchReplaceEdit] = Field(
        ...,
        description="List of search/replace edits to apply to the pipeline JSON string. Applied in order, each operating on the result of the previous edit.",
    )
    # overall_strategy: str = Field(
    #     ...,
    #     description="High-level explanation of the optimization strategy and expected improvements",
    # )
