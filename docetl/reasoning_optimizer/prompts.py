"""
Prompt Library for DocETL Operations and Rewrite Directives
"""

class PromptLibrary:
    
    @staticmethod
    def map_operator() -> str:
        prompt = """
        Map (LLM-powered):

        **Description**
        Processes each document independently by making an LLM call with your prompt template. Creates one output for each input document, with the output conforming to your specified schema.
        
        **When to Use**
        Use when you need to process each document individually - like extracting summarizing, classifying, or generating new fields based on document content.

        **Required Parameters**
        - name: Unique name for the operation
        - type: Must be "map"
        - model: LLM to use to execute the prompt
        - prompt: Jinja2 template with {{ input.key }} for each document field you want to reference
        - output.schema: Dictionary defining the structure and types of the output

        **Optional Parameters**
        - gleaning: Iteratively refines outputs that don't meet quality criteria. The LLM reviews its initial output and improves it based on validation feedback. Config requires:
            -- if: Python expression for when to refine; e.g., "len(output[key] == 0)" (optional)
            -- num_rounds: Max refinement iterations
            -- validation_prompt: What to improve
            -- model: LLM to use to execute the validation prompt and provide feedback (defaults to same model as operation model)
        (Default: gleaning is not enabled)

        - calibrate: (bool) Processes a sample of documents first to create reference examples, then uses those examples in all subsequent prompts to ensure consistent outputs across the dataset
        (Default: calibrate is False)
        - num_calibration_docs: Number of docs to use for calibration (default: 10)

        **Returns**
        Each original document, augmented with new keys specified in output_schema

        **Minimal Example Configuration**
        name: gen_insights
        type: map
        model: gpt-4o-mini
        prompt: From the user log below, list 2-3 concise insights (1-2 words each) and 1-2 supporting actions per insight. Return as a list of dictionaries with 'insight' and 'supporting_actions'. Log: {{ input.log }}
        output:
            schema:
                insights_summary: "string"
        """
        return prompt.strip()
    
    
    @staticmethod
    def reduce_operator() -> str:
        prompt = """
        Reduce:
        The reduce operation aggregates information across multiple documents based on a set of user-specified keys, 
        ultimately producing one output document per unique combination of attribute values. 
        If the given group of documents is too large for the LLM to correctly process, we apply a batched folding 
        approach that starts with an empty accumulator and sequentially folds in batches of more than one document at a time. 

        Required Parameters: \n
        - reduce_key: The key (or list of keys) to use for grouping data. Use _all to group all data into one group.
        """
        return prompt.strip()
    
    @staticmethod
    def resolve_operator() -> str:
        prompt = """
        Resolve:
        The resolve operation canonicalises one or more keys across documents that represent slight 
        variations of the same entity for subsequent grouping and aggregation. 

        Required Parameters: \n
        comparison_prompt: The prompt template to use for comparing potential matches.
        resolution_prompt: The prompt template to use for reducing matched entries.
        """
        return prompt.strip()
    
    @staticmethod
    def split_operator() -> str:
        prompt = """
        Split:
        The Split operation divides long text content into smaller, manageable chunks. Chunks will not overlap in content.
        There are two splitting methods. (1) Token Count Method: The token count method splits the text into chunks based on a specified number of tokens. 
        This is useful when you need to ensure that each chunk fits within the token limit of your language model, or you know that smaller chunks lead to higher performance.
        (2) Delimiter Method: The delimiter method splits the text based on a specified delimiter string. This is particularly useful when you want to split your text at logical boundaries, such as paragraphs or sections.\n

        Required Parameters: \n
        - split_key: a key in the input data that contains a string to be split. \n
        - method: The method to use for splitting. Options are "delimiter" and "token_count". \n
        - method_kwargs: A dictionary of keyword arguments for the splitting method. \n
            (For "delimiter" method: delimiter (string) to use for splitting.
            For "token_count" method: num_tokens (integer) specifying the maximum number of tokens per chunk.)
        """
        return prompt.strip()
    
    @staticmethod
    def gather_operator() -> str:
        example_dict = {
            "previous": {
                "head": {
                    "count": 1,
                    "content_key": "full_content"
                },
                "middle": {
                    "content_key": "summary_content"
                },
                "tail": {
                    "count": 2,
                    "content_key": "full_content"
                }
            },
            "next": {
                "head": {
                    "count": 1,
                    "content_key": "full_content"
                }
            }
        }
        prompt = f"""
        Gather:
        The gather operation complements the split operation by augmenting individual chunks with peripheral 
        information  necessary for understanding the chunk’s content.\n

        Required Parameters: \n
        - doc_id_key: Identifies chunks from the same original document.\n
        - order_key: Specifies the sequence of chunks within a group. \n
        - content_key: Indicates the field containing the chunk content. \n
        - doc_header_key (optional): Denotes a field representing extracted headers for each chunk. \n
        - peripheral_chunks: Specifies how to include context from surrounding chunks. \n
        peripheral_chunks is a configuration expressed in a dictionary describing how much context to include from surrounding chunks. The configuration is divided into two main sections:\n

        previous: Defines how chunks preceding the current chunk are included.\n
        next: Defines how chunks following the current chunk are included.\n
        Each of these sections can contain up to three subsections:\n

        head: The first chunk(s) in the section.\n
        middle: Chunks between the head and tail sections.\n
        tail: The last chunk(s) in the section.\n
        For each subsection, you can specify:\n

        count: The number of chunks to include (for head and tail only).\n
        content_key: The key in the chunk data that contains the content to use.\n
        An example peripheral_chunks dictionary is {example_dict}.\n
        """
        return prompt.strip()
    
    @staticmethod
    def filter_operator() -> str:
        prompt = """
        Filter:
        The filter operation selectively keeps or removes data items based on specified conditions. Unlike Map which transforms all items, 
        Filter evaluates each item against criteria and only passes through items that meet the conditions (evaluate to true), 
        creating a subset of the original dataset and thereby reducing data volume. 
        """
        return prompt.strip()
    
    @staticmethod
    def extract_operator() -> str:
        prompt = """
        Extract (LLM-powered):

        **Description**
        Pulls out specific portions of text exactly as they appear in the source document. The LLM identifies which parts to extract by providing line number ranges or regex patterns. Extracted text is saved to the original field name with "_extracted" suffix (e.g., report_text → report_text_extracted).

        **When to Use**
        Use when you need exact text from documents - like pulling out direct quotes, specific contract clauses, key findings, or any content that must be preserved word-for-word without LLM paraphrasing.

        **Required Parameters**
        - name: Unique name for the operation
        - type: Must be "extract"
        - prompt: Instructions for what to extract (this is NOT a Jinja template; we will run the prompt independently for each key in document_keys)
        - document_keys: List of document fields to extract from
        - model: LLM to use to execute the extraction

        **Optional Parameters**
        - extraction_method: How the LLM specifies what to extract from each document:
            -- "line_number" (default): The LLM outputs the line numbers or ranges in the document to extract. Use when relevant information is best identified by its position within each document.
            -- "regex": The LLM generates a custom regex pattern for each document to match the target text. Use when the information varies in location but follows identifiable text patterns (e.g., emails, dates, or structured phrases).

        **Returns**
        Each original document, augmented with {key}_extracted for each key in the specified document_keys

        **Minimal Example Configuration**
        name: findings
        type: extract
        prompt: Extract all sections that discuss key findings, results, or conclusions from this research report. Focus on paragraphs that:
        - Summarize experimental outcomes
        - Present statistical results
        - Describe discovered insights
        - State conclusions drawn from the research
        Only extract the most important and substantive findings.
        document_keys: ["report_text"]
        model: gpt-4o-mini
        """
        return prompt.strip()

    @staticmethod
    def parallel_map_operator() -> str:
        prompt = """
        Parallel Map (LLM-powered):
        **Description**
        Runs multiple independent map operations concurrently on each document. Each prompt generates specific fields, and all outputs are combined into a single result per input document. More efficient than sequential maps when transformations are independent.

        **When to Use**
        Use when you need multiple independent analyses of the same document - like extracting different types of information, running multiple classifications, or generating various summaries from the same input.

        **Required Parameters**
        - name: Unique name for the operation
        - type: Must be "parallel_map"
        - prompts: List of prompt configs, each with:
            -- prompt: Jinja2 template (should reference document fields using {{ input.key }})
            -- output_keys: List of fields this prompt generates
        - output.schema: Combined schema for all outputs. This must be the union of the output_keys from all prompts—every key in the output schema should be generated by at least one prompt, and no keys should be missing.
       
        **Optional Parameters**
        - model: Default LLM for all prompts
        Per-prompt options:
        - model: (optional) Override the default LLM for this specific prompt
        - gleaning: (optional) Validation and refinement configuration for this prompt, akin to gleaning in map operation

        **Returns**
        Each original document augmented with new keys specified in output_schema

        **Minimal Example Configuration**
        name: analyze_resume
        type: parallel_map
        prompts:
        - prompt: Extract skills from: {{ input.resume }}
            output_keys: [skills]
            model: gpt-4o-mini
        - prompt: Calculate years of experience from: {{ input.resume }}
            output_keys: [years_exp]
        - prompt: Rate writing quality 1-10: {{ input.cover_letter }}
            output_keys: [writing_score]
        output:
        schema:
            skills: list[string]
            years_exp: float
            writing_score: integer

        """
        return prompt.strip()

    # Below are the rewrite directives
    @staticmethod
    def document_chunking() -> str:
        prompt = """
        The Document Chunking Rewrite Directive:

        Use this directive when a map operation is applied to large documents that exceed LLM context windows or effective reasoning capabilities. 

        This directive rewrites a map operation into a pipeline: split -> gather -> map -> reduce. 

        Split: Split the document into a sequence of smaller chunks (without overlap) creating as many new docs as there are chunks. 
        You can suggest a reasonable chunk size for this splitting step so that each chunk can fit comfortably within the LLM’s context limit. 

        Gather: For each chunk, augment with relevant context from nearby chunks to help understand the chunk’s content. 
        You need to decide which surrounding chunks to include and how many. You have choices of inclusion of full chunks, 
        portions of chunks, or transformations (e.g.,  summaries) of chunks.

        Map: Apply a modified map operation per chunk.

        Reduce: Aggregate the individual chunk-level results into a single, coherent output that represents the analysis of 
        the entire original document.

        Example:
            Before:
                map: summarize the entire document
            After:
                split: split the document into contiguous chunks
                gather: gather k neighboring chunks before/after as context
                map: generate summarization per chunk
                reduce: global summarization across all chunks

        """
        return prompt.strip()
    
    @staticmethod
    def multi_level_agg() -> str:
        prompt = """
        The Multi-Level Aggregation Rewrite Directive:

        Use this directive when a reduce operation aggregates a large set of documents at once, or if aggregations
        might lose nuance when performed in a single step, use this directive.
        This directive rewrites a reduce operation into a pipeline: fine-grained-reduce → roll-up-reduce.

        Fine-Grained Reduce: Perform aggregation within smaller, logical subgroups.
        Roll-Up Reduce: Next, aggregate these subgroup results into the final desired output.

        Example:
        Before:
            reduce: summarizing voting patterns by state
        After:
            reduce1: aggregate data by state and city
            reduce2: combine these city-level summaries to the state level
        """
        return prompt.strip()

    @staticmethod
    def chaining() -> str:
        prompt = """
        The Chaining Rewrite Directive:

        Use this directive when a map operation contains multiple instructions. 
        This directive rewrites the original complex map operation to a chain of multiple simpler projections, each building on the previous result. 

        Example:
        Before:
            map: label sentiment and summarize by sentiment.
        After:
            map1: label sentiment
            map2: summarize all reviews of the same sentiment

        """
        return prompt.strip()
    

    @staticmethod
    def reordering() -> str:
        prompt = """
        The Reordering Rewrite Directive:

        Use this directive to reorder operations in a pipeline to reduce cost. Push down filter operation with the highest selectivity. 
        Push down map or extract as far as you can until you get to a task that needs the full document. 

        For example the pipeline “map -> filter -> reduce” can be reordered into “filter -> map -> reduce”
        """
        return prompt.strip()

    @staticmethod
    def metadata_extraction() -> str:
        prompt = """
        Metadata Extraction:

        This is essentially a map operation that extracts important metadata from the full document for accurate processing of document chunks. 
        It can be added in the pipeline before the split operation when there is potential presence of crucial information in metadata 
        (e.g., a table of contents, or a table of information in the beginning of the document) that might not be present in the chunks themselves.
        """
        return prompt.strip()
    
    @staticmethod
    def header_extraction() -> str:
        prompt = """
        Header Extraction:

        This is essentially a map operation that extracts headers from each chunk. It can be added in the pipeline between the split operation and the gather operation. 
        """
        return prompt.strip()




if __name__ == "__main__":
    
    print("\nExample map prompt:")
    print(PromptLibrary.map_operator())

