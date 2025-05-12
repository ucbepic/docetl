"""
The `ExtractOperation` class is a subclass of `BaseOperation` that performs extraction operations on document text fields.
This operation helps to identify and extract specific sections of text from documents based on provided criteria.
"""

from typing import Any, Dict, List, Literal, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

from jinja2 import Template
from pydantic import BaseModel, Field, field_validator

from docetl.operations.base import BaseOperation
from docetl.operations.utils import RichLoopBar, strict_render


class ExtractOperation(BaseOperation):
    class schema(BaseOperation.schema):
        type: str = "extract"
        prompt: str
        document_keys: List[str]
        model: Optional[str] = None
        format_extraction: bool = True
        extraction_key_suffix: Optional[str] = None
        extraction_method: Literal["line_number", "regex"] = "line_number"
        timeout: Optional[int] = None
        skip_on_error: bool = False
        litellm_completion_kwargs: Dict[str, Any] = Field(default_factory=dict)

        @field_validator("document_keys")
        def validate_document_keys(cls, v):
            if not isinstance(v, list):
                raise TypeError("document_keys must be a list")
            if not all(isinstance(key, str) for key in v):
                raise TypeError("All items in document_keys must be strings")
            if len(v) == 0:
                raise ValueError("document_keys list cannot be empty")
            return v

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # Set the extraction key suffix if not provided
        if not self.config.get("extraction_key_suffix"):
            self.extraction_key_suffix = f"_extracted_{self.config['name']}"
        else:
            self.extraction_key_suffix = self.config["extraction_key_suffix"]

    def syntax_check(self) -> None:
        """
        Checks the configuration of the ExtractOperation for required keys and valid structure.

        Raises:
            ValueError: If required keys are missing or invalid in the configuration.
            TypeError: If configuration values have incorrect types.
        """
        # The field_validator in the Pydantic schema handles basic type checks
        config = self.schema(**self.config)

        # Verify that prompt is a valid template
        try:
            Template(config.prompt)
        except Exception as e:
            raise ValueError(f"Invalid Jinja2 template in 'prompt': {str(e)}") from e

        # Check if model is specified (optional) and is a string
        if config.model and not isinstance(config.model, str):
            raise TypeError("'model' in configuration must be a string")

    def _reformat_text_with_line_numbers(self, text: str, line_width: int = 80) -> str:
        """
        Reformats text into lines of specified width and adds line numbers as prefixes.

        Args:
            text (str): The original text to reformat.
            line_width (int): The maximum width for each line. Defaults to 80.

        Returns:
            str: The reformatted text with line numbers.
        """
        if not text:
            return ""

        words = text.split()
        lines = []
        current_line = []
        current_length = 0

        for word in words:
            # Check if adding this word exceeds the line width
            if current_length + len(word) + (1 if current_line else 0) > line_width:
                # If current line is not empty, add it to lines
                if current_line:
                    lines.append(" ".join(current_line))
                    current_line = []
                    current_length = 0

                # If the word itself is longer than line_width, we need to split it
                if len(word) > line_width:
                    # Split the word into chunks of line_width or less
                    for i in range(0, len(word), line_width):
                        chunk = word[i:i + line_width]
                        lines.append(chunk)
                else:
                    current_line.append(word)
                    current_length = len(word)
            else:
                # Add word to current line
                if current_line:
                    current_length += 1 + len(word)  # +1 for the space
                else:
                    current_length = len(word)
                current_line.append(word)

        # Add any remaining words in the current line
        if current_line:
            lines.append(" ".join(current_line))

        # Add line numbers
        numbered_lines = [f"{i+1:4d}: {line}" for i, line in enumerate(lines)]
        return "\n".join(numbered_lines)

    def _execute_line_number_strategy(self, item: Dict, doc_key: str) -> Tuple[List[Dict[str, Any]], float]:
        """
        Executes the line number extraction strategy for a single document key.

        Args:
            item (Dict): The input document.
            doc_key (str): The key of the document text to process.

        Returns:
            Tuple[List[Dict[str, Any]], float]: A tuple containing the extraction results and the cost.
        """
        # Get the text content from the document
        if doc_key not in item or not isinstance(item[doc_key], str):
            if self.config.get("skip_on_error", True):
                self.console.log(f"[yellow]Warning: Key '{doc_key}' not found or not a string in document. Skipping.[/yellow]")
                return [], 0.0
            else:
                raise ValueError(f"Key '{doc_key}' not found or not a string in document")

        text_content = item[doc_key]
        
        # Reformat the text with line numbers
        formatted_text = self._reformat_text_with_line_numbers(text_content)
        
        # Render the prompt
        extraction_instructions = strict_render(self.config["prompt"], {"input": item})
        augmented_prompt_template = """
You are extracting specific content from text documents. Extract information according to these instructions: {{ extraction_instructions }}

The text is formatted with line numbers as follows:
{{ formatted_text }}

INSTRUCTIONS:
1. Analyze the text carefully and identify the exact line ranges that contain the requested information
2. Return ONLY line ranges as a JSON list of objects with 'start_line' and 'end_line' properties
3. Each range should be as precise as possible, including only relevant text
4. If multiple separate sections match the criteria, return multiple range objects
5. If no matching content is found, return an empty list

EXPECTED OUTPUT FORMAT:
{
  "line_ranges": [
    {"start_line": 12, "end_line": 15},
    {"start_line": 28, "end_line": 32}
  ]
}

Do not include explanatory text in your response, only the JSON object.
"""
        
        rendered_prompt = strict_render(augmented_prompt_template, {"extraction_instructions": extraction_instructions, "formatted_text": formatted_text})
        
        # Prepare messages for LLM
        messages = [{"role": "user", "content": rendered_prompt}]
        
        # Define the output schema for line number strategy
        line_number_schema = {
            "line_ranges": "list[{'start_line': int, 'end_line': int}]",
        }
        
        # Call the LLM
        llm_result = self.runner.api.call_llm(
            model=self.config.get("model", self.default_model),
            op_type="extract",
            messages=messages,
            output_schema=line_number_schema,
            timeout_seconds=self.config.get("timeout", 120),
            max_retries_per_timeout=self.config.get("max_retries_per_timeout", 2),
            bypass_cache=self.config.get("bypass_cache", False),
            litellm_completion_kwargs=self.config.get("litellm_completion_kwargs", {}),
            op_config=self.config,
        )
        
        # Parse the response
        try:
            parsed_output = self.runner.api.parse_llm_response(
                llm_result.response,
                schema=line_number_schema,
                manually_fix_errors=self.manually_fix_errors,
            )[0]
            
            # Extract the text based on line numbers
            extracted_texts = []
            lines = formatted_text.split('\n')
            
            if isinstance(parsed_output, str):
                self.console.log(f"[bold red]Error parsing LLM response: {llm_result.response}. Skipping.[/bold red]")
                return [], llm_result.total_cost
            
            for line_range in parsed_output.get("line_ranges", []):
                start_line = line_range.get("start_line", 0)
                end_line = line_range.get("end_line", 0)
                
                # Validate line numbers
                if start_line < 1 or end_line < start_line or end_line > len(lines):
                    if self.config.get("skip_on_error", True):
                        self.console.log(f"[yellow]Warning: Invalid line numbers {start_line}-{end_line} for document with {len(lines)} lines. Skipping this extraction.[/yellow]")
                        continue
                    else:
                        start_line = max(1, min(start_line, len(lines)))
                        end_line = max(start_line, min(end_line, len(lines)))
                
                # Extract the actual text from the specified lines without line numbers
                extracted_content = []
                for i in range(start_line - 1, end_line):
                    line = lines[i]
                    # Remove the line number prefix (e.g., "123: ")
                    if ': ' in line:
                        line = line.split(': ', 1)[1]
                    extracted_content.append(line)
                
                extracted_texts.append("".join(extracted_content))
            
            return extracted_texts, llm_result.total_cost
        
        except Exception as e:
            if self.config.get("skip_on_error", True):
                self.console.log(f"[bold red]Error parsing LLM response: {str(e)}. Skipping.[/bold red]")
                return [], llm_result.total_cost
            else:
                raise RuntimeError(f"Error parsing LLM response: {str(e)}") from e

    def _execute_regex_strategy(self, item: Dict, doc_key: str) -> Tuple[List[str], float]:
        """
        Executes the regex extraction strategy for a single document key.

        Args:
            item (Dict): The input document.
            doc_key (str): The key of the document text to process.

        Returns:
            Tuple[List[str], float]: A tuple containing the extraction results and the cost.
        """
        import re
        
        # Get the text content from the document
        if doc_key not in item or not isinstance(item[doc_key], str):
            if self.config.get("skip_on_error", True):
                self.console.log(f"[yellow]Warning: Key '{doc_key}' not found or not a string in document. Skipping.[/yellow]")
                return [], 0.0
            else:
                raise ValueError(f"Key '{doc_key}' not found or not a string in document")

        text_content = item[doc_key]
        
        # Prepare the context for prompt rendering
        context = {
            "input": item,
            "text_content": text_content
        }
        
        # Render the prompt
        extraction_instructions = strict_render(self.config["prompt"], context)
        augmented_prompt_template = """
You are creating regex patterns to extract specific content from text. Extract information according to these instructions: {{ extraction_instructions }}

The text to analyze is:
{{ text_content }}

INSTRUCTIONS:
1. Create precise regex patterns that will extract ONLY the requested information
2. Return a JSON object with a list of regex patterns
3. Each pattern should:
   - Use proper regex syntax compatible with Python's re module
   - Include capture groups to isolate the exact text needed
   - Handle potential variations in formatting
   - Account for multi-line content where appropriate using (?s) or other flags
4. Test your patterns mentally against the sample text to verify they work

EXPECTED OUTPUT FORMAT:
{
  "patterns": [
    "pattern1",
    "pattern2"
  ]
}

EXAMPLE REGEX PATTERNS:
- Simple extraction: "Name: ([\\w\\s]+)"
- With flags for multiline: "(?s)Abstract:\\s*(.+?)\\n\\n"
- For structured data: "<title>(.*?)</title>"

Return only the JSON object with your patterns, no explanatory text.
"""
        
        rendered_prompt = strict_render(augmented_prompt_template, {"extraction_instructions": extraction_instructions, "text_content": text_content})
        # Prepare messages for LLM
        messages = [{"role": "user", "content": rendered_prompt}]
        
        # Define the output schema for regex strategy
        regex_schema = {
            "patterns": "list[string]"
        }
        
        # Call the LLM
        llm_result = self.runner.api.call_llm(
            model=self.config.get("model", self.default_model),
            op_type="extract",
            messages=messages,
            output_schema=regex_schema,
            timeout_seconds=self.config.get("timeout", 120),
            max_retries_per_timeout=self.config.get("max_retries_per_timeout", 2),
            bypass_cache=self.config.get("bypass_cache", False),
            litellm_completion_kwargs=self.config.get("litellm_completion_kwargs", {}),
            op_config=self.config,
        )
        
        # Parse the response
        try:
            parsed_output = self.runner.api.parse_llm_response(
                llm_result.response,
                schema=regex_schema,
                manually_fix_errors=self.manually_fix_errors,
            )[0]
            
            patterns = parsed_output.get("patterns", [])
            extracted_texts = []
            
            for pattern in patterns:
                try:
                    matches = re.findall(pattern, text_content, re.DOTALL)
                    extracted_texts.extend(matches)
                except re.error as e:
                    if self.config.get("skip_on_error", True):
                        self.console.log(f"[yellow]Warning: Invalid regex pattern '{pattern}': {str(e)}. Skipping.[/yellow]")
                    else:
                        raise ValueError(f"Invalid regex pattern '{pattern}': {str(e)}")
            
            return extracted_texts, llm_result.total_cost
        
        except Exception as e:
            if self.config.get("skip_on_error", True):
                self.console.log(f"[bold red]Error parsing LLM response: {str(e)}. Skipping.[/bold red]")
                return [], llm_result.total_cost
            else:
                raise RuntimeError(f"Error parsing LLM response: {str(e)}") from e

    def execute(self, input_data: List[Dict]) -> Tuple[List[Dict], float]:
        """
        Execute the extraction operation on the input data.

        Args:
            input_data (List[Dict]): List of input data items.

        Returns:
            Tuple[List[Dict], float]: A tuple containing the processed data and the total cost of the operation.
        """
        if not input_data:
            return [], 0.0

        results = []
        total_cost = 0.0
        extraction_method = self.config.get("extraction_method", "line_number")
        
        # Process each document
        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            futures = []
            
            for item in input_data:
                output_item = item.copy()
                
                # Process each document key in the item
                for doc_key in self.config["document_keys"]:
                    if doc_key not in item or not isinstance(item[doc_key], str):
                        if self.config.get("skip_on_error", True):
                            self.console.log(f"[yellow]Warning: Key '{doc_key}' not found or not a string in document. Skipping.[/yellow]")
                            continue
                        else:
                            raise ValueError(f"Key '{doc_key}' not found or not a string in document")
                    
                    # Submit the appropriate extraction strategy based on config
                    if extraction_method == "line_number":
                        future = executor.submit(self._execute_line_number_strategy, item, doc_key)
                    elif extraction_method == "regex":
                        future = executor.submit(self._execute_regex_strategy, item, doc_key)
                    else:
                        raise ValueError(f"Unsupported extraction method: {extraction_method}")
                    
                    futures.append((doc_key, future, output_item))
            
            # Process results as they complete
            pbar = RichLoopBar(
                range(len(futures)),
                desc=f"Processing {self.config['name']} (extract) on all documents",
                console=self.console,
            )
            
            for i in pbar:
                doc_key, future, output_item = futures[i]
                
                try:
                    extracted_texts_duped, cost = future.result()
                    
                    # Remove duplicates and empty strings
                    extracted_texts_duped = [text for text in extracted_texts_duped if text]
                    extracted_texts = []
                    for text in extracted_texts_duped:
                        if text not in extracted_texts:
                            extracted_texts.append(text)
                    
                    total_cost += cost
                    
                    # Generate the output key name
                    output_key = f"{doc_key}{self.extraction_key_suffix}"
                    
                    # Format the extraction based on the config
                    if self.config.get("format_extraction", True):
                        output_item[output_key] = "\n\n".join(extracted_texts)
                    else:
                        output_item[output_key] = extracted_texts
                    
                except Exception as e:
                    if self.config.get("skip_on_error", True):
                        self.console.log(f"[bold red]Error in extraction for document key '{doc_key}': {str(e)}. Skipping.[/bold red]")
                        # Set empty result
                        output_key = f"{doc_key}{self.extraction_key_suffix}"
                        output_item[output_key] = "" if self.config.get("format_extraction", True) else []
                    else:
                        raise RuntimeError(f"Error in extraction for document key '{doc_key}': {str(e)}") from e
                
                # Add to results if this is the last doc_key for this item
                if i == len(futures) - 1 or futures[i+1][2] is not output_item:
                    results.append(output_item)
        
        return results, total_cost 