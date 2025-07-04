import ast
import asyncio
import hashlib
import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Union
from enum import Enum

from litellm import (
    APIConnectionError,
    ModelResponse,
    RateLimitError,
    ServiceUnavailableError,
    completion,
    embedding,
)
from litellm.types.utils import ChatCompletionMessageToolCall, Function
from rich import print as rprint
from rich.console import Console, Group
from rich.panel import Panel
from rich.text import Text

from docetl.utils import completion_cost

from .cache import cache, cache_key, freezeargs
from .llm import (
    InvalidOutputError,
    LLMResult,
    approx_count_tokens,
    timeout,
    truncate_messages,
)
from .validation import (
    convert_dict_schema_to_list_schema,
    convert_val,
    get_user_input_for_schema,
    safe_eval,
    strict_render,
)

BASIC_MODELS = ["gpt-4o-mini", "gpt-4o"]


class OutputMode(Enum):
    """Enumeration of output modes for LLM calls."""
    TOOLS = "tools"
    STRUCTURED_OUTPUT = "structured_output"


def is_deepseek_r1(model: str) -> bool:
    model = model.lower()
    return "deepseek-r1" in model or "deepseek-reasoner" in model


def is_snowflake(model: str) -> bool:
    model = model.lower()
    return "snowflake" in model


class OutputSchemaBuilder:
    """Handles building output schemas for both tools and structured output."""
    
    @staticmethod
    def build_tool_schema(output_schema: Dict[str, Any], scratchpad: Optional[str] = None, model: str = "") -> Dict[str, Any]:
        """Build a tool schema from an output schema."""
        props = {key: convert_val(value) for key, value in output_schema.items()}
        
        if scratchpad is not None:
            props["updated_scratchpad"] = {"type": "string"}

        parameters = {"type": "object", "properties": props}
        parameters["required"] = list(props.keys())

        # Some models don't support additionalProperties
        if "gemini" not in model and "claude" not in model:
            parameters["additionalProperties"] = False

        return parameters
    
    @staticmethod
    def build_structured_output_schema(output_schema: Dict[str, Any], scratchpad: Optional[str] = None) -> Dict[str, Any]:
        """Build a structured output schema from an output schema."""
        props = {key: convert_val(value) for key, value in output_schema.items()}
        
        if scratchpad is not None:
            props["updated_scratchpad"] = {"type": "string"}

        schema = {
            "type": "object",
            "properties": props,
            "required": list(props.keys()),
            "additionalProperties": False
        }
        
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "structured_output", 
                "schema": schema,
                "strict": True
            }
        }


class ResponseParser:
    """Handles parsing responses from different output modes."""
    
    def __init__(self, console: Console):
        self.console = console
    
    def parse_response(
        self, 
        response: Any, 
        schema: Dict[str, Any], 
        output_mode: OutputMode,
        tools: Optional[List[Dict[str, str]]] = None,
        manually_fix_errors: bool = False
    ) -> List[Dict[str, Any]]:
        """Parse response based on the output mode."""
        try:
            if not response:
                raise InvalidOutputError("No response from LLM", "{}", schema, [], [])

            results = []
            for index in range(len(response.choices)):
                if output_mode == OutputMode.STRUCTURED_OUTPUT:
                    results.extend(self._parse_structured_output(response, schema, index))
                else:  # OutputMode.TOOLS
                    results.extend(self._parse_tool_response(response, schema, tools, index))
            
            return results
            
        except InvalidOutputError as e:
            if manually_fix_errors:
                rprint(f"[bold red]Could not parse LLM output:[/bold red] {e.message}\n"
                       f"\tExpected Schema: {e.expected_schema}\n"
                       f"\tPlease manually set this output.")
                rprint(f"\n[bold yellow]LLM-Generated Response:[/bold yellow]\n{response}")
                output = get_user_input_for_schema(schema)
                return [output]
            else:
                raise e
    
    def _parse_structured_output(self, response: Any, schema: Dict[str, Any], index: int = 0) -> List[Dict[str, Any]]:
        """Parse structured output response."""
        try:
            content = response.choices[index].message.content
            
            # Handle deepseek-r1 models' think tags
            if is_deepseek_r1(response.model):
                result = {}
                think_match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
                if think_match:
                    result["think"] = think_match.group(1).strip()
                    # Get the remaining content after </think>
                    main_content = re.split(r"</think>", content, maxsplit=1)[-1].strip()
                    parsed_content = json.loads(main_content)
                else:
                    # If no think tags, parse the content as JSON
                    parsed_content = json.loads(content)
                
                result.update(parsed_content)
                return [result]
            
            # For other models, parse as JSON
            parsed_output = json.loads(content)
            
            # Augment with missing schema keys
            for key in schema:
                if key not in parsed_output:
                    parsed_output[key] = "Not found"
            
            return [parsed_output]
            
        except json.JSONDecodeError:
            raise InvalidOutputError(
                "Could not decode structured output JSON response",
                str(content),
                schema,
                response.choices,
                []
            )
        except Exception as e:
            raise InvalidOutputError(
                f"Error parsing structured output: {e}",
                str(content),
                schema,
                response.choices,
                []
            )
    
    def _parse_tool_response(self, response: Any, schema: Dict[str, Any], tools: Optional[List[Dict[str, str]]], index: int = 0) -> List[Dict[str, Any]]:
        """Parse tool-based response."""
        # Handle single-key string schema without tools
        if not tools and len(schema) == 1:
            key = next(iter(schema))
            content = response.choices[index].message.content
            
            # Handle deepseek-r1 models' think tags
            if is_deepseek_r1(response.model):
                result = {}
                think_match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
                if think_match:
                    result["think"] = think_match.group(1).strip()
                    main_content = re.split(r"</think>", content, maxsplit=1)[-1].strip()
                    result[key] = main_content
                else:
                    result[key] = content
                return [result]
            
            return [{key: content}]

        # Extract tool calls
        if is_snowflake(response.model):
            tool_calls = self._extract_snowflake_tool_calls(response, index)
        else:
            tool_calls = getattr(response.choices[index].message, 'tool_calls', []) or []

        if tools:
            return self._parse_custom_tools(tool_calls, tools)
        else:
            return self._parse_send_output_tools(tool_calls, schema, response)
    
    def _extract_snowflake_tool_calls(self, response: Any, index: int) -> List[ChatCompletionMessageToolCall]:
        """Extract tool calls from Snowflake model response."""
        if not hasattr(response.choices[index].message, "content_list"):
            return []
        
        return [
            ChatCompletionMessageToolCall(
                function=Function(
                    name=content.get("tool_use", {}).get("name"),
                    arguments=content.get("tool_use", {}).get("input"),
                )
            )
            for content in response.choices[index].message.content_list
            if content.get("type") == "tool_use"
        ]
    
    def _parse_custom_tools(self, tool_calls: List[ChatCompletionMessageToolCall], tools: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Parse responses from custom tools."""
        results = []
        for tool_call in tool_calls:
            for tool in tools:
                if tool_call.function.name == tool["function"]["name"]:
                    try:
                        function_args = (
                            json.loads(tool_call.function.arguments)
                            if isinstance(tool_call.function.arguments, str)
                            else tool_call.function.arguments
                        )
                    except json.JSONDecodeError:
                        return [{}]
                    
                    # Execute the function defined in the tool's code
                    local_scope = {}
                    exec(tool["code"].strip(), globals(), local_scope)
                    function_result = local_scope[tool["function"]["name"]](**function_args)
                    function_args.update(function_result)
                    results.append(function_args)
        return results
    
    def _parse_send_output_tools(self, tool_calls: List[ChatCompletionMessageToolCall], schema: Dict[str, Any], response: Any) -> List[Dict[str, Any]]:
        """Parse responses from send_output tool calls."""
        if not tool_calls:
            raise InvalidOutputError(
                "No tool calls in LLM response", "{}", schema, response.choices, []
            )

        outputs = []
        for tool_call in tool_calls:
            if hasattr(response.choices[0], 'finish_reason') and response.choices[0].finish_reason == "content_filter":
                raise InvalidOutputError(
                    "Content filter triggered by LLM provider.",
                    "",
                    schema,
                    response.choices,
                    []
                )
            
            try:
                output_dict = (
                    json.loads(tool_call.function.arguments)
                    if isinstance(tool_call.function.arguments, str)
                    else tool_call.function.arguments
                )
                
                # Augment output_dict with empty values for any keys in the schema that are not in output_dict
                for key in schema:
                    if key not in output_dict:
                        output_dict[key] = "Not found"

                # Handle specific model quirks for parsing
                if "ollama" in response.model or "sagemaker" in response.model:
                    self._handle_model_specific_parsing(output_dict)
                
                outputs.append(output_dict)
                
            except json.JSONDecodeError:
                raise InvalidOutputError(
                    "Could not decode LLM JSON response",
                    str(tool_call.function.arguments),
                    schema,
                    response.choices,
                    []
                )
            except Exception as e:
                raise InvalidOutputError(
                    f"Error parsing LLM response: {e}",
                    str(tool_call.function.arguments),
                    schema,
                    response.choices,
                    []
                )

        return outputs
    
    def _handle_model_specific_parsing(self, output_dict: Dict[str, Any]) -> None:
        """Handle specific parsing for certain models."""
        for key, value in output_dict.items():
            if not isinstance(value, str):
                continue
            try:
                output_dict[key] = ast.literal_eval(value)
            except Exception:
                try:
                    if value.startswith("["):
                        output_dict[key] = ast.literal_eval(value + "]")
                    else:
                        output_dict[key] = value
                except Exception:
                    pass


class LLMCallHandler:
    """Handles the core LLM call logic."""
    
    def __init__(self, runner, console: Console):
        self.runner = runner
        self.console = console
        self.default_lm_api_base = runner.config.get("default_lm_api_base", None)
    
    def make_completion_call(
        self,
        model: str,
        messages: List[Dict[str, str]],
        output_mode: OutputMode,
        output_schema: Dict[str, Any],
        tools: Optional[str] = None,
        scratchpad: Optional[str] = None,
        litellm_completion_kwargs: Dict[str, Any] = {},
        op_config: Dict[str, Any] = {},
    ) -> Any:
        """Make the actual completion call to the LLM."""
        # Build the system prompt
        system_prompt = self._build_system_prompt(model, output_mode, scratchpad)
        
        # Truncate messages if they exceed the model's context length
        messages_with_system_prompt = truncate_messages(
            [{"role": "system", "content": system_prompt}] + messages,
            model,
        )

        # Acquire resources
        self.runner.blocking_acquire("llm_call", weight=1)
        approx_num_tokens = approx_count_tokens(messages)
        self.runner.blocking_acquire("llm_tokens", weight=approx_num_tokens)
        
        if self.runner.is_cancelled:
            raise asyncio.CancelledError("Operation was cancelled")

        # Prepare completion kwargs
        extra_kwargs = self._prepare_completion_kwargs(litellm_completion_kwargs, op_config, model)
        
        # Make the call based on output mode
        if output_mode == OutputMode.STRUCTURED_OUTPUT:
            return self._make_structured_output_call(
                model, messages_with_system_prompt, output_schema, scratchpad, extra_kwargs
            )
        else:  # OutputMode.TOOLS
            return self._make_tool_call(
                model, messages_with_system_prompt, output_schema, tools, scratchpad, extra_kwargs
            )
    
    def _build_system_prompt(self, model: str, output_mode: OutputMode, scratchpad: Optional[str]) -> str:
        """Build the system prompt based on model and output mode."""
        persona = self.runner.config.get("system_prompt", {}).get("persona", "a helpful assistant")
        dataset_description = self.runner.config.get("system_prompt", {}).get(
            "dataset_description", "a collection of unstructured documents"
        )
        
        base_prompt = (
            f"You are a {persona}, helping the user make sense of their data. "
            f"The dataset description is: {dataset_description}. "
            "You will perform the specified task on the provided data, as precisely and exhaustively "
            "(i.e., high recall) as possible."
        )

        if output_mode == OutputMode.STRUCTURED_OUTPUT or "sagemaker" in model or is_deepseek_r1(model):
            system_prompt = base_prompt
        else:
            system_prompt = (
                base_prompt +
                " The result should be a structured output that you will send back to the user, "
                "with the `send_output` function. Do not influence your answers too much based on the "
                "`send_output` function parameter names; just use them to send the result back to the user."
            )

        if scratchpad:
            system_prompt += self._build_scratchpad_instructions()

        return system_prompt
    
    def _build_scratchpad_instructions(self) -> str:
        """Build instructions for scratchpad usage."""
        return """

You are incrementally processing data across multiple batches. You will see:
1. The current batch of data to process
2. The intermediate output so far (what you returned last time)
3. A scratchpad for tracking additional state

IMPORTANT: Only use the scratchpad if your task specifically requires tracking items that appear multiple times across batches. If you only need to track distinct/unique items, leave the scratchpad empty and set updated_scratchpad to null.

The intermediate output contains the result that directly answers the user's task, for **all** the data processed so far, including the current batch. You must return this via the send_output function.

Example task that NEEDS scratchpad - counting words that appear >2 times:
- Call send_output with: {"frequent_words": ["the", "and"]} # Words seen 3+ times - this is your actual result
- Set updated_scratchpad to: {"pending": {"cat": 2, "dog": 1}} # Must track words seen 1-2 times

Example task that does NOT need scratchpad - collecting unique locations:
- Call send_output with: {"locations": ["New York", "Paris"]} # Just the unique items
- Set updated_scratchpad to: null # No need to track counts since we only want distinct items

As you process each batch:
1. Use both the previous output and scratchpad (if needed) to inform your processing
2. Call send_output with your result that combines the current batch with previous output
3. Set updated_scratchpad only if you need to track counts/frequencies between batches

If you use the scratchpad, keep it concise (~500 chars) and easily parsable using:
- Key-value pairs
- JSON-like format
- Simple counters/tallies

Your main result must be sent via send_output. The updated_scratchpad is only for tracking state between batches, and should be null unless you specifically need to track frequencies."""
    
    def _prepare_completion_kwargs(self, litellm_completion_kwargs: Dict[str, Any], op_config: Dict[str, Any], model: str) -> Dict[str, Any]:
        """Prepare the completion kwargs."""
        extra_kwargs = {}
        extra_kwargs.update(litellm_completion_kwargs)
        
        if "n" in op_config.get("output", {}).keys():
            extra_kwargs["n"] = op_config["output"]["n"]
        
        if is_snowflake(model):
            extra_kwargs["allowed_openai_params"] = ["tools", "tool_choice"]
        
        if self.default_lm_api_base:
            extra_kwargs["api_base"] = self.default_lm_api_base
            
        return extra_kwargs
    
    def _make_structured_output_call(
        self, 
        model: str, 
        messages: List[Dict[str, str]], 
        output_schema: Dict[str, Any], 
        scratchpad: Optional[str],
        extra_kwargs: Dict[str, Any]
    ) -> Any:
        """Make a structured output call."""
        schema = OutputSchemaBuilder.build_structured_output_schema(output_schema, scratchpad)
        
        try:
            return completion(
                model=model,
                messages=messages,
                response_format=schema,
                **extra_kwargs,
            )
        except Exception as e:
            self._handle_model_error(model, e)
    
    def _make_tool_call(
        self, 
        model: str, 
        messages: List[Dict[str, str]], 
        output_schema: Dict[str, Any], 
        tools: Optional[str],
        scratchpad: Optional[str],
        extra_kwargs: Dict[str, Any]
    ) -> Any:
        """Make a tool-based call."""
        # Determine if we should use tools
        props = {key: convert_val(value) for key, value in output_schema.items()}
        use_tools = not (
            len(props) == 1
            and list(props.values())[0].get("type") == "string"
            and scratchpad is None
            and ("sagemaker" in model or is_deepseek_r1(model))
        )

        if tools is None and use_tools:
            tools_config, tool_choice = self._build_send_output_tool(output_schema, scratchpad, model)
        elif tools is not None:
            tools_config, tool_choice = self._build_custom_tools(tools)
        else:
            tools_config, tool_choice = None, None

        try:
            if tools_config is not None:
                return completion(
                    model=model,
                    messages=messages,
                    tools=tools_config,
                    tool_choice=tool_choice,
                    **extra_kwargs,
                )
            else:
                return completion(
                    model=model,
                    messages=messages,
                    **extra_kwargs,
                )
        except Exception as e:
            self._handle_model_error(model, e)
    
    def _build_send_output_tool(self, output_schema: Dict[str, Any], scratchpad: Optional[str], model: str) -> tuple:
        """Build the send_output tool configuration."""
        parameters = OutputSchemaBuilder.build_tool_schema(output_schema, scratchpad, model)
        
        if is_snowflake(model):
            tools = [
                {
                    "tool_spec": {
                        "type": "generic",
                        "name": "send_output",
                        "description": "Send output back to the user",
                        "input_schema": parameters,
                    }
                }
            ]
        else:
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "send_output",
                        "description": "Send output back to the user",
                        "parameters": parameters,
                    },
                }
            ]
            
        if "claude" not in model:
            tools[0]["additionalProperties"] = False
            tools[0]["strict"] = True

        tool_choice = {"type": "function", "function": {"name": "send_output"}}
        
        return tools, tool_choice
    
    def _build_custom_tools(self, tools: str) -> tuple:
        """Build custom tools configuration."""
        tools_list = json.loads(tools)
        tool_choice = "required" if any(tool.get("required", False) for tool in tools_list) else "auto"
        tools_config = [
            {"type": "function", "function": tool["function"]} for tool in tools_list
        ]
        return tools_config, tool_choice
    
    def _handle_model_error(self, model: str, error: Exception) -> None:
        """Handle model-specific errors."""
        if model not in BASIC_MODELS and "/" not in model:
            raise ValueError(
                f"Note: You may also need to prefix your model name with the provider, "
                f"e.g. 'openai/gpt-4o-mini' or 'gemini/gemini-1.5-flash' to conform to "
                f"LiteLLM API standards. Original error: {error}"
            )
        raise error


class ValidationHandler:
    """Handles validation and gleaning logic."""
    
    def __init__(self, runner, console: Console, llm_handler: LLMCallHandler, parser: ResponseParser):
        self.runner = runner
        self.console = console
        self.llm_handler = llm_handler
        self.parser = parser
    
    def handle_validation(
        self,
        response: Any,
        output_schema: Dict[str, Any],
        output_mode: OutputMode,
        validation_config: Optional[Dict[str, Any]],
        gleaning_config: Optional[Dict[str, Any]],
        model: str,
        op_type: str,
        messages: List[Dict[str, str]],
        tools: Optional[str] = None,
        scratchpad: Optional[str] = None,
        litellm_completion_kwargs: Dict[str, Any] = {},
        op_config: Dict[str, Any] = {},
        verbose: bool = False,
    ) -> tuple[Any, float, bool]:
        """Handle validation and gleaning processes."""
        total_cost = completion_cost(response)
        
        if gleaning_config:
            response, additional_cost, validated = self._handle_gleaning(
                response, output_schema, output_mode, gleaning_config, model, op_type, 
                messages, tools, scratchpad, litellm_completion_kwargs, op_config, verbose
            )
            total_cost += additional_cost
        elif validation_config:
            response, additional_cost, validated = self._handle_validation_retries(
                response, output_schema, output_mode, validation_config, model, op_type,
                messages, tools, scratchpad, litellm_completion_kwargs, op_config
            )
            total_cost += additional_cost
        else:
            validated = True
            
        return response, total_cost, validated
    
    def _handle_gleaning(
        self, 
        response: Any, 
        output_schema: Dict[str, Any],
        output_mode: OutputMode,
        gleaning_config: Dict[str, Any],
        model: str,
        op_type: str,
        messages: List[Dict[str, str]],
        tools: Optional[str],
        scratchpad: Optional[str],
        litellm_completion_kwargs: Dict[str, Any],
        op_config: Dict[str, Any],
        verbose: bool
    ) -> tuple[Any, float, bool]:
        """Handle gleaning process."""
        additional_cost = 0.0
        num_gleaning_rounds = gleaning_config.get("num_rounds", 2)
        
        parsed_output = (
            self.parser.parse_response(response, output_schema, output_mode, json.loads(tools) if tools else None)[0]
            if isinstance(response, ModelResponse)
            else response
        )

        validator_messages = (
            [
                {
                    "role": "system",
                    "content": f"You are a helpful assistant, intelligently processing data. This is a {op_type} operation.",
                }
            ]
            + messages
            + [{"role": "assistant", "content": json.dumps(parsed_output)}]
        )

        for rnd in range(num_gleaning_rounds):
            # Break early if gleaning condition is not met
            if not self.should_glean(gleaning_config, parsed_output):
                break
                
            # Prepare validator prompt
            validator_prompt = strict_render(
                gleaning_config["validation_prompt"],
                {"output": parsed_output},
            )
            
            self.runner.blocking_acquire("llm_call", weight=1)
            approx_num_tokens = approx_count_tokens(
                validator_messages + [{"role": "user", "content": validator_prompt}]
            )
            self.runner.blocking_acquire("llm_tokens", weight=approx_num_tokens)

            # Build validator tool
            should_refine_params = {
                "type": "object",
                "properties": {
                    "should_refine": {"type": "boolean"},
                    "improvements": {"type": "string"},
                },
                "required": ["should_refine", "improvements"],
            }
            if "gemini" not in model:
                should_refine_params["additionalProperties"] = False

            # Prepare extra kwargs
            extra_kwargs = {}
            if self.llm_handler.default_lm_api_base:
                extra_kwargs["api_base"] = self.llm_handler.default_lm_api_base
            if is_snowflake(model):
                extra_kwargs["allowed_openai_params"] = ["tools", "tool_choice"]

            validator_response = completion(
                model=gleaning_config.get("model", model),
                messages=truncate_messages(
                    validator_messages + [{"role": "user", "content": validator_prompt}],
                    model,
                ),
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "should_refine_answer",
                            "description": "Determine if the output should be refined based on the validation feedback",
                            "strict": True,
                            "parameters": should_refine_params,
                            "additionalProperties": False,
                        },
                    }
                ],
                tool_choice="required",
                **litellm_completion_kwargs,
                **extra_kwargs,
            )
            additional_cost += completion_cost(validator_response)

            # Parse the validator response
            suggestion = json.loads(
                validator_response.choices[0].message.tool_calls[0].function.arguments
            )
            if not suggestion["should_refine"]:
                break

            if verbose:
                self.console.log(
                    f"Validator improvements (gleaning round {rnd + 1}): {suggestion['improvements']}"
                )

            # Prompt for improvement
            improvement_prompt = f"""Based on the validation feedback:

            ```
            {suggestion['improvements']}
            ```

            Please improve your previous response. Ensure that the output adheres to the required schema and addresses any issues raised in the validation."""
            messages.append({"role": "user", "content": improvement_prompt})

            # Call LLM again
            response = self.llm_handler.make_completion_call(
                model, op_type, messages, output_mode, output_schema, tools, scratchpad, litellm_completion_kwargs, op_config
            )
            parsed_output = self.parser.parse_response(response, output_schema, output_mode, json.loads(tools) if tools else None)[0]
            validator_messages[-1] = {
                "role": "assistant",
                "content": json.dumps(parsed_output),
            }

            additional_cost += completion_cost(response)

        return response, additional_cost, True
    
    def _handle_validation_retries(
        self,
        response: Any,
        output_schema: Dict[str, Any],
        output_mode: OutputMode,
        validation_config: Dict[str, Any],
        model: str,
        op_type: str,
        messages: List[Dict[str, str]],
        tools: Optional[str],
        scratchpad: Optional[str],
        litellm_completion_kwargs: Dict[str, Any],
        op_config: Dict[str, Any],
    ) -> tuple[Any, float, bool]:
        """Handle validation retries."""
        additional_cost = 0.0
        num_tries = validation_config.get("num_retries", 2) + 1
        validation_fn = validation_config.get("validation_fn")
        val_rule = validation_config.get("val_rule")

        # Try validation
        i = 0
        validation_result = False
        while not validation_result and i < num_tries:
            parsed_output, validation_result = validation_fn(response)
            if validation_result:
                return response, additional_cost, True

            # Append the validation result to messages
            messages.append({"role": "assistant", "content": json.dumps(parsed_output)})
            messages.append({
                "role": "user",
                "content": f"Your output {parsed_output} failed my validation rule: {str(val_rule)}\n\nPlease try again.",
            })
            
            self.console.log(
                f"[bold red]Validation failed:[/bold red] {val_rule}\n"
                f"\t[yellow]Output:[/yellow] {parsed_output}\n"
                f"\t({i + 1}/{num_tries})"
            )
            i += 1

            response = self.llm_handler.make_completion_call(
                model, op_type, messages, output_mode, output_schema, tools, scratchpad, litellm_completion_kwargs, op_config
            )
            additional_cost += completion_cost(response)

        return response, additional_cost, validation_result
    
    def should_glean(self, gleaning_config: Optional[Dict[str, Any]], output: Dict[str, Any]) -> bool:
        """Determine whether to execute a gleaning round based on an optional conditional expression."""
        if not gleaning_config or "if" not in gleaning_config:
            return True

        condition = gleaning_config.get("if")
        if not isinstance(condition, str):
            raise ValueError(f"Invalid gleaning condition (should be a string): {condition}")

        try:
            return safe_eval(condition, output)
        except Exception as exc:
            self.console.log(
                f"[bold red]Error evaluating gleaning condition '{condition}': {exc}; executing gleaning round anyway[/bold red]"
            )
            return False


class APIWrapper(object):
    """Main API wrapper class - refactored for modularity and structured output support."""
    
    def __init__(self, runner):
        self.runner = runner
        self.default_lm_api_base = runner.config.get("default_lm_api_base", None)
        self.default_embedding_api_base = runner.config.get("default_embedding_api_base", None)
        
        # Initialize component handlers
        self.parser = ResponseParser(runner.console)
        self.llm_handler = LLMCallHandler(runner, runner.console)
        self.validator = ValidationHandler(runner, runner.console, self.llm_handler, self.parser)

    @freezeargs
    def gen_embedding(self, model: str, input: List[str]) -> List[float]:
        """A cached wrapper around litellm.embedding function."""
        # Create a unique key for the cache
        key = hashlib.md5(f"{model}_{input}".encode()).hexdigest()
        input = json.loads(input)

        # If the model starts with "gpt" and there is no openai key, prefix the model with "azure"
        if (
            model.startswith("text-embedding")
            and not os.environ.get("OPENAI_API_KEY")
            and self.runner.config.get("from_docwrangler", False)
        ):
            model = "azure/" + model

        with cache as c:
            # Try to get the result from cache
            result = c.get(key)
            if result is None:
                # If not in cache, compute the embedding
                if not isinstance(input[0], str):
                    input = [json.dumps(item) for item in input]

                input = [item if item else "None" for item in input]

                # FIXME: Should we use a different limit for embedding?
                self.runner.blocking_acquire("embedding_call", weight=1)
                if self.runner.is_cancelled:
                    raise asyncio.CancelledError("Operation was cancelled")

                extra_kwargs = {}
                if self.default_embedding_api_base:
                    extra_kwargs["api_base"] = self.default_embedding_api_base

                result = embedding(model=model, input=input, **extra_kwargs)
                # Cache the result
                c.set(key, result)

        return result

    def call_llm_batch(
        self,
        model: str,
        op_type: str,
        messages: List[Dict[str, str]],
        output_schema: Dict[str, str],
        verbose: bool = False,
        timeout_seconds: int = 120,
        max_retries_per_timeout: int = 2,
        bypass_cache: bool = False,
        litellm_completion_kwargs: Dict[str, Any] = {},
        op_config: Dict[str, Any] = {},
    ) -> LLMResult:
        """Batch version of call_llm - converts dict schema to list schema."""
        # Turn the output schema into a list of schemas
        output_schema = convert_dict_schema_to_list_schema(output_schema)

        # Invoke the LLM call
        return self.call_llm(
            model,
            op_type,
            messages,
            output_schema,
            verbose=verbose,
            timeout_seconds=timeout_seconds,
            max_retries_per_timeout=max_retries_per_timeout,
            bypass_cache=bypass_cache,
            litellm_completion_kwargs=litellm_completion_kwargs,
            op_config=op_config,
        )

    def call_llm(
        self,
        model: str,
        op_type: str,
        messages: List[Dict[str, str]],
        output_schema: Dict[str, str],
        tools: Optional[List[Dict[str, str]]] = None,
        scratchpad: Optional[str] = None,
        timeout_seconds: int = 120,
        max_retries_per_timeout: int = 2,
        validation_config: Optional[Dict[str, Any]] = None,
        gleaning_config: Optional[Dict[str, Any]] = None,
        verbose: bool = False,
        bypass_cache: bool = False,
        initial_result: Optional[Any] = None,
        litellm_completion_kwargs: Dict[str, Any] = {},
        op_config: Dict[str, Any] = {},
        output_mode: OutputMode = OutputMode.TOOLS,  # New parameter
    ) -> LLMResult:
        """Main LLM call method with support for both tools and structured output."""
        key = cache_key(
            model,
            op_type,
            messages,
            output_schema,
            scratchpad,
            self.runner.config.get("system_prompt", {}),
            op_config,
        )

        max_retries = max_retries_per_timeout
        attempt = 0
        rate_limited_attempt = 0
        
        while attempt <= max_retries:
            try:
                output = timeout(timeout_seconds)(self._cached_call_llm)(
                    key,
                    model,
                    op_type,
                    messages,
                    output_schema,
                    json.dumps(tools) if tools else None,
                    scratchpad,
                    validation_config=validation_config,
                    gleaning_config=gleaning_config,
                    verbose=verbose,
                    bypass_cache=bypass_cache,
                    initial_result=initial_result,
                    litellm_completion_kwargs=litellm_completion_kwargs,
                    op_config=op_config,
                    output_mode=output_mode,
                )
                
                # Log input and output if verbose
                if verbose:
                    self._log_verbose_output(messages, output)

                return output
                
            except RateLimitError:
                backoff_time = 4 * (2**rate_limited_attempt)
                max_backoff = 120
                sleep_time = min(backoff_time, max_backoff)
                self.runner.console.log(
                    f"[yellow]Rate limit hit. Retrying in {sleep_time:.2f} seconds...[/yellow]"
                )
                time.sleep(sleep_time)
                rate_limited_attempt += 1
            except APIConnectionError as e:
                self.runner.console.log(
                    f"[bold red]API connection error. Retrying...[/bold red] {e}"
                )
                time.sleep(1)
            except ServiceUnavailableError:
                self.runner.console.log(
                    "[bold red]Service unavailable. Retrying...[/bold red]"
                )
                time.sleep(1)
            except TimeoutError:
                if attempt == max_retries:
                    self.runner.console.log(
                        f"[bold red]LLM call timed out after {max_retries + 1} attempts[/bold red]"
                    )
                    return LLMResult(response=None, total_cost=0.0, validated=False)
                attempt += 1

    def _cached_call_llm(
        self,
        cache_key: str,
        model: str,
        op_type: str,
        messages: List[Dict[str, str]],
        output_schema: Dict[str, str],
        tools: Optional[str] = None,
        scratchpad: Optional[str] = None,
        validation_config: Optional[Dict[str, Any]] = None,
        gleaning_config: Optional[Dict[str, Any]] = None,
        verbose: bool = False,
        bypass_cache: bool = False,
        initial_result: Optional[Any] = None,
        litellm_completion_kwargs: Dict[str, Any] = {},
        op_config: Dict[str, Any] = {},
        output_mode: OutputMode = OutputMode.TOOLS,
    ) -> LLMResult:
        """Cached version of the call_llm function."""
        if (
            model.startswith("gpt")
            and not os.environ.get("OPENAI_API_KEY")
            and self.runner.config.get("from_docwrangler", False)
        ):
            model = "azure/" + model

        total_cost = 0.0
        validated = False
        
        with cache as c:
            response = c.get(cache_key)
            if response is not None and not bypass_cache:
                validated = True
            else:
                if not initial_result:
                    response = self.llm_handler.make_completion_call(
                        model,
                        messages,
                        output_mode,
                        output_schema,
                        tools,
                        scratchpad,
                        litellm_completion_kwargs,
                        op_config,
                    )
                else:
                    response = initial_result

                # Handle validation and gleaning
                response, additional_cost, validated = self.validator.handle_validation(
                    response,
                    output_schema,
                    output_mode,
                    validation_config,
                    gleaning_config,
                    model,
                    op_type,
                    messages,
                    tools,
                    scratchpad,
                    litellm_completion_kwargs,
                    op_config,
                    verbose,
                )
                total_cost += additional_cost

                # Only set the cache if the result is validated
                if validated:
                    c.set(cache_key, response)

        return LLMResult(response=response, total_cost=total_cost, validated=validated)

    def parse_llm_response(
        self,
        response: Any,
        schema: Dict[str, Any] = {},
        tools: Optional[List[Dict[str, str]]] = None,
        manually_fix_errors: bool = False,
        output_mode: OutputMode = OutputMode.TOOLS,  # New parameter
    ) -> List[Dict[str, Any]]:
        """Parse the response from a language model."""
        return self.parser.parse_response(response, schema, output_mode, tools, manually_fix_errors)

    def validate_output(self, operation: Dict, output: Dict, console: Console) -> bool:
        """Validate the output against the specified validation rules in the operation."""
        if "validate" not in operation:
            return True
        for validation in operation["validate"]:
            try:
                if not safe_eval(validation, output):
                    console.log(f"[bold red]Validation failed:[/bold red] {validation}")
                    console.log(f"[yellow]Output:[/yellow] {output}")
                    return False
            except Exception as e:
                console.log(f"[bold red]Validation error:[/bold red] {str(e)}")
                console.log(f"[yellow]Output:[/yellow] {output}")
                return False
        return True

    def _log_verbose_output(self, messages: List[Dict[str, str]], output: LLMResult) -> None:
        """Log verbose output for debugging."""
        # Truncate messages to 500 chars
        messages_str = str(messages)
        truncated_messages = (
            messages_str[:500] + "..."
            if len(messages_str) > 500
            else messages_str
        )

        # Log with nice formatting
        self.runner.console.print(
            Panel(
                Group(
                    Text("Input:", style="bold cyan"),
                    Text(truncated_messages),
                    Text("\nOutput:", style="bold cyan"),
                    Text(str(output)),
                ),
                title="[bold green]LLM Call Details[/bold green]",
                border_style="green",
            )
        )
