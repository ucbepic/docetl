import ast
import hashlib
import json
import re
import copy
import time
from typing import Any, Dict, List, Optional

from litellm import ModelResponse, RateLimitError, completion, embedding
from rich import print as rprint
from rich.console import Console

from docetl.utils import completion_cost

from .cache import cache, cache_key, freezeargs
from .llm import InvalidOutputError, LLMResult, timeout, truncate_messages
from .validation import (
    convert_dict_schema_to_list_schema,
    convert_val,
    get_user_input_for_schema,
    safe_eval,
    strict_render,
)

BASIC_MODELS = ["gpt-4o-mini", "gpt-4o"]


class APIWrapper(object):
    def __init__(self, runner):
        self.runner = runner

    @freezeargs
    def gen_embedding(self, model: str, input: List[str]) -> List[float]:
        """
        A cached wrapper around litellm.embedding function.

        This function uses LRU (Least Recently Used) cache to store and retrieve
        embeddings for given inputs. It can significantly speed up repeated calls
        with the same model and input.

        Args:
            model (str): The name of the embedding model to use.
            input (str): The input text to generate an embedding for.

        Returns:
            List[float]: The embedding vector as a list of floats.

        Note:
            The cache size is set to 1000. Adjust this value based on your memory
            constraints and usage patterns.
        """
        # Create a unique key for the cache
        key = hashlib.md5(f"{model}_{input}".encode()).hexdigest()
        input = json.loads(input)

        with cache as c:
            # Try to get the result from cache
            result = c.get(key)
            if result is None:
                # If not in cache, compute the embedding
                if not isinstance(input[0], str):
                    input = [json.dumps(item) for item in input]

                input = [item if item else "None" for item in input]

                # FIXME: Should we use a different limit for embedding?
                self.runner.rate_limiter.try_acquire(
                    "embedding_call", weight=1)
                result = embedding(model=model, input=input)
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
    ) -> LLMResult:
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
        )

    # Given a list of function calls from llm
    # be able to update a localized state which can be used for the next batch
    def process_func_calls(self, func_calls, state=None):
        for call in func_calls:
            match = re.match(r'(ADD|INCREMENT)\("(.+?)"\)', call)
            if match:
                action, item = match.groups()
                if action == "ADD":
                    state[item] = state.get(item, 0) + 1
                elif action == "INCREMENT" and item in state:
                    state[item] += 1

        return state

    # updated process function
    def processor(self, func_calls, state=None):
        if state is None:
            state = {}

        for call in func_calls:
            # Handling ADD("key")
            match = re.match(r'ADD\("(.+?)"\)', call)
            if match:
                item = match.group(1)
                if item not in state:
                    state[item] = {"count": 1, "summary": ""}
                else:
                    state[item]["count"] += 1

            # Handling INCREMENT("key")
            match = re.match(r'INCREMENT\("(.+?)"\)', call)
            if match:
                item = match.group(1)
                if item in state:
                    state[item]["count"] += 1

            # Handling UPDATE_SUMMARY("key", "data")
            match = re.match(r'UPDATE_SUMMARY\("(.+?)",\s*"(.+?)"\)', call)
            if match:
                item, summary = match.groups()
                if item in state:
                    state[item]["summary"] += " " + \
                        summary
                else:
                    state[item] = {"count": 0, "summary": summary}

        return state

    def process_func_calls_freq(self, func_calls, state=None):
        state = self.process_func_calls(func_calls, state)
        if not state:
            return {}

        # get count for each piece of data
        max_count = max((res.get("count", 0) for res in state.values()))

        if max_count <= 15:
            thresholds = {
                "most frequent": 6,
                "frequent": 4,
                "moderate": 3,
                "low": 1
            }
        else:
            thresholds = {
                "most frequent": 0.80 * max_count,
                "frequent": 0.60 * max_count,
                "moderate": 0.40 * max_count,
                "low": 1
            }

        labeled_scratchpad = {}
        for item, data in state.items():
            count = data["count"]

            if count >= thresholds["most frequent"]:
                frequency_label = "most frequent"
            elif count >= thresholds["frequent"]:
                frequency_label = "frequent"
            elif count >= thresholds["moderate"]:
                frequency_label = "moderate"
            else:
                frequency_label = "low"

            labeled_scratchpad[item] = {
                "frequency": frequency_label,
                "summary": data.get("summary", "")
            }

        return labeled_scratchpad

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
    ) -> LLMResult:
        """
        Cached version of the call_llm function.

        This function serves as a cached wrapper around _call_llm_with_cache. It uses
        the @freezeargs decorator to ensure immutable arguments and @functools.lru_cache
        for caching results.

        Args:
            cache_key (str): A unique key for caching.
            model (str): The model name.
            op_type (str): The operation type.
            messages (List[Dict[str, str]]): The messages to send to the LLM.
            output_schema (Dict[str, str]): The output schema dictionary.
            tools (Optional[str]): The tools to pass to the LLM.
            scratchpad (Optional[str]): The scratchpad to use for the operation.
            validation_config (Optional[Dict[str, Any]]): The validation configuration.
            gleaning_config (Optional[Dict[str, Any]]): The gleaning configuration.
            verbose (bool): Whether to print verbose output.
            bypass_cache (bool): Whether to bypass the cache.
            initial_result (Optional[Any]): The initial result to use for the operation, if exists.
        Returns:
            LLMResult: The response from _call_llm_with_cache.
        """
        total_cost = 0.0
        validated = False
        updated_state = {}
        with cache as c:
            response = c.get(cache_key)
            if response is not None and not bypass_cache:
                validated = True
            else:
                if not initial_result:
                    response = self._call_llm_with_cache(
                        model,
                        op_type,
                        messages,
                        output_schema,
                        tools,
                        scratchpad,
                        litellm_completion_kwargs,
                    )

                    self.runner.console.log(response)

                    # Copy response object to inject updated_scratchpad for next batch
                    newRes = copy.deepcopy(response)

                    arguments_str = newRes.choices[0].message.tool_calls[0].function.arguments

                    # Gather list of function calls in json format
                    arguments_dict = json.loads(arguments_str)
                    func_calls = arguments_dict["func_calls"]

                    updated_state = self.processor(
                        func_calls, {} if isinstance(scratchpad, str) else scratchpad)

                    self.runner.console.log("UPDATED")
                    self.runner.console.log(updated_state)

                    total_cost += completion_cost(response)
                else:
                    response = initial_result

                if gleaning_config:
                    # Retry gleaning prompt + regular LLM
                    num_gleaning_rounds = gleaning_config.get("num_rounds", 2)

                    parsed_output = (
                        self.parse_llm_response(
                            response, output_schema, tools)[0]
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
                        + [{"role": "assistant",
                            "content": json.dumps(parsed_output)}]
                    )

                    for rnd in range(num_gleaning_rounds):
                        # Prepare validator prompt
                        validator_prompt = strict_render(
                            gleaning_config["validation_prompt"],
                            {"output": parsed_output},
                        )
                        self.runner.rate_limiter.try_acquire(
                            "llm_call", weight=1)

                        # Get params for should refine
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

                        validator_response = completion(
                            model=gleaning_config.get("model", model),
                            messages=truncate_messages(
                                validator_messages
                                + [{"role": "user", "content": validator_prompt}],
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
                        )
                        total_cost += completion_cost(validator_response)

                        # Parse the validator response
                        suggestion = json.loads(
                            validator_response.choices[0]
                            .message.tool_calls[0]
                            .function.arguments
                        )
                        if not suggestion["should_refine"]:
                            break

                        if verbose:
                            self.runner.console.log(
                                f"Validator improvements (gleaning round {
                                    rnd + 1}): {suggestion['improvements']}"
                            )

                        # Prompt for improvement
                        improvement_prompt = f"""Based on the validation feedback:

                        ```
                        {suggestion['improvements']}
                        ```

                        Please improve your previous response. Ensure that the output adheres to the required schema and addresses any issues raised in the validation."""
                        messages.append(
                            {"role": "user", "content": improvement_prompt})

                        # Call LLM again
                        # response = self._call_llm_with_cache(
                        #     model,
                        #     op_type,
                        #     messages,
                        #     output_schema,
                        #     tools,
                        #     scratchpad,
                        #     litellm_completion_kwargs,
                        # )
                        parsed_output = self.parse_llm_response(
                            response, output_schema, tools
                        )[0]
                        validator_messages[-1] = {
                            "role": "assistant",
                            "content": json.dumps(parsed_output),
                        }

                        total_cost += completion_cost(response)

                    validated = True

                # If there's validation, handle it here
                elif validation_config:
                    num_tries = validation_config.get("num_retries", 2) + 1
                    validation_fn = validation_config.get("validation_fn")
                    val_rule = validation_config.get("val_rule")

                    # Try validation
                    i = 0
                    validation_result = False
                    while not validation_result and i < num_tries:
                        parsed_output, validation_result = validation_fn(
                            response)
                        if validation_result:
                            validated = True
                            break

                        # Append the validation result to messages
                        messages.append(
                            {
                                "role": "assistant",
                                "content": json.dumps(parsed_output),
                            }
                        )
                        messages.append(
                            {
                                "role": "user",
                                "content": f"Your output {parsed_output} failed my validation rule: {str(val_rule)}\n\nPlease try again.",
                            }
                        )
                        self.runner.console.log(
                            f"[bold red]Validation failed:[/bold red] {
                                val_rule}\n"
                            f"\t[yellow]Output:[/yellow] {parsed_output}\n"
                            f"\t({i + 1}/{num_tries})"
                        )
                        i += 1

                        # response = self._call_llm_with_cache(
                        #     model,
                        #     op_type,
                        #     messages,
                        #     output_schema,
                        #     tools,
                        #     scratchpad,
                        #     litellm_completion_kwargs,
                        # )
                        total_cost += completion_cost(response)

                else:
                    # No validation, so we assume the result is valid
                    validated = True

                # Only set the cache if the result tool calls or output is not empty
                if validated:
                    c.set(cache_key, response)

        return LLMResult(response=response, total_cost=total_cost, validated=validated, updated_state=updated_state)

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
    ) -> LLMResult:
        """
        Wrapper function that uses caching for LLM calls.

        This function generates a cache key and calls the cached version of call_llm.
        It retries the call if it times out after 60 seconds.

        Args:
            model (str): The model name.
            op_type (str): The operation type.
            messages (List[Dict[str, str]]): The messages to send to the LLM.
            output_schema (Dict[str, str]): The output schema dictionary.
            tools (Optional[List[Dict[str, str]]]): The tools to pass to the LLM.
            scratchpad (Optional[str]): The scratchpad to use for the operation.
            timeout_seconds (int): The timeout for the LLM call.
            max_retries_per_timeout (int): The maximum number of retries per timeout.
            bypass_cache (bool): Whether to bypass the cache.
            initial_result (Optional[Any]): The initial result to use for the operation, if exists.
        Returns:
            LLMResult: The result from the cached LLM call.

        Raises:
            TimeoutError: If the call times out after retrying.
        """
        key = cache_key(
            model,
            op_type,
            messages,
            output_schema,
            scratchpad,
            self.runner.config.get("system_prompt", {}),
        )

        max_retries = max_retries_per_timeout
        attempt = 0
        rate_limited_attempt = 0
        while attempt <= max_retries:
            try:
                return timeout(timeout_seconds)(self._cached_call_llm)(
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
                )
            except RateLimitError:
                # TODO: this is a really hacky way to handle rate limits
                # we should implement a more robust retry mechanism
                # Exponential backoff
                backoff_time = 4 * (2**rate_limited_attempt)
                max_backoff = 120  # Maximum backoff time of 60 seconds
                sleep_time = min(backoff_time, max_backoff)
                self.runner.console.log(
                    f"[yellow]Rate limit hit. Retrying in {
                        sleep_time:.2f} seconds...[/yellow]"
                )
                time.sleep(sleep_time)
                rate_limited_attempt += 1
            except TimeoutError:
                if attempt == max_retries:
                    self.runner.console.log(
                        f"[bold red]LLM call timed out after {
                            max_retries + 1} attempts[/bold red]"
                    )
                    # TODO: HITL
                    return LLMResult(response=None, total_cost=0.0, validated=False)
                attempt += 1

    def _call_llm_with_cache(
        self,
        model: str,
        op_type: str,
        messages: List[Dict[str, str]],
        output_schema: Dict[str, str],
        tools: Optional[str] = None,
        scratchpad: Optional[str] = None,
        litellm_completion_kwargs: Dict[str, Any] = {},
        updated_state: Optional[dict[str, Any]] = {}
    ) -> Any:
        """
        Make an LLM call with caching.

        This function prepares the necessary parameters and makes a call to the LLM
        using the provided model, operation type, prompt, and output schema.

        Args:
            model (str): The model name.
            op_type (str): The operation type.
            messages (List[Dict[str, str]]): The messages to send to the LLM.
            output_schema (Dict[str, str]): The output schema dictionary.
            tools (Optional[str]): The tools to pass to the LLM.
            scratchpad (Optional[str]): The scratchpad to use for the operation.
        Returns:
            str: The response from the LLM.
        """
        print("TESTING")
        props = {key: convert_val(value)
                 for key, value in output_schema.items()}
        self.runner.console.log("OLD PROPS")
        self.runner.console.log(props)

        newProps = {"func_calls": []}
        use_tools = True

        if (
            len(props) == 1
            # and list(props.values())[0].get("type") == "string"
            and scratchpad is None
            and ("sagemaker" in model)
        ):
            use_tools = False

        if tools is None and use_tools:
            if scratchpad is not None:
                newProps["updated_scratchpad"] = {"type": "object"}

            parameters = {"type": "object", "properties": props}
            parameters["required"] = list(newProps.keys())

            # TODO: this is a hack to get around the fact that gemini doesn't support additionalProperties
            if "gemini" not in model and "claude" not in model:
                parameters["additionalProperties"] = False

            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "send_output",
                        "description": "Send output back to the user. ",
                        "parameters": parameters,
                    },
                }
            ]
            if "claude" not in model:
                tools[0]["additionalProperties"] = False
                # tools[0]["strict"] = True

            tool_choice = {"type": "function",
                           "function": {"name": "send_output"}}

        elif tools is not None:
            tools = json.loads(tools)
            tool_choice = (
                "required"
                if any(tool.get("required", False) for tool in tools)
                else "auto"
            )
            tools = [
                {"type": "function", "function": tool["function"]} for tool in tools
            ]

        else:
            tools = None
            tool_choice = None

        persona = self.runner.config.get("system_prompt", {}).get(
            "persona", "a helpful assistant"
        )
        dataset_description = self.runner.config.get("system_prompt", {}).get(
            "dataset_description", "a collection of unstructured documents"
        )
        parethetical_op_instructions = (
            "many inputs:one output" if op_type == "reduce" else "one input:one output"
        )


####
# Below comments were in previous prompt, but I commented it out for now for testing.

# IMPORTANT: Only use the scratchpad if your task specifically requires tracking items that appear multiple times across batches. If you only need to track distinct/unique items, leave the scratchpad empty and set updated_scratchpad to null.

# The intermediate output contains the result that directly answers the user's task, for **all** the data processed so far, including the current batch. You must return this via the send_output function.

# Example task that NEEDS scratchpad - counting words that appear >2 times:
# # Words seen 3+ times - this is your actual result
# - Call send_output with: {{"frequent_words": ["the", "and"]}}
# # Must track words seen 1-2 times
# - Set updated_scratchpad to: {{"pending": {{"cat": 2, "dog": 1}}}}

# Example task that does NOT need scratchpad - collecting unique locations:
# # Just the unique items
# - Call send_output with: {{"locations": ["New York", "Paris"]}}
# # No need to track counts since we only want distinct items
# - Set updated_scratchpad to: null
# 3. Set updated_scratchpad accordingly

    # check scratchpad is updating every call

        system_prompt = f"You are a {persona}, helping the user make sense of their data. The dataset description is: {dataset_description}. You will be performing a {
            op_type} operation ({parethetical_op_instructions}). You will perform the specified task on the provided data, as precisely and exhaustively (i.e., high recall) as possible. return a list of function calls, based on the definitions below, via the `send_output` function"
        system_prompt += f"""

        Via the `send_output` function, make sure to return the func_calls list even if empty. 

the function calls available for these batches are:
ADD(key:string) -> updates state by adding data to the intermediate state
INCREMENT(key:string) -> updates state by incrementing previosuly existing data to the intermediate state
UPDATE_SUMMARY(key: string, data:string) -> THIS FUNCTION MUST TAKE TWO ARGUMENTS; Call this function after every ADD or INCREMENT function for new note-worthy that is presented that isn't already in the state. It should only be called when you need to keep track of data other than counts related to specific instances. The first argument of the function should be the same key as what you use for ADD or INCREMENT(ex. when trying to see different symptoms of diseases)

WHEN CALLING ANY FUNCTION CALLS THE KEY MUST BE THE SAME WHEN REFERRING TO THE SAME SUBJECT.


DO NOT INCLUDE ANY COUNTS OR ":" in the argument for the ADD or INCREMENT functions. 

To take care of duplicate keys being found, you can just append multiple INCREMENT calls to the func_calls list.

USE THE INCREMENT FUNCTION AS MANY TIMES AS YOU SEE YOU CAN


As you process each batch:
1. Use both the previous output and scratchpad (if needed) to inform your processing
2.) Update the func_calls list with the functions: ADD(key: string) or INCREMENT(key: string)

You may also use the UPDATE_SUMMARY function and use the same key as you do for the ADD or INCREMENT functions. 

For example, for a dataset of fruits, an example func_calls list via `send_output` would look something like: func_calls: ['ADD("apple")', 'UPDATE_SUMMARY('apple', 'a very healthy fruit')' 'ADD("orange")', 'UPDATE_SUMMARY('orange', 'a very healthy fruit')', 'INCREMENT("apple")', 'ADD("grape")', 'UPDATE_SUMMARY('grape', 'a very healthy fruit')', 'INCREMENT("orange")']
"""

        if scratchpad:
            system_prompt += f"""

You are incrementally processing data across multiple batches. You will see:
1. The current batch of data to process
2. The intermediate output so far (what you returned last time)
3. A scratchpad of the current state (key and value pairs): {scratchpad}


For Counts:
if the key already exists in the scratchpad, you can use the INCREMENT function, but if it doesn't exist, use the ADD function, initially, and if you see it again, use INCREMENT from then on out. 


AFTER DECIDING THE FUNCTION CALLS, update the func_calls via the `send_output` function


Your main result must be sent via send_output."""
        # Truncate messages if they exceed the model's context length
        messages = truncate_messages(messages, model)

        self.runner.rate_limiter.try_acquire("llm_call", weight=1)
        if tools is not None:
            try:
                response = completion(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": system_prompt,
                        },
                    ]
                    + messages,
                    tools=tools,
                    tool_choice=tool_choice,
                    **litellm_completion_kwargs,
                )
            except Exception as e:
                # Check that there's a prefix for the model name if it's not a basic model
                if model not in BASIC_MODELS:
                    if "/" not in model:
                        raise ValueError(
                            f"Note: You may also need to prefix your model name with the provider, e.g. 'openai/gpt-4o-mini' or 'gemini/gemini-1.5-flash' to conform to LiteLLM API standards. Original error: {
                                e}"
                        )
                raise e
        else:
            try:
                response = completion(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": system_prompt,
                        },
                    ]
                    + messages,
                    **litellm_completion_kwargs,
                )
            except Exception as e:
                # Check that there's a prefix for the model name if it's not a basic model
                if model not in BASIC_MODELS:
                    if "/" not in model:
                        raise ValueError(
                            f"Note: You may also need to prefix your model name with the provider, e.g. 'openai/gpt-4o-mini' or 'gemini/gemini-1.5-flash' to conform to LiteLLM API standards. Original error: {
                                e}"
                        )
                raise e

        return response

    def parse_llm_response(
        self,
        response: Any,
        schema: Dict[str, Any] = {},
        tools: Optional[List[Dict[str, str]]] = None,
        manually_fix_errors: bool = False,
        updated_state: Dict[str, Any] = {}
    ) -> List[Dict[str, Any]]:
        """
        Parse the response from a language model.
        This function extracts the tool calls from the LLM response and returns the arguments
        """
        try:
            return self._parse_llm_response_helper(response, schema, tools, updated_state)
        except InvalidOutputError as e:
            if manually_fix_errors:
                rprint(
                    f"[bold red]Could not parse LLM output:[/bold red] {
                        e.message}\n"
                    f"\tExpected Schema: {e.expected_schema}\n"
                    f"\tPlease manually set this output."
                )
                rprint(
                    f"\n[bold yellow]LLM-Generated Response:[/bold yellow]\n{
                        response}"
                )
                output = get_user_input_for_schema(schema)

                return [output]
            else:
                raise e

    def _parse_llm_response_helper(
        self,
        response: Any,
        schema: Dict[str, Any] = {},
        tools: Optional[List[Dict[str, str]]] = None,
        updated_state: Optional[Dict[str, str]] = {}
    ) -> List[Dict[str, Any]]:
        """
        Parse the response from a language model.

        This function extracts the tool calls from the LLM response and returns the arguments
        of any 'send_output' function calls as a list of dictionaries.

        Args:
            response (Any): The response object from the language model.
            schema (Optional[Dict[str, Any]]): The schema that was passed to the LLM.
            tools (Optional[List[Dict[str, str]]]): The tools that were passed to the LLM.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing the parsed output.

        Raises:
            InvalidOutputError: If the response is not valid.
        """

        if not response:
            raise InvalidOutputError(
                "No response from LLM", [{}], schema, [], [])

        tool_calls = (
            response.choices[0].message.tool_calls
            if "tool_calls" in dir(response.choices[0].message)
            else []
        )

        # Check if there are no tools and the schema has a single key-value pair
        if not tools and len(schema) == 1 and not tool_calls:
            key = next(iter(schema))
            self.runner.console.log("PARSING KEYS")
            self.runner.console.log({key: response.choices[0].message.content})
            return [{key: response.choices[0].message.content}]

        # Parse the response based on the provided tools
        if tools:
            # If custom tools are provided, parse accordingly
            tool_calls = response.choices[0].message.tool_calls
            results = []
            for tool_call in tool_calls:
                for tool in tools:
                    self.runner.console.log("PARSER")
                    self.runner.console.log(tool["function"]["name"])
                    if tool_call.function.name == tool["function"]["name"]:
                        try:
                            function_args = json.loads(
                                tool_call.function.arguments)
                        except json.JSONDecodeError:
                            return [{}]
                        # Execute the function defined in the tool's code
                        local_scope = {}
                        exec(tool["code"].strip(), globals(), local_scope)
                        function_result = local_scope[tool["function"]["name"]](
                            **function_args
                        )
                        function_args.update(function_result)
                        results.append(function_args)
            return results
        else:
            if not tool_calls:
                raise InvalidOutputError(
                    "No tool calls in LLM response", [
                        {}], schema, response.choices, []
                )

            outputs = []
            for tool_call in tool_calls:
                if response.choices[0].finish_reason == "content_filter":
                    raise InvalidOutputError(
                        "Content filter triggered by LLM provider.",
                        "",
                        schema,
                        response.choices,
                        tools,
                    )

                try:

                    output_dict = json.loads(
                        response.choices[0].message.content)

                    output_dict['updated_scratchpad'] = updated_state
                    # Augment output_dict with empty values for any keys in the schema that are not in output_dict
                    for key in schema:
                        if key not in output_dict:
                            output_dict[key] = "Not found"

                    if "ollama" in response.model or "sagemaker" in response.model:
                        for key, value in output_dict.items():
                            if not isinstance(value, str):
                                continue
                            try:
                                output_dict[key] = ast.literal_eval(value)
                            except Exception:
                                try:
                                    if value.startswith("["):
                                        output_dict[key] = ast.literal_eval(
                                            value + "]")
                                    else:
                                        output_dict[key] = value
                                except Exception:
                                    pass
                    outputs.append(output_dict)
                except json.JSONDecodeError:
                    raise InvalidOutputError(
                        "Could not decode LLM JSON response",
                        [tool_call.function.arguments],
                        schema,
                        response.choices,
                        tools,
                    )
                except Exception as e:
                    raise InvalidOutputError(
                        f"Error parsing LLM response: {e}",
                        [tool_call.function.arguments],
                        schema,
                        response.choices,
                        tools,
                    )

            return outputs

        # message = response.choices[0].message
        # return [json.loads(message.content)]

    def validate_output(self, operation: Dict, output: Dict, console: Console) -> bool:
        """
        Validate the output against the specified validation rules in the operation.

        Args:
            operation (Dict): The operation dictionary containing validation rules.
            output (Dict): The output to be validated.
            console (Console): The console object for logging.

        Returns:
            bool: True if all validations pass, False otherwise.
        """
        if "validate" not in operation:
            return True
        for validation in operation["validate"]:
            try:
                if not safe_eval(validation, output):
                    console.log(
                        f"[bold red]Validation failed:[/bold red] {validation}")
                    console.log(f"[yellow]Output:[/yellow] {output}")
                    return False
            except Exception as e:
                console.log(f"[bold red]Validation error:[/bold red] {str(e)}")
                console.log(f"[yellow]Output:[/yellow] {output}")
                return False
        return True
