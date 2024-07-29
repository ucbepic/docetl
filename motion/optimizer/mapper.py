"""
This file describes mapper optimizations.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import types
from motion.agent import Agent, OpenAILLM
from typing import List, Tuple, Any
from motion.executor import Operation, apply_operation
from motion.types import *
from litellm import completion, completion_cost
from tqdm import tqdm
import inspect
from motion.operators import LLMMapper, LLMParallelFlatMapper, LLMReducer
from motion.operators.mapper import AddUniqueIDToKey, RemoveUniqueIDFromKey
from copy import deepcopy
import ast


def single_glean_map(
    operation: Operation, error_data: OpError
) -> Tuple[OpOutput, OpError, float, int]:
    original_key = error_data.old_key
    original_value = error_data.old_value
    messages = error_data.prompt + [
        {"role": "assistant", "content": error_data.response.choices[0].message.content}
    ]
    error_message = error_data.error_msg
    additional_cost = 0
    output = None
    gleaning_rounds = 0

    # TODO(shreyashankar): don't hardcode 4
    for _ in range(4):
        gleaning_rounds += 1
        new_prompt = messages + [
            {
                "role": "user",
                "content": f"This is not the correct response. {error_message}\n\nPlease try again.",
            },
        ]

        new_response = completion(model=operation.operator.model, messages=new_prompt)
        additional_cost += completion_cost(new_response)

        new_key, new_value = operation.operator.process_response(
            new_response, key=original_key, value=original_value
        )

        output = OpOutput(
            id=error_data.id,
            prompt=new_prompt,
            response=new_response,
            new_key=new_key,
            new_value=new_value,
        )

        try:
            # Do validation
            operation.operator.validate(
                original_key, original_value, new_key, new_value
            )

            return output, None, additional_cost, gleaning_rounds
        except Exception as e:
            # If the error persists, we update the error data
            error_data = OpError(
                id=error_data.id,
                old_key=original_key,
                old_value=original_value,
                prompt=new_prompt,
                response=new_response,
                new_key=error_data.new_key,
                new_value=error_data.new_value,
                error_msg=str(e),
            )
            messages = new_prompt + [
                {
                    "role": "assistant",
                    "content": new_response.choices[0].message.content,
                }
            ]
            error_message = str(e)

    return output, error_data, additional_cost, gleaning_rounds


def glean_map(
    operation: Operation,
    sample_data: List[Tuple[Any, Any]],
    results: List[OpOutput],
    errors: List[OpError],
    num_workers: int,
    base_cost: float,
) -> List[Dict[str, Any]]:
    # Isolate LLM errors and keep reprompting with the error message and
    # prompt-response pairs until the operation is successful.
    all_gleaning_calls = []
    base_correct = len(results) - len(errors)

    # for error_type, error_data, error_index in errors:
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for error_data in errors:
            futures.append(executor.submit(single_glean_map, operation, error_data))

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"Computing gleaning cost and accuracy estimates for {operation.operator.__class__.__name__}...",
        ):
            output, updated_error, additional_cost, gleaning_rounds = future.result()
            all_gleaning_calls.append(
                (gleaning_rounds, updated_error is None, additional_cost)
            )
            if output:
                # Replace the id in results with the new output
                for i, result in enumerate(results):
                    if result.id == output.id:
                        results[i] = output

            if updated_error is None:
                # Remove the id from errors
                errors = [error for error in errors if error.id != output.id]
            else:
                # Replace the id in errors with the new error
                for i, error in enumerate(errors):
                    if error.id == updated_error.id:
                        errors[i] = updated_error

    # Compute accuracy-gleaning tradeoff
    max_rounds = max((rounds for rounds, _, _ in all_gleaning_calls), default=0)

    # Compute plans for each possible number of gleaning rounds
    plans = []
    for r in range(1, max_rounds + 1):
        # Compute expected cost and accuracy for this sample
        sample_cost = base_cost + sum(
            cost for rounds, _, cost in all_gleaning_calls if rounds <= r
        )
        expected_accuracy = (
            base_correct
            + sum(
                int(success) for rounds, success, _ in all_gleaning_calls if rounds <= r
            )
        ) / len(results)

        plans.append(
            {
                "rounds": r,
                "operations": [operation],
                "results": results,
                "errors": errors,
                "sample_cost": sample_cost,
                "expected_accuracy": expected_accuracy,
            }
        )

    return plans


class SynthesizedMapper(LLMMapper):
    def __init__(self, model: str, subtask_messages: str, **llm_kwargs):
        super().__init__(model, **llm_kwargs)
        exec(subtask_messages, locals())
        self.subtask_messages = locals()["messages"]
        assert (
            isinstance(self.subtask_messages, list) and len(self.subtask_messages) > 0
        )

    def generate_prompt(self, key: K, value: V) -> list:
        formatted_messages = []
        for message in self.subtask_messages:
            formatted_messages.append(
                {
                    "role": message["role"],
                    "content": message["content"].format(key=key, value=value),
                }
            )
        return formatted_messages

    def execute(self, key: K, value: V) -> Tuple[Tuple[RK, RV], Dict]:
        prompt = self.generate_prompt(key[0], value)
        response = completion(messages=prompt, model=self.model, **self.llm_kwargs)
        new_key, new_val = self.process_response(response, key=key[0], value=value)
        return ((new_key, key[1]), new_val), {
            "prompt": prompt,
            "response": response,
        }


class SynthesizedReducer(LLMReducer):
    def __init__(self, model: str, reduce_prompt: str, **llm_kwargs):
        super().__init__(model, **llm_kwargs)
        exec(reduce_prompt, locals())
        self.reduce_prompt = locals()["messages"]
        assert isinstance(self.reduce_prompt, list) and len(self.reduce_prompt) > 0

    def generate_prompt(self, key: K, values: List[V]) -> list:
        messages = []
        for message in self.reduce_prompt:
            messages.append(
                {
                    "role": message["role"],
                    "content": message["content"].format(key=key, values=values),
                }
            )
        return messages

    def execute(self, key: K, values: List[V]) -> Tuple[RV, Dict]:
        prompt = self.generate_prompt(key[0], values)
        response = completion(messages=prompt, model=self.model, **self.llm_kwargs)
        new_val = self.process_response(response, key=key[0], values=values)

        return new_val, {
            "prompt": prompt,
            "response": response,
        }


def decompose_parallel_flatmap(
    operation: Operation,
    sample_data: List[Tuple[Any, Any]],
    results: List[OpOutput],
    errors: List[OpError],
    num_workers: int,
) -> Tuple[List[Operation], List[OpOutput], List[OpError], float, float]:
    """
    See if the mapper should be decomposed into a parallelflatmap (and then reduced)
    """
    if len(errors) == 0:
        return [operation], results, errors, None, None

    # Generate a prompt for the LLM to decide on decomposition
    # Create sample inputs
    sample_inputs = "---------\n".join(
        [
            f"Key: {str(k)[:25] + ('...' if len(str(k)) > 25 else '')}\nValue: {str(v)[:500] + ('...' + str(len(str(v)) - 500) + ' chars remaining' if len(str(v)) > 500 else '')}"
            for k, v in sample_data[:5]
        ]
    )

    # Create sample error
    sample_error = next(
        (
            f"Error: {e.error_msg[:1000] + ('...' + str(len(e.error_msg) - 1000) + ' chars remaining' if len(e.error_msg) > 1000 else '')}"
            for e in sorted(errors, key=lambda x: len(x.error_msg), reverse=True)
        ),
        "No errors available",
    )

    decision_prompt = [
        {
            "role": "system",
            "content": "You are an AI assistant that helps optimize data processing pipelines.",
        },
        {
            "role": "user",
            "content": f"""
        Analyze the following map operation (that operates on one key-value pair at a time) and determine if it should be decomposed into multiple prompts that do different subtasks in parallel, while still operating on one key-value pair at a time:

        Operation: {operation.operator.__class__.__name__}
        Generate prompt method:
        ```python
        {inspect.getsource(operation.operator.generate_prompt)}
        ```
        Process response method:
        ```python
        {inspect.getsource(operation.operator.process_response)}
        ```
        Sample inputs:
        {sample_inputs}

        Error rate: {len(errors) / len(sample_data):.2%}
        Sample error:
        {sample_error}

        Should this operation be decomposed? Respond with YES or NO, followed by a brief explanation.
        """,
        },
    ]
    decompose_response = completion(model="gpt-4o", messages=decision_prompt)

    decision = "YES" in decompose_response.choices[0].message.content
    if not decision:
        return [operation], results, errors, None, None

    # Query the agent for the subtask prompts and the larger reduce prompt that will be used to combine the subtasks' results
    decision_prompt.append(
        {
            "role": "assistant",
            "content": decompose_response.choices[0].message.content,
        }
    )

    # TODO: figure out how to make this not code
    decision_prompt.append(
        {
            "role": "user",
            "content": "Please provide the prompts for each subtask to be run in parallel. Each subtask should be represented as a Python code block containing a list of messages assigned to a variable named 'messages' (each message has a string role and a string content template), where the only variables allowed in the template are 'key' or 'value', in placeholders like '{key}' or '{value}'. Each message's content should not be an f-string, it should be a normal string.",
        }
    )

    decompose_response = completion(model="gpt-4o", messages=decision_prompt)

    # Extract code blocks from the response
    code_blocks = re.findall(
        r"```python\n(.*?)```",
        decompose_response.choices[0].message.content,
        re.DOTALL,
    )

    if len(code_blocks) < 2:
        print(
            f"Could not decompose {operation.operator.__class__.__name__} into a parallel flatmap."
        )
        return [operation], results, errors, None, None

    # Separate subtask prompts and final reduce prompt

    subtask_prompts = code_blocks
    subtask_prompts = [
        prompt.strip().replace('f"', '"').replace("f'", "'")
        for prompt in subtask_prompts
    ]

    # Do another call to get the final reduce prompt
    decision_prompt.append(
        {
            "role": "assistant",
            "content": decompose_response.choices[0].message.content,
        }
    )
    decision_prompt.append(
        {
            "role": "user",
            "content": f"""
            Recall the original task prompt:

            Operation: {operation.operator.__class__.__name__}
            Generate prompt method:
            ```python
            {inspect.getsource(operation.operator.generate_prompt)}
            ```
            
            Please edit the prompt above to incorporate the subtask outputs (you should not remove information from the original prompt). The subtask outputs will be provided as a list of strings, where each string is the output of one subtask. The variable `{{values}}` will be replaced with the list of subtask outputs. Return your edited prompt as a Python list of messages within a Python code block, where each message has a string role and a string content template, where the only variables allowed in the template are 'key' or 'values', in placeholders like '{{key}}' or '{{values}}'. Each message's content should not be an f-string, it should be a normal string.
            """,
        }
    )
    reduce_response = completion(model="gpt-4o", messages=decision_prompt)

    # Extract code blocks from the reduce response
    reduce_code_blocks = re.findall(
        r"```python\n(.*?)```",
        reduce_response.choices[0].message.content,
        re.DOTALL,
    )

    if len(reduce_code_blocks) < 1:
        print(
            f"Could not generate reduce prompt for {operation.operator.__class__.__name__}."
        )
        return [operation], results, errors, None, None

    reduce_prompt = reduce_code_blocks[0].strip().replace('f"', '"').replace("f'", "'")

    def extract_variable_name(code):
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            return target.id
        except SyntaxError:
            pass
        return None

    for i, prompt in enumerate(subtask_prompts):
        var_name = extract_variable_name(prompt)
        if var_name and var_name.startswith("messages"):
            subtask_prompts[i] = prompt.replace(f"{var_name} =", "messages =", 1)

    var_name = extract_variable_name(reduce_prompt)
    if var_name and var_name.startswith("messages"):
        reduce_prompt = reduce_prompt.replace(f"{var_name} =", "messages =", 1)

    print(
        f"Attempting to decompose {operation.operator.__class__.__name__} into a parallel flatmap (of {len(subtask_prompts)} subtasks) and a reducer..."
    )
    print()

    # Print out each of the subtask prompts
    print("Subtask Prompts:")
    for i, prompt in enumerate(subtask_prompts, 1):
        print(f"Subtask {i}:")
        print("\t" + prompt.replace("\n", "\n\t"))
        print()

    # Print out the reduce prompt
    print("Reduce Prompt:")
    print("\t" + reduce_prompt.replace("\n", "\n\t"))
    print()

    # Create LLMParallelFlatMapper with subtask prompts
    parallel_mapper = LLMParallelFlatMapper(
        [
            SynthesizedMapper(operation.operator.model, subtask_prompts[i])
            for i in range(len(subtask_prompts))
        ]
    )

    # Create reducer with subtask prompts
    reducer = SynthesizedReducer(operation.operator.model, reduce_prompt)

    remove_unique_id_from_key = RemoveUniqueIDFromKey()
    remove_unique_id_from_key.validate = operation.operator.validate
    remove_unique_id_from_key.correct = operation.operator.correct

    # Create sequence of operations
    operations = [
        Operation(AddUniqueIDToKey()),
        Operation(parallel_mapper),
        Operation(reducer),
        Operation(remove_unique_id_from_key),
    ]

    # Try running the pipeline on the sample data
    current_data = deepcopy(sample_data)

    total_cost = 0
    new_pipeline_results = []
    new_pipeline_errors = []
    accuracy = None
    for i, op in enumerate(operations):
        if i < len(operations) - 1:
            current_data, cost = apply_operation(current_data, op, num_workers)
        else:
            new_pipeline_results, new_pipeline_errors, cost = apply_operation(
                current_data, op, num_workers, building=True
            )
            accuracy = 1 - len(new_pipeline_errors) / len(sample_data)

        total_cost += cost

    return (
        operations,
        new_pipeline_results,
        new_pipeline_errors,
        total_cost,
        accuracy,
    )
