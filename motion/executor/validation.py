import sys
import json
from typing import List, Union

from motion.types import *
from motion.operators import (
    KeyResolver,
    Operator,
    LLMMapper,
    LLMReducer,
    LLMFlatMapper,
    LLMParallelFlatMapper,
    LLMFilterer,
)


def handle_validation_errors(
    errors: List[Union[OpError, OpFlatError, OpParallelFlatError, OpReduceError]],
    action: ValidatorAction,
    operator: Operator,
) -> List[CorrectedOutput]:
    if not errors:
        return []

    if action == ValidatorAction.PROMPT:
        print("Validation Errors:", file=sys.stderr)
        for error in errors:
            if isinstance(operator, LLMParallelFlatMapper):
                (
                    record_id,
                    input_key,
                    input_value,
                    prompts,
                    responses,
                    output_kv_pairs,
                ) = error
                print(
                    f"  - ID: {record_id}, Input: ({input_key}, {input_value})",
                    file=sys.stderr,
                )
                for i, (prompt, response) in enumerate(zip(prompts, responses)):
                    print(f"    Prompt {i+1}: {prompt}", file=sys.stderr)
                    print(f"    Response {i+1}: {response}", file=sys.stderr)
                for output_key, output_value in output_kv_pairs:
                    print(
                        f"    Output: ({output_key}, {output_value})",
                        file=sys.stderr,
                    )
            elif isinstance(operator, LLMFlatMapper):
                record_id, input_key, input_value, prompt, response, output_kv_pairs = (
                    error
                )
                print(
                    f"  - ID: {record_id}, Input: ({input_key}, {input_value})",
                    file=sys.stderr,
                )
                print(f"    Prompt: {prompt}", file=sys.stderr)
                print(f"    Response: {response}", file=sys.stderr)
                for output_key, output_value in output_kv_pairs:
                    print(
                        f"    Output: ({output_key}, {output_value})",
                        file=sys.stderr,
                    )
            elif isinstance(operator, LLMMapper):
                (
                    record_id,
                    input_key,
                    input_value,
                    prompt,
                    response,
                    output_key,
                    output_value,
                ) = error
                print(
                    f"  - ID: {record_id}, Input: ({input_key}, {input_value}), Output: ({output_key}, {output_value})",
                    file=sys.stderr,
                )
                print(f"    Prompt: {prompt}", file=sys.stderr)
                print(f"    Response: {response}", file=sys.stderr)
            elif isinstance(operator, LLMReducer):
                record_id, key, input_values, prompt, response, output_value = error
                print(
                    f"  - ID: {record_id}, Key: {key}, Input Values: {input_values}, Output: {output_value}",
                    file=sys.stderr,
                )
                print(f"    Prompt: {prompt}", file=sys.stderr)
                print(f"    Response: {response}", file=sys.stderr)
            elif isinstance(operator, KeyResolver):
                record_id, input_key, output_key, prompt, response = error
                print(
                    f"  - ID: {record_id}, Input Key: {input_key}, Resolved Key: {output_key}",
                    file=sys.stderr,
                )
                print(f"    Prompt: {prompt}", file=sys.stderr)
                print(f"    Response: {response}", file=sys.stderr)
            elif isinstance(operator, LLMFilterer):
                record_id, input_key, input_value, prompt, response, output = error
                print(
                    f"  - ID: {record_id}, Input: ({input_key}, {input_value})",
                    file=sys.stderr,
                )
                print(f"    Prompt: {prompt}", file=sys.stderr)
                print(f"    Response: {response}", file=sys.stderr)
                print(f"    Filtered: {'True' if output else 'False'}", file=sys.stderr)

        print(
            "\nEnter corrections as a JSON dictionary mapping ID to [new_key, new_value] for Mapper, [[new_key1, new_value1], [new_key2, new_value2], ...] for FlatMapper, [new_value] for Reducer, or [new_key] for KeyResolver, or press Enter to skip:",
            file=sys.stderr,
        )
        try:
            user_input = sys.stdin.readline().strip()
            if user_input:
                corrections = json.loads(user_input)
                corrected_errors = []
                for error in errors:
                    record_id = error.id
                    if record_id in corrections:
                        if isinstance(operator, LLMFlatMapper):
                            input_key, input_value = error.old_key, error.old_value
                            new_kv_pairs = corrections[record_id]
                            corrected_kv_pairs = operator.correct(
                                input_key, input_value, new_kv_pairs
                            )
                            corrected_errors.append(
                                CorrectedOutput(record_id, None, corrected_kv_pairs)
                            )
                            continue
                        elif isinstance(operator, LLMMapper):
                            input_key, input_value = error.old_key, error.old_value
                            new_key, new_value = corrections[record_id]
                            corrected_key, corrected_value = operator.correct(
                                input_key, input_value, new_key, new_value
                            )
                        elif isinstance(operator, LLMReducer):
                            key, input_values = error.old_key, error.old_values
                            new_value = corrections[record_id][0]
                            corrected_key, corrected_value = operator.correct(
                                key, input_values, new_value
                            )
                        elif isinstance(operator, KeyResolver):
                            input_key, output_key = error.old_key, error.new_key
                            new_key = corrections[record_id][0]
                            corrected_key = operator.correct(input_key, new_key)
                            corrected_value = output_key  # Keep the original value
                        elif isinstance(operator, LLMFilterer):
                            input_key, input_value = error.old_key, error.old_value
                            new_output = corrections[record_id][0]
                            corrected_key, corrected_value = input_key, input_value
                            operator.correct(input_key, input_value, new_output)
                        corrected_errors.append(
                            CorrectedOutput(record_id, corrected_key, corrected_value)
                        )
                return corrected_errors
            return []
        except json.JSONDecodeError:
            print("Invalid JSON input. Skipping corrections.", file=sys.stderr)
        except KeyboardInterrupt:
            print("\nOperation aborted by user.", file=sys.stderr)
            sys.exit(1)
    elif action == ValidatorAction.WARN or action == ValidatorAction.FAIL:
        error_messages = []
        for error in errors:
            if isinstance(operator, LLMParallelFlatMapper):
                prompts_responses = "\n".join(
                    [
                        f"\tPrompt {i+1}: {prompt}\n\tResponse {i+1}: {response}"
                        for i, (prompt, response) in enumerate(
                            zip(error.prompts, error.responses)
                        )
                    ]
                )
                error_messages.append(
                    f"Warning: Validation failed for ID: {error.id}, Input: ({error.old_key}, {error.old_value}), Output: {error.new_key_value_pairs}\n"
                    f"{prompts_responses}"
                )
            elif isinstance(operator, LLMFlatMapper):
                error_messages.append(
                    f"Warning: Validation failed for ID: {error.id}, Input: ({error.old_key}, {error.old_value}), Output: {error.new_key_value_pairs}\n"
                    f"\tPrompt: {error.prompt}\n"
                    f"\tResponse: {error.response}"
                )
            elif isinstance(operator, LLMMapper):
                error_messages.append(
                    f"Warning: Validation failed for ID: {error.id}, Input: ({error.old_key}, {error.old_value}), Output: ({error.new_key}, {error.new_value})\n"
                    f"\tPrompt: {error.prompt}\n"
                    f"\tResponse: {error.response}"
                )
            elif isinstance(operator, LLMReducer):
                error_messages.append(
                    f"Warning: Validation failed for ID: {error.id}, Key: {error.old_key}, Input Values: {error.old_values}, Output: {error.new_value}\n"
                    f"\tPrompt: {error.prompt}\n"
                    f"\tResponse: {error.response}"
                )
            elif isinstance(operator, KeyResolver):
                error_messages.append(
                    f"Warning: Validation failed for ID: {error.id}, Input Key: {error.old_key}, Resolved Key: {error.new_key}\n"
                    f"\tPrompt: {error.prompt}\n"
                    f"\tResponse: {error.response}"
                )
            elif isinstance(operator, LLMFilterer):
                error_messages.append(
                    f"Warning: Validation failed for ID: {error.id}, Input: ({error.old_key}, {error.old_value}), Output: {error.new_value}\n"
                    f"\tPrompt: {error.prompt}\n"
                    f"\tResponse: {error.response}"
                )
        if action == ValidatorAction.WARN:
            # Just print
            for error_message in error_messages:
                print(error_message, file=sys.stderr)
        if action == ValidatorAction.FAIL:
            raise ValueError(f"Validation Errors:\n" + "\n".join(error_messages))

    return []
