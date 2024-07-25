# from concurrent.futures import ThreadPoolExecutor, as_completed
# import sys
# import json
# from typing import List, Any, Dict, Tuple, Union
# from collections import defaultdict
# import uuid

# from motion.types import *
# from motion.operators import (
#     KeyResolver,
#     Operator,
#     LLMMapper,
#     LLMReducer,
#     LLMPairwiseKeyResolver,
#     LLMListKeyResolver,
#     LLMFlatMapper,
#     LLMParallelFlatMapper,
#     LLMFilterer,
# )


# class Operation:
#     def __init__(
#         self,
#         operator: Union[LLMMapper, LLMReducer, KeyResolver, LLMFlatMapper, LLMFilterer],
#     ):
#         self.operator = operator
#         self._is_optimized = False

#     @property
#     def is_optimized(self) -> bool:
#         return self._is_optimized

#     @is_optimized.setter
#     def is_optimized(self, value: bool) -> None:
#         self._is_optimized = value


# def chunk_data(
#     data: List[Tuple[Any, Any]], chunk_size: int
# ) -> List[List[Tuple[Any, Any]]]:
#     return [data[i : i + chunk_size] for i in range(0, len(data), chunk_size)]


# def apply_operation(
#     data: List[Tuple[Any, Any]],
#     operation: Operation,
#     num_workers: int,
#     building: bool = False,
# ) -> Union[
#     List[Tuple[Any, Any]],
#     Tuple[List[Tuple[Any, Any]], List[Tuple[str, Any, Any]]],
# ]:
#     if isinstance(operation.operator, LLMFlatMapper) or isinstance(
#         operation.operator, LLMParallelFlatMapper
#     ):
#         chunk_size = max(len(data) // num_workers, 1)
#         chunks = chunk_data(data, chunk_size)

#         with ThreadPoolExecutor(max_workers=num_workers) as executor:
#             futures = [
#                 executor.submit(flatmap_worker, chunk, operation) for chunk in chunks
#             ]

#             processed_data = []
#             errors = []
#             for future in futures:
#                 result, error = future.result()
#                 processed_data.extend(result)
#                 errors.extend(error)

#     elif isinstance(operation.operator, LLMMapper):
#         chunk_size = max(len(data) // num_workers, 1)
#         chunks = chunk_data(data, chunk_size)

#         with ThreadPoolExecutor(max_workers=num_workers) as executor:
#             futures = [
#                 executor.submit(map_worker, chunk, operation) for chunk in chunks
#             ]

#             processed_data = []
#             errors = []
#             for future in futures:
#                 result, error = future.result()
#                 processed_data.extend(result)
#                 errors.extend(error)

#     elif isinstance(operation.operator, LLMFilterer):
#         chunk_size = max(len(data) // num_workers, 1)
#         chunks = chunk_data(data, chunk_size)

#         with ThreadPoolExecutor(max_workers=num_workers) as executor:
#             futures = [
#                 executor.submit(filter_worker, chunk, operation) for chunk in chunks
#             ]

#             processed_data = []
#             errors = []
#             results = [future.result() for future in futures]
#             for result, error in results:
#                 processed_data.extend(result)
#                 errors.extend(error)

#     elif isinstance(operation.operator, LLMReducer):
#         processed_data, errors = reduce_worker(data, operation, num_workers)

#     elif isinstance(operation.operator, KeyResolver):
#         processed_data, errors = resolve_keys_worker(data, operation)

#     else:
#         raise ValueError(f"Unsupported operator type: {type(operation.operator)}")

#     if building:
#         return processed_data, errors

#     corrected_data = handle_validation_errors(
#         errors, operation.operator.on_fail, operation.operator
#     )

#     processed_data_dict = {}
#     for item in processed_data:
#         if isinstance(item, OpOutput):
#             processed_data_dict[item.id] = (item.new_key, item.new_value)
#         elif isinstance(item, OpFlatOutput):
#             processed_data_dict[item.id] = (None, item.new_key_value_pairs)
#         elif isinstance(item, OpParallelFlatOutput):
#             processed_data_dict[item.id] = (None, item.new_key_value_pairs)
#         elif isinstance(item, OpFilterOutput):
#             processed_data_dict[item.id] = (item.new_key, item.new_value, item.filter)
#         else:
#             raise ValueError(f"Unsupported Op Output type: {type(item)}")

#     for corrected_output in corrected_data:
#         if isinstance(item, OpFilterOutput):
#             processed_data_dict[corrected_output.id] = (
#                 processed_data_dict[corrected_output.id].key,
#                 processed_data_dict[corrected_output.id].value,
#                 corrected_output.value,
#             )

#         else:
#             processed_data_dict[corrected_output.id] = (
#                 corrected_output.key,
#                 corrected_output.value,
#             )

#     # Flatten flatmapper
#     if isinstance(operation.operator, LLMFlatMapper) or isinstance(
#         operation.operator, LLMParallelFlatMapper
#     ):
#         return [(k, v) for value in processed_data_dict.values() for k, v in value[1]]

#     # Apply filter
#     elif isinstance(operation.operator, LLMFilterer):
#         return [
#             (k, v) for k, v, keep in processed_data_dict.values() if keep is not False
#         ]

#     return list(processed_data_dict.values())


# def map_worker(
#     chunk: List[Tuple[Any, Any]],
#     operation: Operation,
# ) -> Tuple[List[OpOutput], List[OpError]]:
#     results, errors = [], []
#     for key, value in chunk:
#         record_id = str(uuid.uuid4())
#         result, p_and_r = operation.operator.execute(key, value)
#         new_key, new_value = result

#         results.append(
#             OpOutput(
#                 id=record_id,
#                 prompt=p_and_r["prompt"],
#                 response=p_and_r["response"],
#                 new_key=new_key,
#                 new_value=new_value,
#             )
#         )
#         if not operation.operator.validate(key, value, new_key, new_value):
#             errors.append(
#                 OpError(
#                     id=record_id,
#                     old_key=key,
#                     old_value=value,
#                     prompt=p_and_r["prompt"],
#                     response=p_and_r["response"],
#                     new_key=new_key,
#                     new_value=new_value,
#                 )
#             )
#     return results, errors


# def filter_worker(
#     chunk: List[Tuple[Any, Any]],
#     operation: Operation,
# ) -> Tuple[List[OpFilterOutput], List[OpError]]:
#     results, errors = [], []
#     for key, value in chunk:
#         record_id = str(uuid.uuid4())
#         result, p_and_r = operation.operator.execute(key, value)

#         results.append(
#             OpFilterOutput(
#                 id=record_id,
#                 prompt=p_and_r["prompt"],
#                 response=p_and_r["response"],
#                 new_key=key,
#                 new_value=value,
#                 filter=result,
#             )
#         )
#         if not operation.operator.validate(key, value, result):
#             errors.append(
#                 OpError(
#                     id=record_id,
#                     old_key=key,
#                     old_value=value,
#                     prompt=p_and_r["prompt"],
#                     response=p_and_r["response"],
#                     new_key=key,
#                     new_value=result,
#                 )
#             )
#     return results, errors


# def flatmap_worker(
#     data: List[OpInput],
#     operation: Operation,
# ) -> Union[
#     Tuple[OpFlatOutput, OpFlatError], Tuple[OpParallelFlatOutput, OpParallelFlatError]
# ]:
#     output_type = (
#         OpParallelFlatOutput
#         if isinstance(operation.operator, LLMParallelFlatMapper)
#         else OpFlatOutput
#     )
#     error_type = (
#         OpParallelFlatError
#         if isinstance(operation.operator, LLMParallelFlatMapper)
#         else OpFlatError
#     )
#     result, errors = [], []

#     for key, value in data:
#         record_id = str(uuid.uuid4())
#         mapped_kv_pairs, p_and_r = operation.operator.execute(key, value)

#         result.append(
#             output_type(id=record_id, **p_and_r, new_key_value_pairs=mapped_kv_pairs)
#         )

#         if not operation.operator.validate(key, value, mapped_kv_pairs):
#             if isinstance(operation.operator, LLMParallelFlatMapper):
#                 errors.append(
#                     error_type(
#                         id=record_id,
#                         old_key=key,
#                         old_value=value,
#                         new_key_value_pairs=mapped_kv_pairs,
#                         prompts=p_and_r["prompts"],
#                         responses=p_and_r["responses"],
#                     )
#                 )
#             elif isinstance(operation.operator, LLMFlatMapper):
#                 errors.append(
#                     error_type(
#                         id=record_id,
#                         old_key=key,
#                         old_value=value,
#                         new_key_value_pairs=mapped_kv_pairs,
#                         prompt=p_and_r["prompt"],
#                         response=p_and_r["response"],
#                     )
#                 )

#     return result, errors


# def reduce_worker(
#     data: List[OpReduceInput],
#     operation: Operation,
#     num_workers: int,
# ) -> Tuple[List[OpOutput], List[OpReduceError]]:
#     result = []
#     errors = []
#     grouped_data = defaultdict(list)
#     for key, value in data:
#         grouped_data[key].append(value)

#     def process_key(key, values, operation):
#         record_id = str(uuid.uuid4())
#         reduced_value, p_and_r = operation.operator.execute(key, list(values))
#         is_valid = operation.operator.validate(key, list(values), reduced_value)
#         return record_id, key, reduced_value, p_and_r, is_valid

#     with ThreadPoolExecutor(max_workers=num_workers) as executor:
#         futures = []
#         for key, values in grouped_data.items():
#             futures.append(executor.submit(process_key, key, values, operation))

#         for future in as_completed(futures):
#             record_id, key, reduced_value, p_and_r, is_valid = future.result()

#             result.append(
#                 OpOutput(
#                     id=record_id,
#                     prompt=p_and_r["prompt"],
#                     response=p_and_r["response"],
#                     new_key=key,
#                     new_value=reduced_value,
#                 )
#             )

#             if not is_valid:
#                 errors.append(
#                     OpReduceError(
#                         id=record_id,
#                         old_key=key,
#                         old_values=list(values),
#                         prompt=p_and_r["prompt"],
#                         response=p_and_r["response"],
#                         new_value=reduced_value,
#                     )
#                 )

#     return result, errors


# def resolve_keys_worker(
#     data: List[OpInput],
#     operation: Operation,
# ) -> Tuple[List[OpOutput], List[OpError]]:
#     groups: Dict[K, List[Tuple[str, K, V]]] = defaultdict(list)
#     resolved_data = []
#     errors = []

#     # First pass: initial grouping
#     for key, value in data:
#         record_id = str(uuid.uuid4())
#         groups[key].append((record_id, key, value))

#     # Second pass: merge groups based on equality or assign keys
#     final_groups: Dict[str, List[Tuple[str, K, V, Dict]]] = {}
#     for key, records in groups.items():
#         eligible_keys = [
#             final_key
#             for final_key in final_groups.keys()
#             if operation.operator.precheck(key, final_key)
#         ]
#         merged = False
#         p_and_r = {"prompt": None, "response": None}
#         if isinstance(operation.operator, LLMListKeyResolver):
#             new_label, p_and_r = operation.operator.execute(key, list(eligible_keys))
#             if new_label != key:
#                 merged = True
#                 final_groups[new_label].extend(
#                     [(r[0], r[1], r[2], p_and_r) for r in records]
#                 )

#         elif isinstance(operation.operator, LLMPairwiseKeyResolver):
#             for final_key in eligible_keys:
#                 is_equal, p_and_r = operation.operator.execute(final_key, key)
#                 if is_equal:
#                     merged = True
#                     merged_keys = {r[1] for r in final_groups[final_key]} | {key}
#                     new_label = operation.operator.get_label_key(merged_keys)
#                     if new_label != final_key:
#                         # If the label changed, we need to update the dictionary key
#                         final_groups[new_label] = final_groups.pop(final_key)
#                     final_groups[new_label].extend(
#                         [(r[0], r[1], r[2], p_and_r) for r in records]
#                     )
#                     break

#         else:
#             raise ValueError(
#                 "Unsupported operator type: {}".format(type(operation.operator))
#             )

#         if not merged:
#             final_groups[key] = [(r[0], r[1], r[2], p_and_r) for r in records]

#     # Third pass: create output and check for errors
#     for final_key, records in final_groups.items():
#         for record_id, original_key, value, p_and_r in records:
#             resolved_data.append(
#                 OpOutput(
#                     id=record_id,
#                     prompt=p_and_r["prompt"],
#                     response=p_and_r["response"],
#                     new_key=final_key,
#                     new_value=value,
#                 )
#             )
#             if not operation.operator.validate(original_key, final_key):
#                 errors.append(
#                     OpError(
#                         id=record_id,
#                         old_key=original_key,
#                         old_value=value,
#                         prompt=p_and_r["prompt"],
#                         response=p_and_r["response"],
#                         new_key=final_key,
#                         new_value=value,
#                     )
#                 )

#     return resolved_data, errors


# def handle_validation_errors(
#     errors: List[Union[OpError, OpFlatError, OpParallelFlatError, OpReduceError]],
#     action: ValidatorAction,
#     operator: Operator,
# ) -> List[CorrectedOutput]:
#     if not errors:
#         return []

#     if action == ValidatorAction.PROMPT:
#         print("Validation Errors:", file=sys.stderr)
#         for error in errors:
#             if isinstance(operator, LLMParallelFlatMapper):
#                 (
#                     record_id,
#                     input_key,
#                     input_value,
#                     prompts,
#                     responses,
#                     output_kv_pairs,
#                 ) = error
#                 print(
#                     f"  - ID: {record_id}, Input: ({input_key}, {input_value})",
#                     file=sys.stderr,
#                 )
#                 for i, (prompt, response) in enumerate(zip(prompts, responses)):
#                     print(f"    Prompt {i+1}: {prompt}", file=sys.stderr)
#                     print(f"    Response {i+1}: {response}", file=sys.stderr)
#                 for output_key, output_value in output_kv_pairs:
#                     print(
#                         f"    Output: ({output_key}, {output_value})",
#                         file=sys.stderr,
#                     )
#             elif isinstance(operator, LLMFlatMapper):
#                 record_id, input_key, input_value, prompt, response, output_kv_pairs = (
#                     error
#                 )
#                 print(
#                     f"  - ID: {record_id}, Input: ({input_key}, {input_value})",
#                     file=sys.stderr,
#                 )
#                 print(f"    Prompt: {prompt}", file=sys.stderr)
#                 print(f"    Response: {response}", file=sys.stderr)
#                 for output_key, output_value in output_kv_pairs:
#                     print(
#                         f"    Output: ({output_key}, {output_value})",
#                         file=sys.stderr,
#                     )
#             elif isinstance(operator, LLMMapper):
#                 (
#                     record_id,
#                     input_key,
#                     input_value,
#                     prompt,
#                     response,
#                     output_key,
#                     output_value,
#                 ) = error
#                 print(
#                     f"  - ID: {record_id}, Input: ({input_key}, {input_value}), Output: ({output_key}, {output_value})",
#                     file=sys.stderr,
#                 )
#                 print(f"    Prompt: {prompt}", file=sys.stderr)
#                 print(f"    Response: {response}", file=sys.stderr)
#             elif isinstance(operator, LLMReducer):
#                 record_id, key, input_values, prompt, response, output_value = error
#                 print(
#                     f"  - ID: {record_id}, Key: {key}, Input Values: {input_values}, Output: {output_value}",
#                     file=sys.stderr,
#                 )
#                 print(f"    Prompt: {prompt}", file=sys.stderr)
#                 print(f"    Response: {response}", file=sys.stderr)
#             elif isinstance(operator, KeyResolver):
#                 record_id, input_key, output_key, prompt, response = error
#                 print(
#                     f"  - ID: {record_id}, Input Key: {input_key}, Resolved Key: {output_key}",
#                     file=sys.stderr,
#                 )
#                 print(f"    Prompt: {prompt}", file=sys.stderr)
#                 print(f"    Response: {response}", file=sys.stderr)
#             elif isinstance(operator, LLMFilterer):
#                 record_id, input_key, input_value, prompt, response, output = error
#                 print(
#                     f"  - ID: {record_id}, Input: ({input_key}, {input_value})",
#                     file=sys.stderr,
#                 )
#                 print(f"    Prompt: {prompt}", file=sys.stderr)
#                 print(f"    Response: {response}", file=sys.stderr)
#                 print(f"    Filtered: {'True' if output else 'False'}", file=sys.stderr)

#         print(
#             "\nEnter corrections as a JSON dictionary mapping ID to [new_key, new_value] for Mapper, [[new_key1, new_value1], [new_key2, new_value2], ...] for FlatMapper, [new_value] for Reducer, or [new_key] for KeyResolver, or press Enter to skip:",
#             file=sys.stderr,
#         )
#         try:
#             user_input = sys.stdin.readline().strip()
#             if user_input:
#                 corrections = json.loads(user_input)
#                 corrected_errors = []
#                 for error in errors:
#                     record_id = error.id
#                     if record_id in corrections:
#                         if isinstance(operator, LLMFlatMapper):
#                             input_key, input_value = error.old_key, error.old_value
#                             new_kv_pairs = corrections[record_id]
#                             corrected_kv_pairs = operator.correct(
#                                 input_key, input_value, new_kv_pairs
#                             )
#                             corrected_errors.append(
#                                 CorrectedOutput(record_id, None, corrected_kv_pairs)
#                             )
#                             continue
#                         elif isinstance(operator, LLMMapper):
#                             input_key, input_value = error.old_key, error.old_value
#                             new_key, new_value = corrections[record_id]
#                             corrected_key, corrected_value = operator.correct(
#                                 input_key, input_value, new_key, new_value
#                             )
#                         elif isinstance(operator, LLMReducer):
#                             key, input_values = error.old_key, error.old_values
#                             new_value = corrections[record_id][0]
#                             corrected_key, corrected_value = operator.correct(
#                                 key, input_values, new_value
#                             )
#                         elif isinstance(operator, KeyResolver):
#                             input_key, output_key = error.old_key, error.new_key
#                             new_key = corrections[record_id][0]
#                             corrected_key = operator.correct(input_key, new_key)
#                             corrected_value = output_key  # Keep the original value
#                         elif isinstance(operator, LLMFilterer):
#                             input_key, input_value = error.old_key, error.old_value
#                             new_output = corrections[record_id][0]
#                             corrected_key, corrected_value = input_key, input_value
#                             operator.correct(input_key, input_value, new_output)
#                         corrected_errors.append(
#                             CorrectedOutput(record_id, corrected_key, corrected_value)
#                         )
#                 return corrected_errors
#             return []
#         except json.JSONDecodeError:
#             print("Invalid JSON input. Skipping corrections.", file=sys.stderr)
#         except KeyboardInterrupt:
#             print("\nOperation aborted by user.", file=sys.stderr)
#             sys.exit(1)
#     elif action == ValidatorAction.WARN or action == ValidatorAction.FAIL:
#         error_messages = []
#         for error in errors:
#             if isinstance(operator, LLMParallelFlatMapper):
#                 prompts_responses = "\n".join(
#                     [
#                         f"\tPrompt {i+1}: {prompt}\n\tResponse {i+1}: {response}"
#                         for i, (prompt, response) in enumerate(
#                             zip(error.prompts, error.responses)
#                         )
#                     ]
#                 )
#                 error_messages.append(
#                     f"Warning: Validation failed for ID: {error.id}, Input: ({error.old_key}, {error.old_value}), Output: {error.new_key_value_pairs}\n"
#                     f"{prompts_responses}"
#                 )
#             elif isinstance(operator, LLMFlatMapper):
#                 error_messages.append(
#                     f"Warning: Validation failed for ID: {error.id}, Input: ({error.old_key}, {error.old_value}), Output: {error.new_key_value_pairs}\n"
#                     f"\tPrompt: {error.prompt}\n"
#                     f"\tResponse: {error.response}"
#                 )
#             elif isinstance(operator, LLMMapper):
#                 error_messages.append(
#                     f"Warning: Validation failed for ID: {error.id}, Input: ({error.old_key}, {error.old_value}), Output: ({error.new_key}, {error.new_value})\n"
#                     f"\tPrompt: {error.prompt}\n"
#                     f"\tResponse: {error.response}"
#                 )
#             elif isinstance(operator, LLMReducer):
#                 error_messages.append(
#                     f"Warning: Validation failed for ID: {error.id}, Key: {error.old_key}, Input Values: {error.old_values}, Output: {error.new_value}\n"
#                     f"\tPrompt: {error.prompt}\n"
#                     f"\tResponse: {error.response}"
#                 )
#             elif isinstance(operator, KeyResolver):
#                 error_messages.append(
#                     f"Warning: Validation failed for ID: {error.id}, Input Key: {error.old_key}, Resolved Key: {error.new_key}\n"
#                     f"\tPrompt: {error.prompt}\n"
#                     f"\tResponse: {error.response}"
#                 )
#             elif isinstance(operator, LLMFilterer):
#                 error_messages.append(
#                     f"Warning: Validation failed for ID: {error.id}, Input: ({error.old_key}, {error.old_value}), Output: {error.new_value}\n"
#                     f"\tPrompt: {error.prompt}\n"
#                     f"\tResponse: {error.response}"
#                 )
#         if action == ValidatorAction.WARN:
#             # Just print
#             for error_message in error_messages:
#                 print(error_message, file=sys.stderr)
#         if action == ValidatorAction.FAIL:
#             raise ValueError(f"Validation Errors:\n" + "\n".join(error_messages))

#     return []
