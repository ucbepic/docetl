from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Union, Any
from motion.types import *
from motion.operators import (
    LLMFlatMapper,
    LLMParallelFlatMapper,
    LLMMapper,
    LLMFilterer,
    LLMReducer,
    KeyResolver,
    Splitter,
)
from motion.executor.operation import Operation
from motion.executor.utils import chunk_data
from motion.executor.workers import (
    map_worker,
    filter_worker,
    flatmap_worker,
    reduce_worker,
    resolve_keys_worker,
    split_worker,
)
from motion.executor.validation import handle_validation_errors
from tqdm import tqdm
from litellm import completion_cost


def apply_operation(
    data: List[Tuple[Any, Any]],
    operation: Operation,
    num_workers: int,
    building: bool = False,
) -> Union[
    Tuple[List[Tuple[Any, Any]], float],
    Tuple[List[Tuple[Any, Any]], List[Tuple[str, Any, Any]], float],
]:
    total_cost = 0
    if isinstance(operation.operator, LLMFlatMapper) or isinstance(
        operation.operator, LLMParallelFlatMapper
    ):
        chunk_size = max(len(data) // num_workers, 1)
        chunks = chunk_data(data, chunk_size)

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(flatmap_worker, chunk, operation) for chunk in chunks
            ]

            processed_data = []
            errors = []
            for future in tqdm(
                futures,
                total=len(futures),
                desc=f"Executing {operation.operator.__class__.__name__}...",
            ):
                result, error = future.result()
                processed_data.extend(result)
                errors.extend(error)

    elif isinstance(operation.operator, Splitter):
        chunk_size = max(len(data) // num_workers, 1)
        chunks = chunk_data(data, chunk_size)

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(split_worker, chunk, operation) for chunk in chunks
            ]

            processed_data = []
            errors = []
            for future in tqdm(
                futures,
                total=len(futures),
                desc=f"Executing {operation.operator.__class__.__name__}...",
            ):
                result, error = future.result()
                processed_data.extend(result)
                errors.extend(error)

    elif isinstance(operation.operator, LLMMapper):
        chunk_size = max(len(data) // num_workers, 1)
        chunks = chunk_data(data, chunk_size)

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(map_worker, chunk, operation) for chunk in chunks
            ]

            processed_data = []
            errors = []
            for future in tqdm(
                futures,
                total=len(futures),
                desc=f"Executing {operation.operator.__class__.__name__}...",
            ):
                result, error = future.result()
                processed_data.extend(result)
                errors.extend(error)

    elif isinstance(operation.operator, LLMFilterer):
        chunk_size = max(len(data) // num_workers, 1)
        chunks = chunk_data(data, chunk_size)

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(filter_worker, chunk, operation) for chunk in chunks
            ]

            processed_data = []
            errors = []
            results = [
                future.result()
                for future in tqdm(
                    futures,
                    total=len(futures),
                    desc=f"Executing {operation.operator.__class__.__name__}...",
                )
            ]
            for result, error in results:
                processed_data.extend(result)
                errors.extend(error)

    elif isinstance(operation.operator, LLMReducer):
        processed_data, errors = reduce_worker(data, operation, num_workers)

    elif isinstance(operation.operator, KeyResolver):
        processed_data, errors = resolve_keys_worker(data, operation)

    else:
        raise ValueError(f"Unsupported operator type: {type(operation.operator)}")

    # Compute the cost
    if processed_data:
        if isinstance(processed_data[0], OpParallelFlatOutput):
            total_cost = sum(
                [
                    completion_cost(r)
                    for item in processed_data
                    for r in item.responses
                    if item.responses and r
                ]
            )
        else:
            total_cost = sum(
                [
                    completion_cost(item.response)
                    for item in processed_data
                    if item.response
                ]
            )

    if building:
        return processed_data, errors, total_cost

    corrected_data = handle_validation_errors(
        errors, operation.operator.on_fail, operation.operator
    )

    processed_data_dict = {}
    for item in processed_data:
        if isinstance(item, OpOutput):
            processed_data_dict[item.id] = (item.new_key, item.new_value)
        elif isinstance(item, OpFlatOutput):
            processed_data_dict[item.id] = (None, item.new_key_value_pairs)
        elif isinstance(item, OpParallelFlatOutput):
            processed_data_dict[item.id] = (None, item.new_key_value_pairs)
        elif isinstance(item, OpFilterOutput):
            processed_data_dict[item.id] = (item.new_key, item.new_value, item.filter)
        else:
            raise ValueError(f"Unsupported Op Output type: {type(item)}")

    for corrected_output in corrected_data:
        if isinstance(item, OpFilterOutput):
            processed_data_dict[corrected_output.id] = (
                processed_data_dict[corrected_output.id].key,
                processed_data_dict[corrected_output.id].value,
                corrected_output.value,
            )

        else:
            processed_data_dict[corrected_output.id] = (
                corrected_output.key,
                corrected_output.value,
            )

    # Flatten flatmapper
    if (
        isinstance(operation.operator, LLMFlatMapper)
        or isinstance(operation.operator, LLMParallelFlatMapper)
        or isinstance(operation.operator, Splitter)
    ):
        return [
            (k, v) for value in processed_data_dict.values() for k, v in value[1]
        ], total_cost

    # Apply filter
    elif isinstance(operation.operator, LLMFilterer):
        return [
            (k, v) for k, v, keep in processed_data_dict.values() if keep is not False
        ], total_cost

    return list(processed_data_dict.values()), total_cost
