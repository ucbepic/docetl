from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Union, Any
import uuid
from collections import defaultdict
from motion.types import *
from motion.operators import (
    LLMFlatMapper,
    LLMParallelFlatMapper,
    LLMPairwiseKeyResolver,
    LLMListKeyResolver,
)
from motion.executor.operation import Operation
from tqdm import tqdm


def map_worker(
    chunk: List[Tuple[Any, Any]],
    operation: Operation,
) -> Tuple[List[OpOutput], List[OpError]]:
    results, errors = [], []
    for key, value in chunk:
        record_id = str(uuid.uuid4())
        result, p_and_r = operation.operator.execute(key, value)
        new_key, new_value = result

        results.append(
            OpOutput(
                id=record_id,
                prompt=p_and_r.get("prompt"),
                response=p_and_r.get("response"),
                new_key=new_key,
                new_value=new_value,
            )
        )
        try:
            operation.operator.validate(key, value, new_key, new_value)
        except Exception as e:
            errors.append(
                OpError(
                    id=record_id,
                    old_key=key,
                    old_value=value,
                    prompt=p_and_r.get("prompt"),
                    response=p_and_r.get("response"),
                    new_key=new_key,
                    new_value=new_value,
                    error_msg=str(e),
                )
            )
    return results, errors


def filter_worker(
    chunk: List[Tuple[Any, Any]],
    operation: Operation,
) -> Tuple[List[OpFilterOutput], List[OpError]]:
    results, errors = [], []
    for key, value in chunk:
        record_id = str(uuid.uuid4())
        result, p_and_r = operation.operator.execute(key, value)

        results.append(
            OpFilterOutput(
                id=record_id,
                prompt=p_and_r["prompt"],
                response=p_and_r["response"],
                new_key=key,
                new_value=value,
                filter=result,
            )
        )
        try:
            operation.operator.validate(key, value, result)
        except Exception as e:
            errors.append(
                OpError(
                    id=record_id,
                    old_key=key,
                    old_value=value,
                    prompt=p_and_r["prompt"],
                    response=p_and_r["response"],
                    new_key=key,
                    new_value=result,
                    error_msg=str(e),
                )
            )
    return results, errors


def flatmap_worker(
    data: List[OpInput],
    operation: Operation,
) -> Union[
    Tuple[OpFlatOutput, OpFlatError], Tuple[OpParallelFlatOutput, OpParallelFlatError]
]:
    output_type = (
        OpParallelFlatOutput
        if isinstance(operation.operator, LLMParallelFlatMapper)
        else OpFlatOutput
    )
    error_type = (
        OpParallelFlatError
        if isinstance(operation.operator, LLMParallelFlatMapper)
        else OpFlatError
    )
    result, errors = [], []

    for key, value in data:
        record_id = str(uuid.uuid4())
        mapped_kv_pairs, p_and_r = operation.operator.execute(key, value)

        result.append(
            output_type(id=record_id, **p_and_r, new_key_value_pairs=mapped_kv_pairs)
        )

        try:
            operation.operator.validate(key, value, mapped_kv_pairs)
        except Exception as e:
            if isinstance(operation.operator, LLMParallelFlatMapper):
                errors.append(
                    error_type(
                        id=record_id,
                        old_key=key,
                        old_value=value,
                        new_key_value_pairs=mapped_kv_pairs,
                        prompts=p_and_r["prompts"],
                        responses=p_and_r["responses"],
                        error_msg=str(e),
                    )
                )
            elif isinstance(operation.operator, LLMFlatMapper):
                errors.append(
                    error_type(
                        id=record_id,
                        old_key=key,
                        old_value=value,
                        new_key_value_pairs=mapped_kv_pairs,
                        prompt=p_and_r["prompt"],
                        response=p_and_r["response"],
                        error_msg=str(e),
                    )
                )

    return result, errors


def split_worker(
    data: List[Tuple[Any, Any]],
    operation: Operation,
) -> Tuple[List[OpFlatOutput], List[OpFlatError]]:
    result = []
    errors = []

    for key, value in data:
        record_id = str(uuid.uuid4())
        split_kv_pairs, _ = operation.operator.execute(key, value)

        result.append(
            OpFlatOutput(
                id=record_id,
                prompt=None,
                response=None,
                new_key_value_pairs=split_kv_pairs,
            )
        )

        try:
            operation.operator.validate(key, value, split_kv_pairs)
        except Exception as e:
            errors.append(
                OpFlatError(
                    id=record_id,
                    old_key=key,
                    old_value=value,
                    new_key_value_pairs=split_kv_pairs,
                    prompt=None,
                    response=None,
                    error_msg=str(e),
                )
            )

    return result, errors


def reduce_worker(
    data: List[OpReduceInput],
    operation: Operation,
    num_workers: int,
) -> Tuple[List[OpOutput], List[OpReduceError]]:
    result = []
    errors = []
    grouped_data = defaultdict(list)
    for key, value in data:
        grouped_data[key].append(value)

    def process_key(key, values, operation):
        record_id = str(uuid.uuid4())
        reduced_value, p_and_r = operation.operator.execute(key, list(values))
        try:
            operation.operator.validate(key, list(values), reduced_value)
            is_valid = True
        except Exception as e:
            is_valid = False
            error_msg = str(e)
        return (
            record_id,
            key,
            reduced_value,
            p_and_r,
            is_valid,
            error_msg if not is_valid else None,
        )

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for key, values in grouped_data.items():
            futures.append(executor.submit(process_key, key, values, operation))

        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Reducing..."
        ):
            record_id, key, reduced_value, p_and_r, is_valid, error_msg = (
                future.result()
            )

            result.append(
                OpOutput(
                    id=record_id,
                    prompt=p_and_r["prompt"],
                    response=p_and_r["response"],
                    new_key=key,
                    new_value=reduced_value,
                )
            )

            if not is_valid:
                errors.append(
                    OpReduceError(
                        id=record_id,
                        old_key=key,
                        old_values=list(grouped_data[key]),
                        prompt=p_and_r["prompt"],
                        response=p_and_r["response"],
                        new_value=reduced_value,
                        error_msg=error_msg,
                    )
                )

    return result, errors


def resolve_keys_worker(
    data: List[OpInput],
    operation: Operation,
) -> Tuple[List[OpOutput], List[OpError]]:
    groups: Dict[K, List[Tuple[str, K, V]]] = defaultdict(list)
    resolved_data = []
    errors = []

    # First pass: initial grouping
    for key, value in data:
        record_id = str(uuid.uuid4())
        groups[key].append((record_id, key, value))

    # Second pass: merge groups based on equality or assign keys
    final_groups: Dict[str, List[Tuple[str, K, V, Dict]]] = {}
    for key, records in tqdm(
        groups.items(),
        desc=f"Resolving keys for {operation.operator.__class__.__name__}...",
    ):
        eligible_keys = [
            final_key
            for final_key in final_groups.keys()
            if operation.operator.precheck(key, final_key)
        ]
        merged = False
        p_and_r = {"prompt": None, "response": None}
        if isinstance(operation.operator, LLMListKeyResolver):
            new_label, p_and_r = operation.operator.execute(key, list(eligible_keys))
            if new_label != key:
                merged = True
                final_groups[new_label].extend(
                    [(r[0], r[1], r[2], p_and_r) for r in records]
                )

        elif isinstance(operation.operator, LLMPairwiseKeyResolver):
            for final_key in eligible_keys:
                is_equal, p_and_r = operation.operator.execute(final_key, key)
                if is_equal:
                    merged = True
                    merged_keys = {r[1] for r in final_groups[final_key]} | {key}
                    new_label = operation.operator.get_label_key(merged_keys)
                    if new_label != final_key:
                        # If the label changed, we need to update the dictionary key
                        final_groups[new_label] = final_groups.pop(final_key)
                    final_groups[new_label].extend(
                        [(r[0], r[1], r[2], p_and_r) for r in records]
                    )
                    break

        else:
            raise ValueError(
                "Unsupported operator type: {}".format(type(operation.operator))
            )

        if not merged:
            final_groups[key] = [(r[0], r[1], r[2], p_and_r) for r in records]

    # Third pass: create output and check for errors
    for final_key, records in final_groups.items():
        for record_id, original_key, value, p_and_r in records:
            resolved_data.append(
                OpOutput(
                    id=record_id,
                    prompt=p_and_r["prompt"],
                    response=p_and_r["response"],
                    new_key=final_key,
                    new_value=value,
                )
            )
            try:
                operation.operator.validate(original_key, final_key, value)
            except Exception as e:
                errors.append(
                    OpError(
                        id=record_id,
                        old_key=original_key,
                        old_value=value,
                        prompt=p_and_r["prompt"],
                        response=p_and_r["response"],
                        new_key=final_key,
                        new_value=value,
                        error_msg=str(e),
                    )
                )

    return resolved_data, errors
