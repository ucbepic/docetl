import random
import sys
import json
import multiprocessing
from typing import List, Any, Dict, Tuple, Optional, Union, Iterable, Set
from collections import defaultdict
import uuid

from motion.types import K, V, ValidatorAction
from motion.operators import (
    Mapper,
    Reducer,
    KeyResolver,
    Operator,
    FlatMapper,
    Filterer,
)


class Operation:
    def __init__(
        self, operator: Union[Mapper, Reducer, KeyResolver, FlatMapper, Filterer]
    ):
        self.operator = operator


class Dataset:
    def __init__(self, data: Iterable[Tuple[str, Any]], num_workers: int = None):
        for item in data[:10]:
            if not isinstance(item, tuple) or len(item) != 2:
                raise ValueError("Each item must be a tuple of (K, V)")

        self.data = list(data)
        self.num_workers = num_workers or multiprocessing.cpu_count()
        self.operations: List[Operation] = []
        self.optimized_operations: List[Operation] = []

    def map(self, mapper: Mapper) -> "Dataset":
        self.operations.append(Operation(mapper))
        return self

    def flat_map(self, flatmapper: FlatMapper) -> "Dataset":
        self.operations.append(Operation(flatmapper))
        return self

    def resolve_keys(self, key_resolver: KeyResolver) -> "Dataset":
        self.operations.append(Operation(key_resolver))
        return self

    def reduce(self, reducer: Reducer) -> "Dataset":
        self.operations.append(Operation(reducer))
        return self

    def filter(self, filterer: Filterer) -> "Dataset":
        self.operations.append(Operation(filterer))
        return self

    def build(self, sample_size: int = 1000) -> "Dataset":
        # Sample the data
        sample_data = random.sample(self.data, min(sample_size, len(self.data)))

        optimized_operations = []

        for operation in self.operations:
            # Apply the operation to the sample data
            try:
                result, errors = self._apply_operation(sample_data, operation)

                if errors:
                    # If there are validation errors, attempt to optimize the operation
                    optimized_operation = optimize(operation, sample_data, errors)
                    optimized_operations.append(optimized_operation)
                    sample_data = result
                else:
                    # If no errors, keep the original operation
                    optimized_operations.append(operation)
                    sample_data = result

            except Exception as e:
                print(
                    f"Error applying operation: {str(e)}. Keeping original operation."
                )
                optimized_operations.append(operation)

        # Update the operations with the optimized version
        self.optimized_operations = optimized_operations

        return self

    def execute(self) -> List[Tuple[str, Any, Any]]:
        ops = (
            self.optimized_operations if self.optimized_operations else self.operations
        )

        current_data = self.data
        for operation in ops:
            current_data = self._apply_operation(current_data, operation)
        return current_data

    def _chunk_data(
        self, data: List[Tuple[str, Any, Any]], chunk_size: int
    ) -> List[List[Tuple[str, Any, Any]]]:
        return [data[i : i + chunk_size] for i in range(0, len(data), chunk_size)]

    def _apply_operation(
        self,
        data: List[Tuple[str, Any, Any]],
        operation: Operation,
    ) -> List[Tuple[str, Any, Any]]:
        if isinstance(operation.operator, FlatMapper):
            chunk_size = max(len(data) // self.num_workers, 1)
            chunks = self._chunk_data(data, chunk_size)

            with multiprocessing.Pool(self.num_workers) as pool:
                results = pool.starmap(
                    self._flatmap_worker, [(chunk, operation) for chunk in chunks]
                )

            processed_data = []
            errors = []
            for result, error in results:
                processed_data.extend(result)
                errors.extend(error)

        elif isinstance(operation.operator, Mapper):
            chunk_size = max(len(data) // self.num_workers, 1)
            chunks = self._chunk_data(data, chunk_size)

            with multiprocessing.Pool(self.num_workers) as pool:
                results = pool.starmap(
                    self._map_worker, [(chunk, operation) for chunk in chunks]
                )

            processed_data = []
            errors = []
            for result, error in results:
                processed_data.extend(result)
                errors.extend(error)

        elif isinstance(operation.operator, Filterer):
            chunk_size = max(len(data) // self.num_workers, 1)
            chunks = self._chunk_data(data, chunk_size)

            with multiprocessing.Pool(self.num_workers) as pool:
                results = pool.starmap(
                    self._filter_worker, [(chunk, operation) for chunk in chunks]
                )

            processed_data = []
            errors = []
            for result, error in results:
                processed_data.extend(result)
                errors.extend(error)

        elif isinstance(operation.operator, Reducer):
            processed_data, errors = self._reduce_worker(data, operation)

        elif isinstance(operation.operator, KeyResolver):
            processed_data, errors = self._resolve_keys_worker(data, operation)

        else:
            raise ValueError(f"Unsupported operator type: {type(operation.operator)}")

        corrected_data = self._handle_validation_errors(
            errors, operation.operator.on_fail, operation.operator
        )

        processed_data_dict = {
            record_id: (key, value) for record_id, key, value in processed_data
        }

        for record_id, key, value in corrected_data:
            if isinstance(operation.operator, Filterer):
                if value is not False:
                    processed_data_dict[record_id] = (key, value)
            else:
                processed_data_dict[record_id] = (key, value)

        # Flatten flatmapper
        if isinstance(operation.operator, FlatMapper):
            return [
                (k, v) for value in processed_data_dict.values() for k, v in value[1]
            ]

        return list(processed_data_dict.values())

    def _map_worker(
        self,
        chunk: List[Tuple[str, Any, Any]],
        operation: Operation,
    ) -> Tuple[List[Tuple[str, Any, Any]], List[Tuple[str, Any, Any, Any, Any]]]:
        result = []
        errors = []
        for key, value in chunk:
            record_id = str(uuid.uuid4())
            mapped_key, mapped_value = operation.operator.map(key, value)
            result.append((record_id, mapped_key, mapped_value))
            if not operation.operator.validate(key, value, mapped_key, mapped_value):
                errors.append((record_id, key, value, mapped_key, mapped_value))
        return result, errors

    def _filter_worker(
        self,
        chunk: List[Tuple[str, Any, Any]],
        operation: Operation,
    ) -> Tuple[List[Tuple[str, Any, Any]], List[Tuple[str, Any, Any, Any, Any]]]:
        result = []
        errors = []
        for key, value in chunk:
            record_id = str(uuid.uuid4())
            if operation.operator.filter(key, value):
                result.append((record_id, key, value))
            else:
                errors.append((record_id, key, value))
        return result, errors

    def _flatmap_worker(
        self,
        data: List[Tuple[str, Any, Any]],
        operation: Operation,
    ) -> Tuple[
        List[Tuple[str, Any, Any]], List[Tuple[str, Any, Any, List[Tuple[Any, Any]]]]
    ]:
        result = []
        errors = []
        for key, value in data:
            record_id = str(uuid.uuid4())
            mapped_kv_pairs = operation.operator.map(key, value)
            result.append((record_id, None, mapped_kv_pairs))
            if not operation.operator.validate(key, value, mapped_kv_pairs):
                errors.append((record_id, key, value, mapped_kv_pairs))
        return result, errors

    def _reduce_worker(
        self,
        data: List[Tuple[str, Any, Any]],
        operation: Operation,
    ) -> Tuple[List[Tuple[str, Any, Any]], List[Tuple[str, Any, List[Any], Any]]]:
        result = []
        errors = []
        grouped_data = defaultdict(list)
        for key, value in data:
            grouped_data[key].append(value)

        for key, values in grouped_data.items():
            reduced_value = operation.operator.reduce(key, list(values))
            new_id = str(uuid.uuid4())
            result.append((new_id, key, reduced_value))
            if not operation.operator.validate(key, list(values), reduced_value):
                errors.append((new_id, key, list(values), reduced_value))
        return result, errors

    def _resolve_keys_worker(
        self,
        data: List[Tuple[str, Any, Any]],
        operation: Operation,
    ) -> Tuple[List[Tuple[str, Any, Any]], List[Tuple[str, Any, Any]]]:
        groups: Dict[str, List[Tuple[str, str, Any]]] = defaultdict(list)
        resolved_data = []
        errors = []

        # First pass: initial grouping
        for key, value in data:
            record_id = str(uuid.uuid4())
            groups[key].append((record_id, key, value))

        # Second pass: merge groups based on equality or assign keys
        final_groups: Dict[str, List[Tuple[str, str, Any]]] = {}
        for key, records in groups.items():
            merged = False
            eligible_keys = [
                final_key
                for final_key in final_groups.keys()
                if operation.operator.precheck(key, final_key)
            ]
            for final_key in eligible_keys:
                if operation.operator._use_are_equal:
                    if operation.operator.are_equal(final_key, key):
                        merged = True
                        merged_keys = {r[1] for r in final_groups[final_key]} | {key}
                        new_label = operation.operator.get_label_key(merged_keys)
                        if new_label != final_key:
                            # If the label changed, we need to update the dictionary key
                            final_groups[new_label] = final_groups.pop(final_key)
                        final_groups[new_label].extend(records)
                        break
                else:
                    new_label = operation.operator.assign_keys(
                        key, list(final_groups.keys())
                    )
                    if new_label != key:
                        merged = True
                        final_groups[new_label].extend(records)
                        break
            if not merged:
                final_groups[key] = records

        # Third pass: create output and check for errors
        for final_key, records in final_groups.items():
            for record_id, original_key, value in records:
                resolved_data.append((record_id, final_key, value))
                if not operation.operator.validate(original_key, final_key):
                    errors.append((record_id, original_key, final_key))

        return resolved_data, errors

    def _handle_validation_errors(
        self,
        errors: List[
            Union[
                Tuple[str, Any, Any, Any, Any],
                Tuple[str, Any, List[Any], Any],
                Tuple[str, Any, Any],
            ]
        ],
        action: ValidatorAction,
        operator: Operator,
    ) -> List[Tuple[str, Any, Any]]:
        if not errors:
            return []

        if action == ValidatorAction.PROMPT:
            print("Validation Errors:", file=sys.stderr)
            for error in errors:
                if isinstance(operator, FlatMapper):
                    record_id, input_key, input_value, output_kv_pairs = error
                    print(
                        f"  - ID: {record_id}, Input: ({input_key}, {input_value})",
                        file=sys.stderr,
                    )
                    for output_key, output_value in output_kv_pairs:
                        print(
                            f"    Output: ({output_key}, {output_value})",
                            file=sys.stderr,
                        )

                elif isinstance(operator, Mapper):
                    record_id, input_key, input_value, output_key, output_value = error
                    print(
                        f"  - ID: {record_id}, Input: ({input_key}, {input_value}), Output: ({output_key}, {output_value})",
                        file=sys.stderr,
                    )
                elif isinstance(operator, Reducer):
                    record_id, key, input_values, output_value = error
                    print(
                        f"  - ID: {record_id}, Key: {key}, Input Values: {input_values}, Output: {output_value}",
                        file=sys.stderr,
                    )
                elif isinstance(operator, KeyResolver):
                    record_id, input_key, output_key = error
                    print(
                        f"  - ID: {record_id}, Input Key: {input_key}, Resolved Key: {output_key}",
                        file=sys.stderr,
                    )
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
                        record_id = error[0]
                        if record_id in corrections:
                            if isinstance(operator, FlatMapper):
                                input_key, input_value, _, _ = error[1:]
                                new_kv_pairs = corrections[record_id]
                                corrected_kv_pairs = operator.correct(
                                    input_key, input_value, new_kv_pairs
                                )
                                corrected_errors.append(
                                    (record_id, None, corrected_kv_pairs)
                                )
                                continue
                            elif isinstance(operator, Mapper):
                                input_key, input_value, _, _ = error[1:]
                                new_key, new_value = corrections[record_id]
                                corrected_key, corrected_value = operator.correct(
                                    input_key, input_value, new_key, new_value
                                )
                            elif isinstance(operator, Reducer):
                                key, input_values, _ = error[1:]
                                new_value = corrections[record_id][0]
                                corrected_key, corrected_value = operator.correct(
                                    key, input_values, new_value
                                )
                            elif isinstance(operator, KeyResolver):
                                input_key, _ = error[1:]
                                new_key = corrections[record_id][0]
                                corrected_key = operator.correct(input_key, new_key)
                                corrected_value = error[2]  # Keep the original value
                            corrected_errors.append(
                                (record_id, corrected_key, corrected_value)
                            )
                    return corrected_errors
                return []
            except json.JSONDecodeError:
                print("Invalid JSON input. Skipping corrections.", file=sys.stderr)
            except KeyboardInterrupt:
                print("\nOperation aborted by user.", file=sys.stderr)
                sys.exit(1)
        elif action == ValidatorAction.WARN:
            for error in errors:
                if isinstance(operator, Mapper):
                    record_id, input_key, input_value, output_key, output_value = error
                    print(
                        f"Warning: Validation failed for ID: {record_id}, Input: ({input_key}, {input_value}), Output: ({output_key}, {output_value})",
                        file=sys.stderr,
                    )
                elif isinstance(operator, Reducer):
                    record_id, key, input_values, output_value = error
                    print(
                        f"Warning: Validation failed for ID: {record_id}, Key: {key}, Input Values: {input_values}, Output: {output_value}",
                        file=sys.stderr,
                    )
                elif isinstance(operator, KeyResolver):
                    record_id, input_key, output_key = error
                    print(
                        f"Warning: Validation failed for ID: {record_id}, Input Key: {input_key}, Resolved Key: {output_key}",
                        file=sys.stderr,
                    )
        elif action == ValidatorAction.FAIL:
            error_messages = []
            for error in errors:
                if isinstance(operator, Mapper):
                    record_id, input_key, input_value, output_key, output_value = error
                    error_messages.append(
                        f"ID: {record_id}, Input: ({input_key}, {input_value}), Output: ({output_key}, {output_value})"
                    )
                elif isinstance(operator, Reducer):
                    record_id, key, input_values, output_value = error
                    error_messages.append(
                        f"ID: {record_id}, Key: {key}, Input Values: {input_values}, Output: {output_value}"
                    )
                elif isinstance(operator, KeyResolver):
                    record_id, input_key, output_key = error
                    error_messages.append(
                        f"ID: {record_id}, Input Key: {input_key}, Resolved Key: {output_key}"
                    )
            raise ValueError(f"Validation Errors:\n" + "\n".join(error_messages))

        return []
