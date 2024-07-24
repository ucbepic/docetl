from motion.workers import Operation
from typing import List, Tuple, Any


def optimize(
    operation: Operation,
    sample_data: List[Tuple[Any, Any]],
    errors: List[Tuple[str, Any, int]],
    num_workers: int,
) -> Operation:
    """Optimize the given operator."""
    return operation
