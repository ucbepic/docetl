from typing import List, Tuple, Any


def chunk_data(
    data: List[Tuple[Any, Any]], chunk_size: int
) -> List[List[Tuple[Any, Any]]]:
    return [data[i : i + chunk_size] for i in range(0, len(data), chunk_size)]
