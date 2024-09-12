from docetl.operations.map import MapOperation, ParallelMapOperation
from docetl.operations.filter import FilterOperation
from docetl.operations.unnest import UnnestOperation
from docetl.operations.equijoin import EquijoinOperation
from docetl.operations.split import SplitOperation
from docetl.operations.reduce import ReduceOperation
from docetl.operations.resolve import ResolveOperation
from docetl.operations.gather import GatherOperation


def get_operation(operation_type: str):
    operations = {
        "map": MapOperation,
        "parallel_map": ParallelMapOperation,
        "filter": FilterOperation,
        "unnest": UnnestOperation,
        "equijoin": EquijoinOperation,
        "split": SplitOperation,
        "reduce": ReduceOperation,
        "resolve": ResolveOperation,
        "gather": GatherOperation,
    }
    return operations.get(operation_type)
