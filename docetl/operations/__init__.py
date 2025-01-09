import importlib.metadata
from docetl.operations.cluster import ClusterOperation
from docetl.operations.code_operations import CodeFilterOperation, CodeMapOperation, CodeReduceOperation
from docetl.operations.equijoin import EquijoinOperation
from docetl.operations.filter import FilterOperation
from docetl.operations.gather import GatherOperation
from docetl.operations.map import MapOperation
from docetl.operations.reduce import ReduceOperation
from docetl.operations.resolve import ResolveOperation
from docetl.operations.split import SplitOperation
from docetl.operations.sample import SampleOperation
from docetl.operations.unnest import UnnestOperation
from docetl.operations.scan import ScanOperation

mapping = {
    "cluster": ClusterOperation,
    "code_filter": CodeFilterOperation,
    "code_map": CodeMapOperation,
    "code_reduce": CodeReduceOperation,
    "equijoin": EquijoinOperation,
    "filter": FilterOperation,
    "gather": GatherOperation,
    "map": MapOperation,
    "reduce": ReduceOperation,
    "resolve": ResolveOperation,
    "split": SplitOperation,
    "sample": SampleOperation,
    "unnest": UnnestOperation,
    "scan": ScanOperation
}

def get_operation(operation_type: str):
    """Loads a single operation by name""" 
    try:
        entrypoint = importlib.metadata.entry_points(group="docetl.operation")[
            operation_type
        ]
        return entrypoint.load()
    except KeyError:
        if operation_type in mapping:
            return mapping[operation_type]
        raise KeyError(f"Unrecognized operation {operation_type}")

def get_operations():
    """Load all available operations and return them as a dictionary"""
    operations = mapping.copy()
    operations.update({
        op.name: op.load()
        for op in importlib.metadata.entry_points(group="docetl.operation")
    })
    return operations
