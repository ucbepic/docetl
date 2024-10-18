import importlib.metadata


def get_operation(operation_type: str):
    """Loads a single operation by name""" 
    try:
        entrypoint = importlib.metadata.entry_points(group="docetl.operation")[
            operation_type
        ]
    except KeyError as e:
        raise KeyError(f"Unrecognized operation {operation_type}")
    return entrypoint.load()

def get_operations():
    """Load all available operations and return them as a dictionary"""
    return {
        name: op.load()
        for name, op in importlib.metadata.entry_points(group="docetl.operation").items()}
