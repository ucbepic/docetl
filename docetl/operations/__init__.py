import importlib.metadata

def get_operation(operation_type: str):
    try:
        entrypoint = importlib.metadata.entry_points(
            group="docetl.operation"
        )[operation_type]
    except KeyError as e:
        raise KeyError(f"Unrecognized operation {operation_type}")
    return entrypoint.load()
 
