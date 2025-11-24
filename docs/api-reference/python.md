# Python API

## Operations

::: docetl.schemas.MapOp
    options:
        show_root_heading: true
        heading_level: 3
        show_if_no_docstring: false
        docstring_options:
            ignore_init_summary: false
            trim_doctest_flags: true

::: docetl.schemas.ResolveOp
    options:
        show_root_heading: true
        heading_level: 3
        show_if_no_docstring: false
        docstring_options:
            ignore_init_summary: false
            trim_doctest_flags: true

::: docetl.schemas.ReduceOp
    options:
        show_root_heading: true
        heading_level: 3
        show_if_no_docstring: false
        docstring_options:
            ignore_init_summary: false
            trim_doctest_flags: true

::: docetl.schemas.ParallelMapOp
    options:
        show_root_heading: true
        heading_level: 3
        show_if_no_docstring: false
        docstring_options:
            ignore_init_summary: false
            trim_doctest_flags: true

::: docetl.schemas.FilterOp
    options:
        show_root_heading: true
        heading_level: 3
        show_if_no_docstring: false
        docstring_options:
            ignore_init_summary: false
            trim_doctest_flags: true

::: docetl.schemas.EquijoinOp
    options:
        show_root_heading: true
        heading_level: 3
        show_if_no_docstring: false
        docstring_options:
            ignore_init_summary: false
            trim_doctest_flags: true

::: docetl.schemas.SplitOp
    options:
        show_root_heading: true
        heading_level: 3
        show_if_no_docstring: false
        docstring_options:
            ignore_init_summary: false
            trim_doctest_flags: true

::: docetl.schemas.GatherOp
    options:
        show_root_heading: true
        heading_level: 3
        show_if_no_docstring: false
        docstring_options:
            ignore_init_summary: false
            trim_doctest_flags: true

::: docetl.schemas.UnnestOp
    options:
        show_root_heading: true
        heading_level: 3
        show_if_no_docstring: false
        docstring_options:
            ignore_init_summary: false
            trim_doctest_flags: true

::: docetl.schemas.SampleOp
    options:
        show_root_heading: true
        heading_level: 3
        show_if_no_docstring: false
        docstring_options:
            ignore_init_summary: false
            trim_doctest_flags: true

::: docetl.schemas.ClusterOp
    options:
        show_root_heading: true
        heading_level: 3
        show_if_no_docstring: false
        docstring_options:
            ignore_init_summary: false
            trim_doctest_flags: true

:::: docetl.schemas.CodeMapOp
    options:
        show_root_heading: true
        heading_level: 3
        show_if_no_docstring: false
        docstring_options:
            ignore_init_summary: false
            trim_doctest_flags: true

:::: docetl.schemas.CodeReduceOp
    options:
        show_root_heading: true
        heading_level: 3
        show_if_no_docstring: false
        docstring_options:
            ignore_init_summary: false
            trim_doctest_flags: true

:::: docetl.schemas.CodeFilterOp
    options:
        show_root_heading: true
        heading_level: 3
        show_if_no_docstring: false
        docstring_options:
            ignore_init_summary: false
            trim_doctest_flags: true

:::: docetl.schemas.ExtractOp
    options:
        show_root_heading: true
        heading_level: 3
        show_if_no_docstring: false
        docstring_options:
            ignore_init_summary: false
            trim_doctest_flags: true

### Callable support for code ops

Code operations (`code_map`, `code_reduce`, `code_filter`) accept either a string containing Python code that defines a `transform` function, or a regular Python function. When you pass a function, it does not need to be named `transform`; DocETL binds it internally.

Example:

```python
from docetl.api import CodeMapOp

def my_map(doc: dict) -> dict:
    return {"double": doc["x"] * 2}

code_map = CodeMapOp(name="double_x", type="code_map", code=my_map)
```

- Map: `fn(doc: dict) -> dict`
- Filter: `fn(doc: dict) -> bool`
- Reduce: `fn(group: list[dict]) -> dict`

See also: [Code Operators](../operators/code.md), [Extract Operator](../operators/extract.md)

## Dataset and Pipeline

::: docetl.schemas.Dataset
    options:
        show_root_heading: true
        heading_level: 3
        show_if_no_docstring: false
        docstring_options:
            ignore_init_summary: false
            trim_doctest_flags: true

::: docetl.schemas.ParsingTool
    options:
        show_root_heading: true
        heading_level: 3
        show_if_no_docstring: false
        docstring_options:
            ignore_init_summary: false
            trim_doctest_flags: true


::: docetl.schemas.PipelineStep
    options:
        show_root_heading: true
        heading_level: 3
        show_if_no_docstring: false
        docstring_options:
            ignore_init_summary: false
            trim_doctest_flags: true

::: docetl.schemas.PipelineOutput
    options:
        show_root_heading: true
        heading_level: 3
        show_if_no_docstring: false
        docstring_options:
            ignore_init_summary: false
            trim_doctest_flags: true

::: docetl.api.Pipeline
    options:
        show_root_heading: true
        heading_level: 3
        show_if_no_docstring: false
        docstring_options:
            ignore_init_summary: false
            trim_doctest_flags: true
