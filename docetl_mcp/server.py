import asyncio
import glob
import json
import os
from dataclasses import asdict
from typing import Any, Dict, Iterable, List, Tuple

import yaml

from docetl.runner import DSLRunner
from docetl.operations import get_operations
from docetl.parsing_tools import get_parsing_tools

# Optional imports guarded at runtime
try:
    import fitz  # PyMuPDF
except Exception:  # pragma: no cover - optional dependency
    fitz = None  # type: ignore

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import TextContent
except Exception as e:  # pragma: no cover - server won't start without mcp installed
    raise RuntimeError(
        "The 'mcp' package is required to run the DocETL MCP server. "
        "Install with: pip install mcp"
    ) from e


server = Server("docetl-mcp")


# --------------- Utility helpers ---------------

DEFAULT_INCLUDE_GLOBS = [
    "**/*.txt",
    "**/*.md",
    "**/*.pdf",
    "**/*.docx",
    "**/*.pptx",
    "**/*.xlsx",
]

SUPPORTED_EXTS = {".txt", ".md", ".pdf", ".docx", ".pptx", ".xlsx"}


def _resolve_abs(path: str) -> str:
    return path if os.path.isabs(path) else os.path.abspath(path)


def _collect_files(
    directory: str, include: List[str] | None, exclude: List[str] | None
) -> List[str]:
    base = _resolve_abs(directory)
    patterns = include or DEFAULT_INCLUDE_GLOBS
    exclude = exclude or []

    files: List[str] = []
    for pattern in patterns:
        files.extend(glob.glob(os.path.join(base, pattern), recursive=True))

    # De-duplicate and filter by supported extensions
    files = sorted(
        {f for f in files if os.path.isfile(f) and os.path.splitext(f)[1].lower() in SUPPORTED_EXTS}
    )

    # Apply exclude patterns
    if exclude:
        excluded: set[str] = set()
        for pattern in exclude:
            excluded.update(glob.glob(os.path.join(base, pattern), recursive=True))
        files = [f for f in files if f not in excluded]

    return files


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _read_pdf_text(path: str, doc_per_page: bool) -> List[str]:
    if fitz is None:
        # No PyMuPDF available; fall back to single record with placeholder
        return [f"[PDF parsing unavailable: install docetl[parsing]] Path: {path}"]
    texts: List[str] = []
    with fitz.open(path) as doc:
        if doc_per_page:
            for idx, page in enumerate(doc):
                texts.append(f"Page {idx+1}:\n{page.get_text() or ''}")
        else:
            buf = []
            for idx, page in enumerate(doc):
                buf.append(f"Page {idx+1}:\n{page.get_text() or ''}")
            texts.append("\n\n".join(buf))
    return texts


def _ensure_dir(path: str) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)


def _dump_json(path: str, data: Any) -> None:
    _ensure_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _try_load_yaml_or_json(
    yaml_path: str | None = None, yaml_text: str | None = None, config: dict | None = None
) -> dict:
    if config is not None:
        return config
    if yaml_text is not None:
        return yaml.safe_load(yaml_text)
    if yaml_path is not None:
        with open(_resolve_abs(yaml_path), "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    raise ValueError("Provide one of: config (dict), yaml (str), or yaml_path (str)")


def _ok(payload: dict) -> List[TextContent]:
    return [TextContent(type="text", text=json.dumps(payload, indent=2))]


def _error(message: str, extra: dict | None = None) -> List[TextContent]:
    payload = {"error": message}
    if extra:
        payload.update(extra)
    return [TextContent(type="text", text=json.dumps(payload, indent=2))]


# --------------- Tools ---------------


@server.tool(
    "dataset.create_from_directory",
    "Create a DocETL dataset JSON from files in a directory.",
    {
        "type": "object",
        "properties": {
            "directory": {"type": "string"},
            "include": {"type": "array", "items": {"type": "string"}},
            "exclude": {"type": "array", "items": {"type": "string"}},
            "mode": {"type": "string", "enum": ["eager", "lazy"], "default": "eager"},
            "doc_per_page": {"type": "boolean", "default": False},
            "pdf_ocr": {"type": "boolean", "default": False},
            "output_path": {"type": "string"},
            "include_metadata": {"type": "boolean", "default": True},
        },
        "required": ["directory", "output_path"],
        "additionalProperties": False,
    },
)
async def dataset_create_from_directory(
    directory: str,
    output_path: str,
    include: List[str] | None = None,
    exclude: List[str] | None = None,
    mode: str = "eager",
    doc_per_page: bool = False,
    pdf_ocr: bool = False,
    include_metadata: bool = True,
):
    directory = _resolve_abs(directory)
    output_path = _resolve_abs(output_path)
    files = _collect_files(directory, include, exclude)

    if not files:
        return _error("No files found for given patterns", {"directory": directory})

    records: List[dict] = []
    suggested_parsing: List[dict] = []

    # Suggested parsing only when in lazy mode
    if mode == "lazy":
        # Map extensions to parser names and kwargs
        lazy_map: Dict[str, Tuple[str, Dict[str, Any]]] = {
            ".txt": ("txt_to_string", {}),
            ".md": ("txt_to_string", {}),
            ".docx": ("docx_to_string", {}),
            ".pptx": ("pptx_to_string", {}),
            ".xlsx": ("xlsx_to_string", {}),
            ".pdf": ("paddleocr_pdf_to_string", {"doc_per_page": doc_per_page, "ocr_enabled": pdf_ocr}),
        }

        for f in files:
            _, ext = os.path.splitext(f)
            ext = ext.lower()
            rec = {"file_path": f, "ext": ext.lstrip(".")}
            records.append(rec)
        # One consolidated suggestion per extension present
        seen_exts = {os.path.splitext(f)[1].lower() for f in files}
        for ext in sorted(seen_exts):
            if ext in lazy_map:
                function, kwargs = lazy_map[ext]
                suggested_parsing.append(
                    {
                        "function": function,
                        "function_kwargs": kwargs,
                        "input_key": "file_path",
                        "output_key": "src",
                    }
                )
    else:
        # Eager mode: create records with extracted text for a subset of formats.
        for f in files:
            _, ext = os.path.splitext(f)
            ext = ext.lower()
            if ext in {".txt", ".md"}:
                texts = [_read_text(f)]
            elif ext == ".pdf":
                texts = _read_pdf_text(f, doc_per_page=doc_per_page)
            else:
                # Unsupported eager extraction → fall back to lazy-like record (file_path only)
                texts = []

            if texts:
                for text in texts:
                    rec = {"src": text}
                    if include_metadata:
                        rec.update({"path": f, "ext": ext.lstrip(".")})
                    records.append(rec)
            else:
                # If we cannot eagerly parse, at least include as a file reference
                records.append({"file_path": f, "ext": ext.lstrip(".")})

    _dump_json(output_path, records)

    sample_preview = records[: min(3, len(records))]
    payload: Dict[str, Any] = {
        "dataset_path": output_path,
        "num_items": len(records),
        "sample": sample_preview,
    }
    if mode == "lazy":
        payload["suggested_parsing"] = suggested_parsing
    return _ok(payload)


@server.tool(
    "dataset.create_from_directory_azure_di",
    "Create a dataset JSON configured for Azure Document Intelligence reading (requires keys).",
    {
        "type": "object",
        "properties": {
            "directory": {"type": "string"},
            "include": {"type": "array", "items": {"type": "string"}},
            "exclude": {"type": "array", "items": {"type": "string"}},
            "doc_per_page": {"type": "boolean", "default": False},
            "output_path": {"type": "string"},
        },
        "required": ["directory", "output_path"],
        "additionalProperties": False,
    },
)
async def dataset_create_from_directory_azure_di(
    directory: str,
    output_path: str,
    include: List[str] | None = None,
    exclude: List[str] | None = None,
    doc_per_page: bool = False,
):
    # Validate keys up-front
    if not os.getenv("DOCUMENTINTELLIGENCE_API_KEY") or not os.getenv("DOCUMENTINTELLIGENCE_ENDPOINT"):
        return _error(
            "Azure Document Intelligence keys not set. Set DOCUMENTINTELLIGENCE_API_KEY and DOCUMENTINTELLIGENCE_ENDPOINT."
        )

    directory = _resolve_abs(directory)
    output_path = _resolve_abs(output_path)
    files = _collect_files(directory, include, exclude)

    if not files:
        return _error("No files found for given patterns", {"directory": directory})

    # Always lazy: rely on azure_di_read
    records = [{"file_path": f, "ext": os.path.splitext(f)[1].lower().lstrip(".")} for f in files]
    _dump_json(output_path, records)

    suggested_parsing = [
        {
            "function": "azure_di_read",
            "function_kwargs": {"doc_per_page": doc_per_page},
            "input_key": "file_path",
            "output_key": "src",
        }
    ]
    return _ok(
        {
            "dataset_path": output_path,
            "num_items": len(records),
            "sample": records[: min(3, len(records))],
            "suggested_parsing": suggested_parsing,
        }
    )


@server.tool(
    "dataset.sample",
    "Sample N rows from a dataset JSON created by create_from_directory.",
    {
        "type": "object",
        "properties": {
            "dataset_path": {"type": "string"},
            "n": {"type": "integer", "minimum": 1, "default": 3},
        },
        "required": ["dataset_path"],
        "additionalProperties": False,
    },
)
async def dataset_sample(dataset_path: str, n: int = 3):
    dataset_path = _resolve_abs(dataset_path)
    if not os.path.exists(dataset_path):
        return _error("Dataset path not found", {"dataset_path": dataset_path})
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        return _error("Dataset JSON is not a list of records", {"dataset_path": dataset_path})
    return _ok({"rows": data[: max(0, n)]})


@server.tool(
    "ops.list",
    "List available DocETL operations.",
    {"type": "object", "properties": {}, "additionalProperties": False},
)
async def ops_list():
    ops = get_operations()
    resp = []
    for name, cls in sorted(ops.items(), key=lambda x: x[0]):
        doc = (cls.__doc__ or "").strip()
        resp.append({"name": name, "doc": doc})
    return _ok({"operations": resp})


@server.tool(
    "parsing_tools.list",
    "List available DocETL parsing tools.",
    {"type": "object", "properties": {}, "additionalProperties": False},
)
async def parsing_tools_list():
    tools = get_parsing_tools()
    return _ok({"parsing_tools": sorted(tools)})


@server.tool(
    "pipeline.schema",
    "Return the DocETL JSON schema for pipeline configs.",
    {"type": "object", "properties": {}, "additionalProperties": False},
)
async def pipeline_schema():
    schema = DSLRunner.json_schema
    return _ok({"schema": schema})


@server.tool(
    "pipeline.validate",
    "Validate a pipeline (YAML string, YAML path, or config dict).",
    {
        "type": "object",
        "properties": {
            "yaml": {"type": "string"},
            "yaml_path": {"type": "string"},
            "config": {"type": "object"},
            "max_threads": {"type": "integer"},
        },
        "additionalProperties": False,
    },
)
async def pipeline_validate(
    yaml: str | None = None, yaml_path: str | None = None, config: dict | None = None, max_threads: int | None = None
):
    try:
        cfg = _try_load_yaml_or_json(yaml_path=yaml_path, yaml_text=yaml, config=config)
        # Build runner for syntax check only
        _ = DSLRunner(cfg, max_threads=max_threads)
        normalized = yaml and yaml or yaml.safe_dump(cfg)
        return _ok({"valid": True, "errors": [], "normalizedYaml": normalized})
    except Exception as e:
        return _ok({"valid": False, "errors": [str(e)]})


@server.tool(
    "pipeline.run",
    "Run a pipeline (YAML string, YAML path, or config dict).",
    {
        "type": "object",
        "properties": {
            "yaml": {"type": "string"},
            "yaml_path": {"type": "string"},
            "config": {"type": "object"},
            "max_threads": {"type": "integer"},
            "optimize": {"type": "boolean", "default": False},
            "save_optimized_path": {"type": "string"},
        },
        "additionalProperties": False,
    },
)
async def pipeline_run(
    yaml: str | None = None,
    yaml_path: str | None = None,
    config: dict | None = None,
    max_threads: int | None = None,
    optimize: bool = False,
    save_optimized_path: str | None = None,
):
    try:
        cfg = _try_load_yaml_or_json(yaml_path=yaml_path, yaml_text=yaml, config=config)
        runner = DSLRunner(cfg, max_threads=max_threads)
        cost: float
        if optimize:
            _, cost = runner.optimize(save=bool(save_optimized_path), save_path=save_optimized_path)
            # After optimize, re-run to produce outputs with optimized config
            # The optimize() returns either a new runner or a dict; but we already saved optimized yaml if requested.
            # For simplicity, proceed to run the original runner (user can also run the optimized yaml separately).
        cost = runner.load_run_save()
        return _ok({"cost": cost, "output_path": runner.get_output_path(require=True)})
    except Exception as e:
        return _error("Pipeline run failed", {"message": str(e)})


# --------- Examples and Docs (minimal embedded content) ----------

EXAMPLES: Dict[str, str] = {
    "summarize-minimal": """\
datasets:
  input:
    type: file
    path: /abs/path/to/dataset.json

default_model: gpt-4o-mini

operations:
  - name: summarize
    type: map
    output:
      schema:
        summary: str
    prompt: |
      Summarize the following document briefly:
      {{ input.src }}

pipeline:
  steps:
    - name: step1
      input: input
      operations:
        - summarize
  output:
    type: file
    path: /abs/path/to/output.json
    intermediate_dir: intermediate
""",
    "split-gather": """\
datasets:
  input:
    type: file
    path: /abs/path/to/dataset.json

default_model: gpt-4o-mini

operations:
  - name: split
    type: split
    chunk_size: 2000
    chunk_overlap: 200
    text_key: src
    output:
      schema:
        chunk: str
  - name: analyze
    type: map
    output:
      schema:
        facts: list[str]
    prompt: |
      Extract salient facts from the chunk:
      {{ input.chunk }}
  - name: gather
    type: gather
    gather_key: path
    gather_size: 100
    output:
      schema:
        chunks: list[dict]
  - name: reduce
    type: reduce
    reduce_key: [path]
    output:
      schema:
        summary: str
    prompt: |
      Using all facts from these chunks, write a unified summary:
      {% for item in inputs %}- {{ item.facts }}{% endfor %}

pipeline:
  steps:
    - name: step1
      input: input
      operations:
        - split
        - analyze
        - gather
        - reduce
  output:
    type: file
    path: /abs/path/to/output.json
""",
    "resolve-deduplicate": """\
datasets:
  input:
    type: file
    path: /abs/path/to/dataset.json

default_model: gpt-4o-mini

operations:
  - name: extract
    type: map
    output:
      schema:
        company: str
    prompt: |
      From the document, extract the company name:
      {{ input.src }}
  - name: resolve_companies
    type: resolve
    blocking_keys: ["company"]
    blocking_threshold: 0.62
    comparison_prompt: |
      Are these two names the same company?
      1: {{ input1.company }}
      2: {{ input2.company }}
    output:
      schema:
        company: str
    resolution_prompt: |
      Determine a canonical company name for:
      {% for entry in inputs %}- {{ entry.company }}{% endfor %}
  - name: reduce_by_company
    type: reduce
    reduce_key: [company]
    output:
      schema:
        summary: str
    prompt: |
      Summarize information for {{ reduce_key }} using all inputs.
      {% for item in inputs %}- {{ item.src }}{% endfor %}

pipeline:
  steps:
    - name: step1
      input: input
      operations:
        - extract
        - resolve_companies
        - reduce_by_company
  output:
    type: file
    path: /abs/path/to/output.json
""",
}

DOCS: Dict[str, str] = {
    "index": "DocETL MCP: Ingest files into datasets, author arbitrary pipelines, and run them. Use ops.list, parsing_tools.list, pipeline.schema to compose pipelines.",
    "pipelines": "Pipelines consist of datasets, operations, steps, and output. See pipeline.schema for the full JSON schema.",
    "operators": "Supported operators: map, reduce, resolve, parallel_map, filter, equijoin, split, gather, unnest, cluster, sample, code_map, code_reduce, code_filter, extract, etc.",
}


@server.tool(
    "examples.list",
    "List available example pipeline templates.",
    {"type": "object", "properties": {}, "additionalProperties": False},
)
async def examples_list():
    return _ok({"examples": sorted(EXAMPLES.keys())})


@server.tool(
    "examples.get",
    "Get an example pipeline YAML by name.",
    {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
        "additionalProperties": False,
    },
)
async def examples_get(name: str):
    if name not in EXAMPLES:
        return _error("Example not found", {"available": sorted(EXAMPLES.keys())})
    return _ok({"name": name, "yaml": EXAMPLES[name]})


@server.tool(
    "docs.get",
    "Get a brief documentation page by name.",
    {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
        "additionalProperties": False,
    },
)
async def docs_get(name: str):
    if name not in DOCS:
        return _error("Doc not found", {"available": sorted(DOCS.keys())})
    return _ok({"name": name, "text": DOCS[name]})


async def _amain() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream)


def main() -> None:
    asyncio.run(_amain())


if __name__ == "__main__":
    main()


