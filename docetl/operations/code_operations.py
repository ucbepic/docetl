import ast
import inspect
import os
import textwrap
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from pydantic import Field, field_validator

from docetl.operations.base import BaseOperation, Cardinality
from docetl.operations.utils import RichLoopBar


def _name_field_reads(tree: ast.AST, param: str) -> "frozenset[str] | None":
    """Reads of ``param`` within *tree* as constant subscripts or constant
    ``.get`` calls, or None on any other use of the name (fail closed)."""
    fields: set[str] = set()
    consumed: set[int] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id == param
        ):
            if not (
                isinstance(node.slice, ast.Constant)
                and isinstance(node.slice.value, str)
            ):
                return None
            fields.add(node.slice.value)
            consumed.add(id(node.value))
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == param
        ):
            if not (
                node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                return None
            fields.add(node.args[0].value)
            consumed.add(id(node.func.value))
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == param and id(node) not in consumed:
            return None
    return frozenset(fields)


def extract_doc_field_reads(code: Any) -> "frozenset[str] | None":
    """The set of document fields a code op's transform reads, or None.

    Sound for plan-rewrite decisions: every use of the document parameter
    must be a constant subscript (``doc["k"]``) or a constant ``.get``
    call (``doc.get("k")``). Any other use — aliasing, ``**doc``,
    iteration, dynamic keys, reassignment — returns None (fail closed).
    Callables are analyzed via ``inspect.getsource`` best-effort.

    The document parameter is identified through ``inspect.signature``
    for callables (bound methods drop ``self`` there but keep it in the
    source AST, so positional matching would silently analyze the wrong
    name) and matched against the AST argument list by name. Transforms
    must take exactly one parameter; anything else returns None.
    """
    if isinstance(code, str):
        return _doc_field_reads_str(code)
    if not callable(code):
        return None
    return _doc_field_reads_callable(code)


def _doc_field_reads_str(code: str) -> "frozenset[str] | None":
    try:
        module = ast.parse(code)
    except Exception:
        return None
    # String code must define `transform` (see resolve_transform).
    fn = next(
        (
            node
            for node in ast.walk(module)
            if isinstance(node, ast.FunctionDef) and node.name == "transform"
        ),
        None,
    )
    return _single_param_reads(fn, expected_param=None)


def _doc_field_reads_callable(code: Any) -> "frozenset[str] | None":
    try:
        params = list(inspect.signature(code).parameters.values())
        source = textwrap.dedent(inspect.getsource(code))
        module = ast.parse(source)
    except Exception:
        return None
    positional = [
        p for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    if len(positional) != 1 or len(params) != 1:
        return None
    fn = next(
        (
            node
            for node in ast.walk(module)
            if isinstance(node, (ast.FunctionDef, ast.Lambda))
        ),
        None,
    )
    return _single_param_reads(fn, expected_param=positional[0].name)


def _single_param_reads(
    fn: "ast.FunctionDef | ast.Lambda | None", expected_param: "str | None"
) -> "frozenset[str] | None":
    if fn is None or not fn.args.args:
        return None
    if expected_param is None:
        # String-defined transform: the contract is transform(doc).
        if len(fn.args.args) != 1:
            return None
        param = fn.args.args[0].arg
    else:
        # Match the runtime signature's parameter by name; a bound
        # method's AST keeps `self` at position 0, so position is a lie.
        names = [a.arg for a in fn.args.args]
        if expected_param not in names:
            return None
        param = expected_param
    return _name_field_reads(fn, param)


def extract_eval_field_reads(expr: Any, var: str = "output") -> "frozenset[str] | None":
    """Fields of *var* a ``validate``-style python expression reads, or
    None if it can't be soundly enumerated. Same access grammar as
    ``extract_doc_field_reads``: ``output["k"]`` and ``output.get("k")``
    only; any other use of *var* fails closed."""
    if not isinstance(expr, str):
        return None
    try:
        tree = ast.parse(expr, mode="eval")
    except Exception:
        return None
    return _name_field_reads(tree, var)


def resolve_transform(code: Any):
    """Return the transform callable from a ``code`` config value.

    ``code`` is either a callable (used directly; lambdas and closures work)
    or a string of Python source defining a ``transform`` function.
    """
    if callable(code):
        return code
    namespace = {}
    exec(code, namespace)
    transform = namespace.get("transform")
    if not callable(transform):
        raise ValueError("Code must define a 'transform' function")
    return transform


def _validate_code(v: Any) -> Any:
    if isinstance(v, str) or callable(v):
        return v
    raise TypeError("code must be a string or a callable")


class _RowWiseCodeTraits:
    """Shared plan traits for per-row code ops (code_map, code_filter):
    reads come from AST analysis of the transform; execution is row-local
    and order-preserving. Defined once so a fix to read derivation can't
    apply to one op type and not the other."""

    @classmethod
    def fields_read(cls, config: dict[str, Any]) -> "frozenset[str] | None":
        return extract_doc_field_reads(config.get("code"))

    @classmethod
    def is_row_local(cls, config: dict[str, Any]) -> bool:
        return True

    @classmethod
    def preserves_order(cls, config: dict[str, Any]) -> bool:
        return True


class CodeMapOperation(_RowWiseCodeTraits, BaseOperation):
    class schema(BaseOperation.schema):
        type: str = "code_map"
        code: Any
        concurrent_thread_count: int = os.cpu_count()
        drop_keys: list[str] | None = None
        limit: int | None = Field(None, gt=0)

        validate_code = field_validator("code")(_validate_code)

    def syntax_check(self) -> None:
        config = self.schema(**self.config)
        try:
            resolve_transform(config.code)
        except Exception as e:
            raise ValueError(f"Invalid code configuration: {str(e)}")

    # ── plan traits ────────────────────────────────────────────────

    @classmethod
    def cardinality(cls, config: dict[str, Any]) -> Cardinality:
        if config.get("limit"):
            return Cardinality.MANY_TO_MANY
        return Cardinality.ONE_TO_ONE

    # fields_written stays None: arbitrary code can add any keys.

    def execute(self, input_data: list[dict]) -> tuple[list[dict], float]:
        limit_value = self.config.get("limit")
        if limit_value is not None:
            input_data = input_data[:limit_value]

        transform_fn = resolve_transform(self.config["code"])

        results = []
        with ThreadPoolExecutor(
            max_workers=self.config.get("concurrent_thread_count", os.cpu_count())
        ) as executor:
            futures = [executor.submit(transform_fn, doc) for doc in input_data]
            pbar = RichLoopBar(
                range(len(futures)),
                desc=f"Processing {self.config['name']} (code_map)",
                console=self.console,
            )
            for i in pbar:
                result = futures[i].result()
                if self.config.get("drop_keys"):
                    result = {
                        k: v
                        for k, v in result.items()
                        if k not in self.config["drop_keys"]
                    }
                doc = input_data[i]
                merged_result = {**doc, **result}
                results.append(merged_result)

        return results, 0.0


class CodeReduceOperation(BaseOperation):
    class schema(BaseOperation.schema):
        type: str = "code_reduce"
        code: Any
        concurrent_thread_count: int = os.cpu_count()
        limit: int | None = Field(None, gt=0)

        validate_code = field_validator("code")(_validate_code)

    def syntax_check(self) -> None:
        config = self.schema(**self.config)
        try:
            resolve_transform(config.code)
        except Exception as e:
            raise ValueError(f"Invalid code configuration: {str(e)}")

    # ── plan traits ────────────────────────────────────────────────

    @classmethod
    def cardinality(cls, config: dict[str, Any]) -> Cardinality:
        return Cardinality.MANY_TO_ONE

    def execute(self, input_data: list[dict]) -> tuple[list[dict], float]:
        reduce_fn = resolve_transform(self.config["code"])

        reduce_keys = self.config.get("reduce_key", "_all")
        if not isinstance(reduce_keys, list):
            reduce_keys = [reduce_keys]

        if reduce_keys == ["_all"] or reduce_keys == "_all":
            grouped_data = [("_all", input_data)]
        else:

            def get_group_key(item):
                return tuple(item[key] for key in reduce_keys)

            grouped_data = {}
            for item in input_data:
                key = get_group_key(item)
                if key not in grouped_data:
                    grouped_data[key] = []
                grouped_data[key].append(item)

            grouped_data = list(grouped_data.items())

        limit_value = self.config.get("limit")
        if limit_value is not None:
            # Sort by group size (smallest first) and take the limit
            grouped_data = sorted(grouped_data, key=lambda x: len(x[1]))
            grouped_data = grouped_data[:limit_value]

        results = []
        with ThreadPoolExecutor(
            max_workers=self.config.get("concurrent_thread_count", os.cpu_count())
        ) as executor:
            futures = [executor.submit(reduce_fn, group) for _, group in grouped_data]
            pbar = RichLoopBar(
                range(len(futures)),
                desc=f"Processing {self.config['name']} (code_reduce)",
                console=self.console,
            )
            for i, (key, group) in zip(pbar, grouped_data):
                result = futures[i].result()

                # Apply pass-through at the group level
                if self.config.get("pass_through", False) and group:
                    for k, v in group[0].items():
                        if k not in result:
                            result[k] = v

                # Also add the reduce key
                if reduce_keys != ["_all"]:
                    for k in reduce_keys:
                        if k not in result:
                            result[k] = group[0][k]

                result[f"_counts_prereduce_{self.config['name']}"] = len(group)

                results.append(result)

        return results, 0.0


class CodeFilterOperation(_RowWiseCodeTraits, BaseOperation):
    class schema(BaseOperation.schema):
        type: str = "code_filter"
        code: Any
        concurrent_thread_count: int = os.cpu_count()
        limit: int | None = Field(None, gt=0)

        validate_code = field_validator("code")(_validate_code)

    def syntax_check(self) -> None:
        config = self.schema(**self.config)
        try:
            resolve_transform(config.code)
        except Exception as e:
            raise ValueError(f"Invalid code configuration: {str(e)}")

    # ── plan traits ────────────────────────────────────────────────

    @classmethod
    def cardinality(cls, config: dict[str, Any]) -> Cardinality:
        return Cardinality.SELECTION

    @classmethod
    def fields_written(cls, config: dict[str, Any]) -> "frozenset[str]":
        return frozenset()  # kept rows pass through untouched

    def execute(self, input_data: list[dict]) -> tuple[list[dict], float]:
        filter_fn = resolve_transform(self.config["code"])

        limit_value = self.config.get("limit")
        results = []
        with ThreadPoolExecutor(
            max_workers=self.config.get("concurrent_thread_count", os.cpu_count())
        ) as executor:
            futures = [executor.submit(filter_fn, doc) for doc in input_data]
            pbar = RichLoopBar(
                range(len(futures)),
                desc=f"Processing {self.config['name']} (code_filter)",
                console=self.console,
            )
            for i in pbar:
                should_keep = futures[i].result()
                if should_keep:
                    results.append(input_data[i])
                    if limit_value is not None and len(results) >= limit_value:
                        break
        return results, 0.0
