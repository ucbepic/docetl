"""Run a fused run of filter, map, and reduce operations on Quail Server.

DocETL builds this operation itself when a pipeline sets ``engine: quail``:
``build_operation_graph`` replaces each run of consecutive Quail-eligible
operations with one ``quail`` operation whose config lists the run's operation
configs under ``ops``. Users do not write ``type: quail`` themselves.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from collections import defaultdict
from typing import Any

from pydantic import Field

from docetl.engines.quail.compile import ID_COLUMN, compile_run, convert
from docetl.operations.base import BaseOperation, Cardinality

# (endpoint, sql) pairs Quail Server has already planned in this process;
# every operation construction runs syntax_check, so plan each query once.
_PLANNED: set[tuple[str, str]] = set()


def _cell(value: Any) -> str | None:
    if value is None or isinstance(value, str):
        return value
    return json.dumps(value, default=str)


def _require_pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.flight  # noqa: F401 - pa.flight

        from docetl.engines.quail.client import QuailClient
    except ImportError as error:
        raise ImportError(
            "The Quail engine needs pyarrow: pip install 'docetl[quail]'"
        ) from error
    return pa, QuailClient


class QuailOperation(BaseOperation):
    class schema(BaseOperation.schema):
        type: str = "quail"
        engine: dict[str, Any]
        ops: list[dict[str, Any]] = Field(min_length=1)

    @classmethod
    def is_llm(cls, config: dict[str, Any]) -> bool:
        return True

    @classmethod
    def cardinality(cls, config: dict[str, Any]) -> Cardinality:
        types = {op.get("type") for op in config.get("ops", [])}
        if types <= {"map"}:
            return Cardinality.ONE_TO_ONE
        return Cardinality.MANY_TO_MANY

    def syntax_check(self) -> None:
        probe = compile_run(self.config["ops"], "t")
        try:
            pa, QuailClient = _require_pyarrow()
        except ImportError:
            return  # execute raises with the install hint
        # Plan the query on the server against an empty table of the same
        # shape, so a query Quail cannot plan fails before any GPU time.
        names = [ID_COLUMN, *probe.input_fields]
        table = "docetl_check_" + hashlib.sha256("\0".join(names).encode()).hexdigest()[:16]
        sql = compile_run(self.config["ops"], table).sql
        endpoint = self.config["engine"]["endpoint"]
        if (endpoint, sql) in _PLANNED:
            return
        empty = pa.table({name: pa.array([], type=pa.string()) for name in names})
        try:
            with QuailClient(endpoint) as client:
                client.upload(table, empty, ID_COLUMN)
                client.explain(sql)
        except pa.flight.FlightUnavailableError as error:
            self.console.log(
                f"[yellow]Quail engine: could not reach {endpoint} to check "
                f"{self.config['name']}: {error}[/yellow]"
            )
            return
        except pa.flight.FlightError as error:
            raise ValueError(f"Quail cannot plan {self.config['name']}: {error}") from error
        _PLANNED.add((endpoint, sql))

    def _oversized_group(self, input_data: list[dict], reads) -> int | None:
        """Return the largest group's estimated tokens if it exceeds the limit."""
        keys, fields = reads
        sizes: dict[tuple, int] = defaultdict(int)
        for row in input_data:
            group = tuple(_cell(row.get(k)) for k in keys)
            sizes[group] += sum(len(_cell(row.get(f)) or "") for f in fields)
        largest = max(sizes.values(), default=0) // 4  # about 4 characters per token
        return largest if largest > self.config["engine"]["max_group_tokens"] else None

    def _run_on_litellm(self, input_data: list[dict]) -> list[dict]:
        """Run each operation of the run on LiteLLM, in order."""
        data = input_data
        for op in self.config["ops"]:
            data = self.runner._run_operation(op, data)
        return data

    def execute(self, input_data: list[dict], is_build: bool = False) -> tuple[list[dict], float]:
        if not input_data:
            return [], 0.0
        pa, QuailClient = _require_pyarrow()
        ops = self.config["ops"]
        engine = self.config["engine"]
        names = " -> ".join(op["name"] for op in ops)

        probe = compile_run(ops, "t")
        if probe.group_reads is not None:
            tokens = self._oversized_group(input_data, probe.group_reads)
            if tokens is not None:
                message = (
                    f"a reduce group has about {tokens} tokens, over "
                    f"max_group_tokens ({engine['max_group_tokens']})"
                )
                if engine["fallback"] == "strict":
                    raise ValueError(f"Quail engine (strict): {message}")
                self.console.log(f"[yellow]Quail engine: {message}; running {names} on LiteLLM.[/yellow]")
                # Sub-operations add their own cost to the runner's total.
                return self._run_on_litellm(input_data), 0.0

        # Zero-padded ids sort in row order, which STRING_AGG keeps.
        ids = [f"{i:010d}" for i in range(len(input_data))]
        columns = {ID_COLUMN: ids}
        for f in probe.input_fields:
            columns[f] = [_cell(row.get(f)) for row in input_data]
        table = pa.table({k: pa.array(v, type=pa.string()) for k, v in columns.items()})

        # The table name and query id come from the data and the operations,
        # so a rerun of the same run reads Quail Server's saved rows.
        digest = hashlib.sha256(probe.sql.encode())
        sink = pa.BufferOutputStream()
        with pa.ipc.new_stream(sink, table.schema) as writer:
            writer.write_table(table)
        digest.update(sink.getvalue().to_pybytes())
        key = digest.hexdigest()[:24]
        table_name = f"docetl_{key}"
        compiled = compile_run(ops, table_name)
        # The same query id reads Quail Server's saved rows; bypass_cache asks
        # for a fresh run with a new id (the uploaded table stays the same).
        query_id = f"docetl-{key}"
        if self.bypass_cache or self.config.get("bypass_cache"):
            query_id += f"-{uuid.uuid4().hex[:12]}"

        self.console.log(
            f"[cyan]Quail[/cyan] running {names} on {len(input_data)} rows at {engine['endpoint']}"
        )
        with QuailClient(engine["endpoint"]) as client:
            client.upload(table_name, table, ID_COLUMN)
            result = client.run(compiled.sql, query_id=query_id)

        # Quail returns column names qualified by their table alias.
        rows = {name.split(".", 1)[-1]: values for name, values in result.to_pydict().items()}
        count = result.num_rows
        if not compiled.grouped:
            output = []
            for i, row_id in enumerate(rows[ID_COLUMN]):
                row = dict(input_data[int(row_id)])
                for name in compiled.dropped:
                    row.pop(name, None)
                for name, out in compiled.outputs.items():
                    row[name] = convert(rows[name][i], out)
                output.append(row)
            return output, 0.0

        # Group rows: restore each uploaded key's original value and type.
        originals: dict[str, dict[str | None, Any]] = defaultdict(dict)
        for name, out in compiled.outputs.items():
            if out.kind == "key" and name in probe.input_fields:
                for row in input_data:
                    originals[name].setdefault(_cell(row.get(name)), row.get(name))
        output = []
        for i in range(count):
            row = {}
            for name, out in compiled.outputs.items():
                value = rows[name][i]
                if out.kind == "key" and name in originals:
                    row[name] = originals[name].get(value, value)
                else:
                    row[name] = convert(value, out)
            output.append(row)
        return output, 0.0
