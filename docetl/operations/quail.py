"""Run a fused run of filter and map operations on Quail Server.

DocETL builds this operation itself when a pipeline sets ``engine: quail``:
``build_operation_graph`` replaces each run of consecutive Quail-eligible
operations with one ``quail`` operation whose config lists the run's operation
configs under ``ops``. Users do not write ``type: quail`` themselves.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import pyarrow as pa
from pydantic import Field

from docetl.engines.quail.client import QuailClient
from docetl.engines.quail.compile import ID_COLUMN, compile_run, convert
from docetl.operations.base import BaseOperation, Cardinality


def _cell(value: Any) -> str | None:
    if value is None or isinstance(value, str):
        return value
    return json.dumps(value, default=str)


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
        if any(op.get("type") == "filter" for op in config.get("ops", [])):
            return Cardinality.MANY_TO_MANY
        return Cardinality.ONE_TO_ONE

    def syntax_check(self) -> None:
        compile_run(self.config["ops"], "syntax_check")

    def execute(self, input_data: list[dict], is_build: bool = False) -> tuple[list[dict], float]:
        if not input_data:
            return [], 0.0
        ops = self.config["ops"]
        names = " -> ".join(op["name"] for op in ops)

        # The table name and query id come from the data and the operations,
        # so a rerun of the same run reads Quail Server's saved rows.
        probe = compile_run(ops, "t")
        ids = [str(i) for i in range(len(input_data))]
        columns = {ID_COLUMN: ids}
        for f in probe.input_fields:
            columns[f] = [_cell(row.get(f)) for row in input_data]
        table = pa.table({k: pa.array(v, type=pa.string()) for k, v in columns.items()})
        digest = hashlib.sha256()
        digest.update(probe.sql.encode())
        sink = pa.BufferOutputStream()
        with pa.ipc.new_stream(sink, table.schema) as writer:
            writer.write_table(table)
        digest.update(sink.getvalue().to_pybytes())
        key = digest.hexdigest()[:24]
        table_name = f"docetl_{key}"
        compiled = compile_run(ops, table_name)

        self.console.log(
            f"[cyan]Quail[/cyan] running {names} on {len(input_data)} rows "
            f"at {self.config['engine']['endpoint']}"
        )
        with QuailClient(self.config["engine"]["endpoint"]) as client:
            client.upload(table_name, table, ID_COLUMN)
            result = client.run(compiled.sql, query_id=f"docetl-{key}")

        # Quail returns column names qualified by their table alias.
        rows = result.to_pydict()
        rows = {name.split(".", 1)[-1]: values for name, values in rows.items()}
        output = []
        for i, row_id in enumerate(rows[ID_COLUMN]):
            row = dict(input_data[int(row_id)])
            for key_name in compiled.dropped:
                row.pop(key_name, None)
            for name, out in compiled.outputs.items():
                row[name] = convert(rows[name][i], out)
            output.append(row)
        return output, 0.0
