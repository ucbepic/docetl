"""Run one hand-written Quail SQL query over the input rows.

A ``quail_sql`` operation hands Quail Server a SQL query over a table named
``input``, which holds the rows' fields as text plus ``_docetl_id``. The
query may use any of Quail's AI functions (``AI.IF``, ``AI.SCORE``,
``AI.CLASSIFY``, ``AI_EXTRACT``, ``AI.MAP``) and any DataFusion SQL: for
example, an ``AI.IF`` that decides whether to run an ``AI_EXTRACT`` with
``CASE WHEN``, or ``AI.SCORE`` to rank rows. It must return ``_docetl_id`` and
the output schema's columns; a row it leaves out is dropped.

    - name: find_clauses
      type: quail_sql
      model: quail/qwen3.8-27b-fp8
      sql: |
        SELECT i._docetl_id,
               CASE WHEN AI.IF(PROMPT('{0} Does the contract name a governing law?', i.document))
                    THEN get_field(AI_EXTRACT(i.document, named_struct('a', 'Which sentence states the governing law?')), 'a')
               END AS governing_law
        FROM input i
      output:
        schema:
          governing_law: string

Quail reads BigQuery-dialect SQL. Quail Server plans the query against an
empty table at startup, so a query it cannot plan fails before any GPU time.
"""

from __future__ import annotations

import hashlib
import re
import uuid
from typing import Any

from pydantic import Field

from docetl.engines.quail.compile import ID_COLUMN, OutputColumn, _item_kind, convert, engine_for
from docetl.operations.base import BaseOperation, Cardinality
from docetl.operations.quail import _PLANNED, _cell, _endpoints, _require_pyarrow


def _with_input(sql: str, table: str) -> str:
    """Put a CTE named ``input`` over ``table`` in front of ``sql``."""
    head = f"input AS (SELECT * FROM `{table}`)"
    body = sql.strip().rstrip(";")
    if re.match(r"(?is)^with\s", body):
        return f"WITH {head}, " + re.sub(r"(?is)^with\s+", "", body, count=1)
    return f"WITH {head}\n{body}"


def _reads(sql: str, fields: list[str]) -> list[str]:
    """The row fields the query names, out of ``fields``."""
    return [f for f in fields if re.search(rf"(?<![\w`]){re.escape(f)}(?![\w`])|`{re.escape(f)}`", sql)]


class QuailSqlOperation(BaseOperation):
    class schema(BaseOperation.schema):
        type: str = "quail_sql"
        sql: str
        model: str
        output: dict[str, Any]
        input_fields: list[str] | None = Field(
            default=None, description="Row fields to upload; by default, those the query names."
        )

    @classmethod
    def is_llm(cls, config: dict[str, Any]) -> bool:
        return True

    @classmethod
    def cardinality(cls, config: dict[str, Any]) -> Cardinality:
        return Cardinality.MANY_TO_MANY

    def _engine(self) -> dict:
        engine = engine_for(self.config, self.runner.config if self.runner else {})
        if engine is None:
            raise ValueError(f"quail_sql {self.config['name']!r} needs a quail/<model> model")
        return engine

    def _columns(self) -> dict[str, OutputColumn]:
        out = {}
        for name, type_str in self.config["output"]["schema"].items():
            kind, _ = _item_kind(str(type_str))
            out[name] = OutputColumn(name, kind)
        return out

    def syntax_check(self) -> None:
        super().syntax_check()
        if ID_COLUMN not in self.config["sql"]:
            raise ValueError(f"quail_sql {self.config['name']!r} must select {ID_COLUMN}")
        try:
            pa, QuailClient = _require_pyarrow()
        except ImportError:
            return
        engine = self._engine()
        fields = self.config.get("input_fields") or []
        names = [ID_COLUMN, *fields]
        table = "docetl_check_" + hashlib.sha256("\0".join(names).encode()).hexdigest()[:16]
        sql = _with_input(self.config["sql"], table)
        endpoint = _endpoints(engine["endpoint"])[0]
        if (endpoint, sql) in _PLANNED or not fields:
            return  # without input_fields, the query is planned when it runs
        empty = pa.table({name: pa.array([], type=pa.string()) for name in names})
        try:
            with QuailClient(endpoint) as client:
                client.upload(table, empty, ID_COLUMN)
                client.explain(sql)
        except pa.flight.FlightUnavailableError:
            return
        except pa.flight.FlightError as error:
            raise ValueError(f"Quail cannot plan {self.config['name']}: {error}") from error
        _PLANNED.add((endpoint, sql))

    def execute(self, input_data: list[dict], is_build: bool = False) -> tuple[list[dict], float]:
        if not input_data:
            return [], 0.0
        pa, QuailClient = _require_pyarrow()
        engine = self._engine()
        keys = sorted({k for row in input_data for k in row})
        fields = self.config.get("input_fields") or _reads(self.config["sql"], keys)
        ids = [f"{i:010d}" for i in range(len(input_data))]
        columns = {ID_COLUMN: ids}
        for f in fields:
            columns[f] = [_cell(row.get(f)) for row in input_data]
        table = pa.table({k: pa.array(v, type=pa.string()) for k, v in columns.items()})
        digest = hashlib.sha256(self.config["sql"].encode())
        sink = pa.BufferOutputStream()
        with pa.ipc.new_stream(sink, table.schema) as writer:
            writer.write_table(table)
        digest.update(sink.getvalue().to_pybytes())
        key = digest.hexdigest()[:24]
        table_name = f"docetl_{key}"
        sql = _with_input(self.config["sql"], table_name)
        query_id = f"docetl-sql-{key}"
        if self.bypass_cache or self.config.get("bypass_cache"):
            query_id += f"-{uuid.uuid4().hex[:12]}"
        replicas = _endpoints(engine["endpoint"])
        endpoint = replicas[int(key, 16) % len(replicas)]
        self.console.log(f"[cyan]Quail[/cyan] running {self.config['name']} (SQL) on {len(input_data)} rows")
        with QuailClient(endpoint) as client:
            client.upload(table_name, table, ID_COLUMN)
            try:
                result = client.run(sql, query_id=query_id)
            except pa.flight.FlightError as error:
                raise ValueError(f"Quail cannot run {self.config['name']}: {error}") from error
            seconds = client.query_seconds
        cost = seconds / 3600 * engine.get("gpu_cost_per_hour", 0.0)
        self.latency_seconds = seconds
        rows = {name.split(".", 1)[-1]: values for name, values in result.to_pydict().items()}
        if ID_COLUMN not in rows:
            raise ValueError(f"quail_sql {self.config['name']!r} returned no {ID_COLUMN} column")
        outputs = self._columns()
        missing = [name for name in outputs if name not in rows]
        if missing:
            raise ValueError(f"quail_sql {self.config['name']!r} returned no column for {missing}")
        output = []
        for i, row_id in enumerate(rows[ID_COLUMN]):
            row = dict(input_data[int(str(row_id).split(".", 1)[0])])
            for name, out in outputs.items():
                row[name] = convert(rows[name][i], out)
            output.append((str(row_id), row))
        # Quail returns rows in no set order; zero-padded ids sort as the input did.
        return [row for _, row in sorted(output, key=lambda pair: pair[0])], cost
