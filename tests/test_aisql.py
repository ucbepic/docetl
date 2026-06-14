"""AI SQL layer: Arrow adapters, DuckDB delegate, SQL compilation.

LLM-free — the AI operators are exercised structurally (we assert the
compiled plan), and execution tests use relational/code paths only.
"""

import pyarrow as pa

import docetl


class TestArrowAdapters:
    ROWS = [{"id": 1, "text": "a"}, {"id": 2, "text": "b"}, {"id": 3, "text": "c"}]

    # The runner needs at least one op, so every case carries a code_map
    # (the realistic shape: load Arrow, then transform).
    NOOP = "def transform(doc):\n    return {}"

    def test_from_arrow_feeds_pipeline(self):
        table = pa.Table.from_pylist(self.ROWS)
        out = docetl.from_arrow(table).code_map(
            code="def transform(doc):\n    return {'n': doc['id'] * 2}"
        ).collect()
        assert [r["n"] for r in out] == [2, 4, 6]
        assert out[0] == {"id": 1, "text": "a", "n": 2}

    def test_to_arrow_returns_table(self):
        out = docetl.from_list(self.ROWS).code_map(code=self.NOOP).to_arrow()
        assert isinstance(out, pa.Table)
        assert out.to_pylist() == self.ROWS

    def test_round_trip(self):
        table = pa.Table.from_pylist(self.ROWS)
        out = docetl.from_arrow(table).code_map(code=self.NOOP).to_arrow()
        assert out.to_pylist() == self.ROWS


class TestDuckDBEngine:
    def test_register_and_filter(self):
        from docetl.aisql import DuckDBEngine

        rows = [{"id": i, "price": i * 5} for i in range(1, 6)]
        with DuckDBEngine() as eng:
            eng.register("items", pa.Table.from_pylist(rows))
            out = eng.sql("SELECT id FROM items WHERE price > 10 ORDER BY id")
        assert out.to_pylist() == [{"id": 3}, {"id": 4}, {"id": 5}]

    def test_reads_parquet_file(self, tmp_path):
        import pyarrow.parquet as pq

        from docetl.aisql import DuckDBEngine

        path = tmp_path / "calls.parquet"
        pq.write_table(
            pa.Table.from_pylist(
                [{"id": 1, "dur": 100}, {"id": 2, "dur": 400}, {"id": 3, "dur": 500}]
            ),
            path,
        )
        with DuckDBEngine() as eng:
            out = eng.sql(f"SELECT id FROM '{path}' WHERE dur > 300 ORDER BY id")
        assert out.to_pylist() == [{"id": 2}, {"id": 3}]


class TestCompile:
    def test_leading_example(self):
        from docetl.aisql import compile_sql, RelationalStage, SemanticStage

        stages = compile_sql(
            "SELECT ai_summarize(transcript) AS summary "
            "FROM calls WHERE duration > 300"
        ).stages
        assert len(stages) == 3
        assert isinstance(stages[0], RelationalStage)
        assert stages[0].sql == "SELECT * FROM calls WHERE duration > 300"
        assert not stages[0].reads_prev
        assert isinstance(stages[1], SemanticStage)
        (op,) = stages[1].operations
        assert op["type"] == "map"
        assert op["output"]["schema"] == {"summary": "string"}
        assert "{{ input.transcript }}" in op["prompt"]
        assert isinstance(stages[2], RelationalStage)
        assert stages[2].sql == "SELECT summary FROM _prev"
        assert stages[2].reads_prev

    def test_pure_relational_single_stage(self):
        from docetl.aisql import compile_sql, RelationalStage

        stages = compile_sql("SELECT id FROM t WHERE x > 1").stages
        assert len(stages) == 1
        assert isinstance(stages[0], RelationalStage)
        assert stages[0].sql == "SELECT id FROM t WHERE x > 1"

    def test_passthrough_column_kept(self):
        from docetl.aisql import compile_sql

        stages = compile_sql("SELECT id, ai_summarize(body) AS s FROM docs").stages
        assert stages[0].sql == "SELECT * FROM docs"
        assert stages[2].sql == "SELECT id, s FROM _prev"

    def test_ai_in_select_needs_alias(self):
        import pytest

        from docetl.aisql import compile_sql

        with pytest.raises(NotImplementedError, match="alias"):
            compile_sql("SELECT ai_summarize(body) FROM docs")


class TestRun:
    def _parquet(self, tmp_path):
        import pyarrow.parquet as pq

        path = tmp_path / "products.parquet"
        pq.write_table(
            pa.Table.from_pylist(
                [{"id": i, "price": i * 5} for i in range(1, 6)]
            ),
            path,
        )
        return path

    def test_run_pure_relational(self, tmp_path):
        from docetl.aisql import run_sql

        path = self._parquet(tmp_path)
        out = run_sql(f"SELECT id FROM '{path}' WHERE price > 10 ORDER BY id")
        assert out.to_pylist() == [{"id": 3}, {"id": 4}, {"id": 5}]

    def test_hybrid_threads_duckdb_and_docetl(self, tmp_path):
        # Manually-built plan with a code_map semantic stage: exercises the
        # real DuckDB -> DocETL -> DuckDB handoff (register/_prev, Arrow
        # threading) end to end, no LLM.
        from docetl.aisql import (
            CompiledQuery,
            RelationalStage,
            SemanticStage,
            run_compiled,
        )

        path = self._parquet(tmp_path)
        compiled = CompiledQuery(
            stages=[
                RelationalStage(sql=f"SELECT * FROM '{path}' WHERE price > 10"),
                SemanticStage(
                    operations=[
                        {
                            "name": "tax",
                            "type": "code_map",
                            "code": "def transform(doc):\n    return {'tax': doc['price'] * 0.1}",
                        }
                    ]
                ),
                RelationalStage(sql="SELECT id, tax FROM _prev ORDER BY id", reads_prev=True),
            ]
        )
        out = run_compiled(compiled).to_pylist()
        assert out == [
            {"id": 3, "tax": 1.5},
            {"id": 4, "tax": 2.0},
            {"id": 5, "tax": 2.5},
        ]


class TestSplitter:
    def test_and_splits_relational_and_ai_filter(self):
        from docetl.aisql import compile_sql

        stages = compile_sql(
            "SELECT ai_summarize(transcript) AS s FROM calls "
            "WHERE duration > 300 AND ai_filter(transcript, 'about billing?')"
        ).stages
        assert len(stages) == 3
        # relational half of WHERE goes to DuckDB
        assert stages[0].sql == "SELECT * FROM calls WHERE duration > 300"
        # ai_filter runs before the SELECT-list map (shrink first)
        types = [(o["type"], o["name"]) for o in stages[1].operations]
        assert types == [("filter", "aifilter_0"), ("map", "ai_s")]
        assert stages[1].operations[0]["output"]["schema"] == {"keep": "boolean"}
        assert stages[2].sql == "SELECT s FROM _prev"

    def test_ai_filter_only(self):
        from docetl.aisql import compile_sql

        stages = compile_sql("SELECT * FROM t WHERE ai_filter(body, 'spam?')").stages
        assert stages[0].sql == "SELECT * FROM t"  # no relational WHERE
        assert [o["type"] for o in stages[1].operations] == ["filter"]
        assert stages[2].sql == "SELECT * FROM _prev"

    def test_multiple_relational_conjuncts_preserved(self):
        from docetl.aisql import compile_sql

        stages = compile_sql(
            "SELECT * FROM t WHERE a > 1 AND b < 5 AND ai_filter(x, 'q')"
        ).stages
        assert stages[0].sql == "SELECT * FROM t WHERE a > 1 AND b < 5"

    def test_or_with_ai_rejected(self):
        import pytest

        from docetl.aisql import compile_sql

        with pytest.raises(NotImplementedError, match="OR/NOT"):
            compile_sql("SELECT id FROM t WHERE a > 1 OR ai_filter(x, 'q')")


class TestSplitterExecution:
    def test_filter_threads_across_boundary(self, tmp_path):
        # A code_filter standing in for ai_filter: validates that a filter
        # op in the semantic stage drops rows and threads through the
        # DuckDB -> DocETL -> DuckDB handoff, no LLM.
        import pyarrow.parquet as pq

        from docetl.aisql import (
            CompiledQuery,
            RelationalStage,
            SemanticStage,
            run_compiled,
        )

        path = tmp_path / "products.parquet"
        pq.write_table(
            pa.Table.from_pylist([{"id": i, "price": i * 5} for i in range(1, 6)]),
            path,
        )
        compiled = CompiledQuery(
            stages=[
                RelationalStage(sql=f"SELECT * FROM '{path}' WHERE price > 5"),
                SemanticStage(
                    operations=[
                        {
                            "name": "even",
                            "type": "code_filter",
                            "code": "def transform(doc):\n    return doc['id'] % 2 == 0",
                        }
                    ]
                ),
                RelationalStage(sql="SELECT id FROM _prev ORDER BY id", reads_prev=True),
            ]
        )
        # price>5 -> ids 2..5; even -> 2,4
        assert run_compiled(compiled).to_pylist() == [{"id": 2}, {"id": 4}]


class TestComparisonSplit:
    def test_score_comparison_cost_correct_order(self):
        from docetl.aisql import compile_sql

        stages = compile_sql(
            "SELECT ai_summarize(text) AS summary FROM tickets "
            "WHERE ai_score(text, 'how urgent?') > 0.8"
        ).stages
        # scan, score-map, DuckDB shrink (drops hidden col), select-map, project
        assert [type(s).__name__ for s in stages] == [
            "RelationalStage",
            "SemanticStage",
            "RelationalStage",
            "SemanticStage",
            "RelationalStage",
        ]
        assert stages[1].operations[0]["output"]["schema"] == {"__aicmp_0": "number"}
        assert stages[2].sql == (
            "SELECT * EXCLUDE (__aicmp_0) FROM _prev WHERE __aicmp_0 > 0.8"
        )
        assert stages[3].operations[0]["output"]["schema"] == {"summary": "string"}
        assert stages[4].sql == "SELECT summary FROM _prev"

    def test_score_only_no_select_maps(self):
        from docetl.aisql import compile_sql

        stages = compile_sql(
            "SELECT * FROM t WHERE ai_score(body, 'quality 0-1') >= 0.5"
        ).stages
        # no SELECT-list maps -> no second semantic stage
        assert [type(s).__name__ for s in stages] == [
            "RelationalStage",
            "SemanticStage",
            "RelationalStage",
            "RelationalStage",
        ]
        assert "EXCLUDE (__aicmp_0)" in stages[2].sql

    def test_both_sides_ai_rejected(self):
        import pytest

        from docetl.aisql import compile_sql

        with pytest.raises(NotImplementedError, match="exactly one side"):
            compile_sql(
                "SELECT id FROM t WHERE ai_score(a,'q') > ai_score(b,'q')"
            )


class TestComparisonExecution:
    def test_exclude_and_residual_thread(self, tmp_path):
        # code_map stands in for ai_score: validates the EXCLUDE + residual
        # filter stage and the multi-stage relational threading, no LLM.
        import pyarrow.parquet as pq

        from docetl.aisql import (
            CompiledQuery,
            RelationalStage,
            SemanticStage,
            run_compiled,
        )

        path = tmp_path / "v.parquet"
        pq.write_table(
            pa.Table.from_pylist([{"id": i, "val": i} for i in range(1, 6)]), path
        )
        compiled = CompiledQuery(
            stages=[
                RelationalStage(sql=f"SELECT * FROM '{path}'"),
                SemanticStage(
                    operations=[
                        {
                            "name": "score",
                            "type": "code_map",
                            "code": "def transform(doc):\n    return {'score': doc['val'] * 0.1}",
                        }
                    ]
                ),
                RelationalStage(
                    sql="SELECT * EXCLUDE (score) FROM _prev WHERE score > 0.25",
                    reads_prev=True,
                ),
                RelationalStage(sql="SELECT id FROM _prev ORDER BY id", reads_prev=True),
            ]
        )
        out = run_compiled(compiled).to_pylist()
        assert out == [{"id": 3}, {"id": 4}, {"id": 5}]
        # hidden score column was dropped by EXCLUDE
        assert all("score" not in r for r in out)


class TestReduce:
    def test_group_by_ai_agg(self):
        from docetl.aisql import compile_sql

        stages = compile_sql(
            "SELECT category, ai_agg(review, 'summarize the reviews') AS summary "
            "FROM reviews GROUP BY category"
        ).stages
        assert [type(s).__name__ for s in stages] == [
            "RelationalStage",
            "SemanticStage",
            "RelationalStage",
        ]
        assert stages[0].sql == "SELECT * FROM reviews"
        (op,) = stages[1].operations
        assert op["type"] == "reduce"
        assert op["reduce_key"] == ["category"]
        assert "{{ inputs }}" in op["prompt"] or "for item in inputs" in op["prompt"]
        assert op["output"]["schema"] == {"summary": "string"}
        assert stages[2].sql == "SELECT category, summary FROM _prev"

    def test_multiple_ai_agg_rejected(self):
        import pytest

        from docetl.aisql import compile_sql

        with pytest.raises(NotImplementedError, match="one ai_agg"):
            compile_sql(
                "SELECT k, ai_agg(a, 'x') AS m, ai_agg(b, 'y') AS n FROM t GROUP BY k"
            )

    def test_ai_in_where_with_group_rejected(self):
        import pytest

        from docetl.aisql import compile_sql

        with pytest.raises(NotImplementedError, match="WHERE with GROUP BY"):
            compile_sql(
                "SELECT k, ai_agg(a,'x') AS m FROM t WHERE ai_filter(a,'q') GROUP BY k"
            )


class TestReduceExecution:
    def test_code_reduce_threads(self, tmp_path):
        import pyarrow.parquet as pq

        from docetl.aisql import (
            CompiledQuery,
            RelationalStage,
            SemanticStage,
            run_compiled,
        )

        path = tmp_path / "r.parquet"
        pq.write_table(
            pa.Table.from_pylist(
                [{"cat": "a", "v": 1}, {"cat": "a", "v": 2}, {"cat": "b", "v": 5}]
            ),
            path,
        )
        compiled = CompiledQuery(
            stages=[
                RelationalStage(sql=f"SELECT * FROM '{path}'"),
                SemanticStage(
                    operations=[
                        {
                            "name": "agg",
                            "type": "code_reduce",
                            "reduce_key": ["cat"],
                            "code": "def transform(items):\n    return {'total': sum(i['v'] for i in items)}",
                        }
                    ]
                ),
                RelationalStage(sql="SELECT cat, total FROM _prev ORDER BY cat", reads_prev=True),
            ]
        )
        assert run_compiled(compiled).to_pylist() == [
            {"cat": "a", "total": 3},
            {"cat": "b", "total": 5},
        ]


class TestJoin:
    def test_join_on_ai_match(self):
        from docetl.aisql import compile_sql, JoinStage, RelationalStage

        stages = compile_sql(
            "SELECT * FROM products p JOIN listings l "
            "ON ai_match(p.name, l.title, 'same product?')"
        ).stages
        assert isinstance(stages[0], JoinStage)
        assert stages[0].left_sql == "SELECT * FROM products AS p"
        assert stages[0].right_sql == "SELECT * FROM listings AS l"
        op = stages[0].operation
        assert op["type"] == "equijoin"
        assert "{{ left.name }}" in op["comparison_prompt"]
        assert "{{ right.title }}" in op["comparison_prompt"]
        assert isinstance(stages[1], RelationalStage)
        assert stages[1].sql == "SELECT * FROM _prev"

    def test_non_ai_match_on_rejected(self):
        import pytest

        from docetl.aisql import compile_sql

        with pytest.raises(NotImplementedError, match="ai_match"):
            compile_sql(
                "SELECT * FROM a JOIN b ON ai_score(a.x, 'q') JOIN c ON a.id = c.id"
            )

    def test_join_two_input_wiring(self, tmp_path, monkeypatch):
        # Monkeypatch equijoin to an inner join so we can verify, LLM-free,
        # that run_compiled runs both scans and feeds them to the join.
        import pyarrow.parquet as pq

        import docetl
        from docetl.aisql import CompiledQuery, JoinStage, RelationalStage, run_compiled

        lpath, rpath = tmp_path / "l.parquet", tmp_path / "r.parquet"
        pq.write_table(pa.Table.from_pylist([{"id": 1, "a": "x"}, {"id": 2, "a": "y"}]), lpath)
        pq.write_table(pa.Table.from_pylist([{"id": 1, "b": "P"}, {"id": 2, "b": "Q"}]), rpath)

        seen = {}

        def rows(frame):
            # read the memory dataset directly; from_arrow frames have no
            # operations, so they can't be .collect()'d
            return frame._datasets[frame._first_dataset]["path"]

        def fake_equijoin(self, right, **kw):
            seen["left"] = rows(self)
            seen["right"] = rows(right)
            lj = {r["id"]: r for r in rows(self)}
            merged = [{**lj[r["id"]], **r} for r in rows(right) if r["id"] in lj]
            # needs an op to be executable by to_arrow (op-less frames can't run)
            return docetl.from_list(merged).code_map(
                code="def transform(doc):\n    return {}"
            )

        monkeypatch.setattr(docetl.Frame, "equijoin", fake_equijoin)
        compiled = CompiledQuery(
            stages=[
                JoinStage(
                    left_sql=f"SELECT * FROM '{lpath}'",
                    right_sql=f"SELECT * FROM '{rpath}'",
                    operation={"name": "j", "type": "equijoin", "comparison_prompt": "x"},
                ),
                RelationalStage(sql="SELECT id, a, b FROM _prev ORDER BY id", reads_prev=True),
            ]
        )
        out = run_compiled(compiled).to_pylist()
        assert {r["id"] for r in seen["left"]} == {1, 2}
        assert {r["id"] for r in seen["right"]} == {1, 2}
        assert out == [{"id": 1, "a": "x", "b": "P"}, {"id": 2, "a": "y", "b": "Q"}]


class TestResolve:
    def test_resolve_named_args(self):
        from docetl.aisql import compile_sql

        stages = compile_sql(
            "SELECT * FROM ai_resolve(customers, on := name, prompt := 'same customer?')"
        ).stages
        assert stages[0].sql == "SELECT * FROM customers"
        (op,) = stages[1].operations
        assert op["type"] == "resolve"
        assert "{{ input1.name }}" in op["comparison_prompt"]
        assert "{{ input2.name }}" in op["comparison_prompt"]
        assert op["output"]["schema"] == {"name": "string"}
        assert stages[2].sql == "SELECT * FROM _prev"

    def test_resolve_positional_args(self):
        from docetl.aisql import compile_sql

        stages = compile_sql(
            "SELECT * FROM ai_resolve(customers, 'name', 'same customer?')"
        ).stages
        assert stages[1].operations[0]["type"] == "resolve"

    def test_resolve_missing_args_rejected(self):
        import pytest

        from docetl.aisql import compile_sql

        with pytest.raises(NotImplementedError, match="on:=column"):
            compile_sql("SELECT * FROM ai_resolve(customers)")


class TestPlanIRBridge:
    """Milestone 6: a compiled query's semantic operations are ordinary
    DocETL ops that lift into the plan IR (the substrate the rewrite rules
    and MOAR consume) with no special-casing."""

    def test_select_where_semantic_lifts_and_validates(self):
        from docetl.aisql import compile_sql, semantic_pipelines
        from docetl.plan import lift, validate

        compiled = compile_sql(
            "SELECT ai_summarize(t) AS s FROM x "
            "WHERE a > 1 AND ai_filter(t, 'relevant?')"
        )
        pipelines = semantic_pipelines(compiled)
        assert pipelines  # the filter + map stage
        for cfg in pipelines:
            plan = lift(cfg)
            errors = [i for i in validate(plan) if i.level == "error"]
            assert not errors, errors
            # the AI functions became real, recognized DocETL operators
            assert {n.op_type for n in plan.nodes() if n.op_type} <= {
                "map",
                "filter",
                "reduce",
                "resolve",
                "scan",
            }

    def test_reduce_query_lifts(self):
        from docetl.aisql import compile_sql, semantic_pipelines
        from docetl.plan import lift, validate

        compiled = compile_sql(
            "SELECT k, ai_agg(v, 'summarize') AS s FROM t GROUP BY k"
        )
        (cfg,) = semantic_pipelines(compiled)
        plan = lift(cfg)
        assert not [i for i in validate(plan) if i.level == "error"]
        assert any(n.op_type == "reduce" for n in plan.nodes())

    def test_rewrite_rules_run_over_compiled_ops(self):
        # The plan-IR rewrite engine accepts the compiled pipeline. The
        # AI-SQL compiler already pushed selections across the engine
        # boundary, so the in-pipeline rules typically find nothing to do
        # — the point is they run cleanly on the same IR.
        from docetl.aisql import compile_sql, semantic_pipelines
        from docetl.plan import apply_rewrites_to_config

        compiled = compile_sql("SELECT ai_summarize(t) AS s FROM x")
        (cfg,) = semantic_pipelines(compiled)
        out, applied = apply_rewrites_to_config(cfg)
        assert isinstance(applied, list)  # ran without error


class TestEmptyIntermediate:
    def test_filter_dropping_all_rows_yields_empty_not_crash(self, tmp_path):
        # Regression: an empty semantic result used to lose its schema
        # (from_pylist([])) and crash the next DuckDB stage.
        import pyarrow.parquet as pq

        from docetl.aisql import (
            CompiledQuery,
            RelationalStage,
            SemanticStage,
            run_compiled,
        )

        path = tmp_path / "v.parquet"
        pq.write_table(pa.Table.from_pylist([{"id": 1, "v": 1}, {"id": 2, "v": 2}]), path)
        compiled = CompiledQuery(
            stages=[
                RelationalStage(sql=f"SELECT * FROM '{path}'"),
                SemanticStage(
                    operations=[
                        {
                            "name": "none",
                            "type": "code_filter",
                            "code": "def transform(doc):\n    return False",
                        }
                    ]
                ),
                RelationalStage(sql="SELECT id FROM _prev", reads_prev=True),
            ]
        )
        out = run_compiled(compiled)
        assert out.num_rows == 0
        assert "id" in out.column_names
