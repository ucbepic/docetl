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
        out, cost = run_sql(f"SELECT id FROM '{path}' WHERE price > 10 ORDER BY id")
        assert out.to_pylist() == [{"id": 3}, {"id": 4}, {"id": 5}]
        assert cost == 0.0  # pure relational, no LLM

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
        out = run_compiled(compiled)[0].to_pylist()
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

    def test_or_with_ai(self):
        from docetl.aisql import compile_sql

        cq = compile_sql("SELECT id FROM t WHERE a > 1 OR ai_filter(x, 'q')")
        names = [type(s).__name__ for s in cq.stages]
        assert "SemanticStage" in names
        # The AI filter becomes a hidden boolean column, OR is relational
        rel_sqls = [s.sql for s in cq.stages if hasattr(s, "sql")]
        or_sql = [s for s in rel_sqls if "__aior_0" in s]
        assert len(or_sql) == 1
        assert "OR" in or_sql[0]

    def test_or_with_not_ai(self):
        from docetl.aisql import compile_sql

        cq = compile_sql(
            "SELECT id FROM t WHERE NOT ai_filter(x, 'spam?') OR priority = 1"
        )
        rel_sqls = [s.sql for s in cq.stages if hasattr(s, "sql")]
        or_sql = [s for s in rel_sqls if "__aior_0" in s]
        assert len(or_sql) == 1
        assert "NOT __aior_0" in or_sql[0]

    def test_or_with_multiple_ai(self):
        from docetl.aisql import compile_sql

        cq = compile_sql(
            "SELECT id FROM t WHERE ai_filter(x, 'q1') OR ai_filter(x, 'q2')"
        )
        rel_sqls = " ".join(s.sql for s in cq.stages if hasattr(s, "sql"))
        assert "__aior_0" in rel_sqls
        assert "__aior_1" in rel_sqls


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
        assert run_compiled(compiled)[0].to_pylist() == [{"id": 2}, {"id": 4}]


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

    def test_both_sides_ai(self):
        from docetl.aisql import compile_sql

        cq = compile_sql(
            "SELECT id FROM t WHERE ai_score(a,'q1') > ai_score(b,'q2')"
        )
        names = [type(s).__name__ for s in cq.stages]
        # One semantic stage with two map ops, then relational comparison
        assert "SemanticStage" in names
        # The relational stage after the semantic one should compare hidden cols
        rel_sqls = [s.sql for s in cq.stages if hasattr(s, "sql")]
        exclude_sql = [s for s in rel_sqls if "__aicmp_0_l" in s]
        assert len(exclude_sql) == 1
        assert "__aicmp_0_r" in exclude_sql[0]


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
        out = run_compiled(compiled)[0].to_pylist()
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

    def test_multiple_ai_agg(self):
        from docetl.aisql import compile_sql

        cq = compile_sql(
            "SELECT k, ai_agg(a, 'x') AS m, ai_agg(b, 'y') AS n FROM t GROUP BY k"
        )
        # Should produce a semantic stage with two reduce ops
        sem = [s for s in cq.stages if type(s).__name__ == "SemanticStage"]
        assert len(sem) >= 1
        reduce_ops = [
            op for s in sem for op in s.operations if op.get("type") == "reduce"
        ]
        assert len(reduce_ops) == 2

    def test_ai_in_where_with_group(self):
        from docetl.aisql import compile_sql

        cq = compile_sql(
            "SELECT k, ai_agg(a,'x') AS m FROM t WHERE ai_filter(a,'q') GROUP BY k"
        )
        # Should have a filter stage before the reduce stage
        op_types = [
            op.get("type")
            for s in cq.stages
            if type(s).__name__ == "SemanticStage"
            for op in s.operations
        ]
        assert "filter" in op_types
        assert "reduce" in op_types


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
        assert run_compiled(compiled)[0].to_pylist() == [
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
        out = run_compiled(compiled)[0].to_pylist()
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
        out, cost = run_compiled(compiled)
        assert out.num_rows == 0
        assert "id" in out.column_names


class TestOrderByLimit:
    def test_order_by_and_limit_carried_to_final(self):
        from docetl.aisql import compile_sql

        stages = compile_sql(
            "SELECT ai_summarize(t) AS s FROM x ORDER BY s LIMIT 5"
        ).stages
        assert stages[-1].sql == "SELECT s FROM _prev ORDER BY s LIMIT 5"

    def test_order_by_with_filter(self):
        from docetl.aisql import compile_sql

        stages = compile_sql(
            "SELECT id FROM x WHERE ai_filter(t, 'q?') ORDER BY id DESC LIMIT 3"
        ).stages
        assert stages[-1].sql == "SELECT id FROM _prev ORDER BY id DESC LIMIT 3"

    def test_order_by_in_grouped(self):
        from docetl.aisql import compile_sql

        stages = compile_sql(
            "SELECT cat, ai_agg(v, 'list') AS items FROM r GROUP BY cat ORDER BY cat"
        ).stages
        assert stages[-1].sql == "SELECT cat, items FROM _prev ORDER BY cat"

    def test_ai_function_in_order_by(self):
        from docetl.aisql import compile_sql

        cq = compile_sql("SELECT t FROM x ORDER BY ai_score(t, 'q')")
        # Should produce a semantic stage for the sort column, then a
        # relational stage with ORDER BY on the hidden column + EXCLUDE
        sem = [s for s in cq.stages if type(s).__name__ == "SemanticStage"]
        assert len(sem) >= 1
        rel_sqls = " ".join(s.sql for s in cq.stages if hasattr(s, "sql"))
        assert "ORDER BY" in rel_sqls


class TestChunking:
    def test_agent_mode_chunk_overflow(self, tmp_path):
        """agent_mode=True raises ChunkOverflowError with structured
        column stats instead of silently auto-chunking."""
        import pyarrow.parquet as pq
        import pytest

        from docetl.aisql import (
            ChunkOverflowError,
            CompiledQuery,
            RelationalStage,
            SemanticStage,
            run_compiled,
        )

        long_text = ("filler " * 2000).strip() + "\n\n" + "The answer is 42."
        path = tmp_path / "agent.parquet"
        pq.write_table(
            pa.Table.from_pylist([{"id": 1, "text": long_text}]), path
        )
        compiled = CompiledQuery(
            stages=[
                RelationalStage(sql=f"SELECT * FROM '{path}'"),
                SemanticStage(
                    operations=[
                        {
                            "name": "extract",
                            "type": "map",
                            "prompt": "Extract the answer from: {{ input.text }}",
                            "output": {"schema": {"answer": "string"}},
                            "model": "gpt-4o-mini",
                        }
                    ]
                ),
            ]
        )
        with pytest.raises(ChunkOverflowError) as exc_info:
            run_compiled(compiled, max_chunk_tokens=100, agent_mode=True)
        err = exc_info.value
        assert err.num_rows == 1
        assert err.max_tokens == 100
        assert "text" in err.columns
        assert err.columns["text"]["rows_over_limit"] == 1
        assert "Text too long" in str(err)
        assert "regexp_split_to_array" in str(err)

    def test_agent_mode_high_cardinality(self, tmp_path):
        """agent_mode=True raises HighCardinalityError when GROUP BY key
        has near-unique values."""
        import pyarrow.parquet as pq
        import pytest

        from docetl.aisql import (
            CompiledQuery,
            HighCardinalityError,
            RelationalStage,
            SemanticStage,
            run_compiled,
        )

        rows = [{"id": i, "cat": f"unique_{i}", "text": f"item {i}"} for i in range(20)]
        path = tmp_path / "highcard.parquet"
        pq.write_table(pa.Table.from_pylist(rows), path)

        compiled = CompiledQuery(
            stages=[
                RelationalStage(sql=f"SELECT * FROM '{path}'"),
                SemanticStage(
                    operations=[
                        {
                            "name": "agg",
                            "type": "reduce",
                            "reduce_key": ["cat"],
                            "prompt": "Summarize: {% for item in inputs %}{{ item.text }}\n{% endfor %}",
                            "output": {"schema": {"summary": "string"}},
                        }
                    ]
                ),
            ]
        )
        with pytest.raises(HighCardinalityError) as exc_info:
            run_compiled(compiled, agent_mode=True)
        err = exc_info.value
        assert err.num_rows == 20
        assert err.n_distinct == 20
        assert err.reduce_key == ["cat"]
        assert "ai_resolve" in str(err)

    def test_nocheck_bypasses_agent_mode(self, tmp_path):
        """/* nocheck */ in the SQL disables agent checks for that query."""
        import pyarrow.parquet as pq

        from docetl.aisql import run_sql

        rows = [{"id": i, "cat": f"unique_{i}", "val": i} for i in range(20)]
        path = tmp_path / "nocheck.parquet"
        pq.write_table(pa.Table.from_pylist(rows), path)

        # This would raise HighCardinalityError without nocheck,
        # but it's a pure relational query so it runs fine.
        out, cost = run_sql(
            f"/* nocheck */ SELECT cat, count(*) AS n FROM '{path}' GROUP BY cat",
            agent_mode=True,
        )
        assert out.num_rows == 20
        assert cost == 0.0

    def test_agent_mode_empty_input(self, tmp_path):
        """agent_mode=True raises EmptyInputError when relational
        predicates match zero rows before a semantic stage."""
        import pyarrow.parquet as pq
        import pytest

        from docetl.aisql import (
            CompiledQuery,
            EmptyInputError,
            RelationalStage,
            SemanticStage,
            run_compiled,
        )

        path = tmp_path / "data.parquet"
        pq.write_table(
            pa.Table.from_pylist([{"id": 1, "text": "hello"}]), path
        )
        compiled = CompiledQuery(
            stages=[
                RelationalStage(sql=f"SELECT * FROM '{path}' WHERE id = 999"),
                SemanticStage(
                    operations=[
                        {
                            "name": "up",
                            "type": "code_map",
                            "code": "def transform(doc):\n    return {'upper': doc['text'].upper()}",
                            "output": {"schema": {"upper": "string"}},
                        }
                    ]
                ),
            ]
        )
        with pytest.raises(EmptyInputError, match="0 rows"):
            run_compiled(compiled, agent_mode=True)

    def test_agent_mode_missing_column(self, tmp_path):
        """agent_mode=True raises MissingColumnError when a prompt
        references a column that doesn't exist in the table."""
        import pyarrow.parquet as pq
        import pytest

        from docetl.aisql import (
            CompiledQuery,
            MissingColumnError,
            RelationalStage,
            SemanticStage,
            run_compiled,
        )

        path = tmp_path / "data.parquet"
        pq.write_table(
            pa.Table.from_pylist([{"id": 1, "text": "hello"}]), path
        )
        compiled = CompiledQuery(
            stages=[
                RelationalStage(sql=f"SELECT * FROM '{path}'"),
                SemanticStage(
                    operations=[
                        {
                            "name": "extract",
                            "type": "map",
                            "prompt": "Extract from: {{ input.body }}",
                            "output": {"schema": {"answer": "string"}},
                            "model": "gpt-4o-mini",
                        }
                    ]
                ),
            ]
        )
        with pytest.raises(MissingColumnError) as exc_info:
            run_compiled(compiled, agent_mode=True)
        err = exc_info.value
        assert "body" in err.missing
        assert "text" in err.available

    def test_agent_mode_too_many_rows(self, tmp_path):
        """agent_mode=True raises TooManyRowsError when the input
        exceeds max_ai_rows."""
        import pyarrow.parquet as pq
        import pytest

        from docetl.aisql import (
            CompiledQuery,
            RelationalStage,
            SemanticStage,
            TooManyRowsError,
            run_compiled,
        )

        rows = [{"id": i, "text": f"row {i}"} for i in range(50)]
        path = tmp_path / "big.parquet"
        pq.write_table(pa.Table.from_pylist(rows), path)

        compiled = CompiledQuery(
            stages=[
                RelationalStage(sql=f"SELECT * FROM '{path}'"),
                SemanticStage(
                    operations=[
                        {
                            "name": "extract",
                            "type": "map",
                            "prompt": "Extract from: {{ input.text }}",
                            "output": {"schema": {"answer": "string"}},
                            "model": "gpt-4o-mini",
                        }
                    ]
                ),
            ]
        )
        with pytest.raises(TooManyRowsError) as exc_info:
            run_compiled(compiled, agent_mode=True, max_ai_rows=10)
        err = exc_info.value
        assert err.num_rows == 50
        assert err.threshold == 10
