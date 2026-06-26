"""Plan schema propagation: parity with Frame.schema() plus join merging."""

from docetl.frame import from_list
from docetl.plan import lift, output_schema, propagate_schemas


def plan_schema(frame):
    return output_schema(lift(frame._build_config(checkpoint=False)))


class TestFrameSchemaParity:
    def test_structural_ops_tracked(self):
        frame = (
            from_list([{"text": "a b c"}])
            .map(prompt="p {{ input.text }}",
                 output={"schema": {"summary": "string", "tags": "list[string]"}})
            .unnest(unnest_key="tags")
            .split("sp", split_key="summary", method="token_count",
                   method_kwargs={"num_tokens": 10})
            .gather("g", content_key="summary_chunk", doc_id_key="sp_id",
                    order_key="sp_chunk_num")
            .extract("ex", prompt="e {{ input.summary }}", document_keys=["summary"])
        )
        assert plan_schema(frame) == frame.schema()
        assert plan_schema(frame)["summary_extracted_ex"] == "string"

    def test_filter_key_consumed(self):
        frame = (
            from_list([{"x": 1}])
            .map(prompt="p {{ input.x }}", output={"schema": {"summary": "string"}})
            .filter(prompt="f {{ input.summary }}", output={"schema": {"keep": "boolean"}})
        )
        assert plan_schema(frame) == frame.schema() == {"summary": "string"}

    def test_drop_keys_still_apply(self):
        frame = from_list([{"x": 1}]).map(
            prompt="p {{ input.x }}",
            output={"schema": {"a": "string", "b": "string"}},
            drop_keys=["b"],
        )
        assert plan_schema(frame) == frame.schema() == {"a": "string"}

    def test_cluster_and_code_ops(self):
        frame = (
            from_list([{"t": "x"}])
            .cluster("cl", embedding_keys=["t"], summary_schema={"s": "string"},
                     summary_prompt="p {{ inputs }}")
            .code_map("cm", code="def transform(doc):\n    return {'unknowable': 1}")
        )
        assert plan_schema(frame) == frame.schema()


class TestJoinSchemas:
    def test_join_merges_sides(self):
        left = from_list([{"k": 1}], name="l").map(
            "lm", prompt="x {{ input.k }}", output={"schema": {"a": "string"}})
        right = from_list([{"k": 2}], name="r").map(
            "rm", prompt="y {{ input.k }}", output={"schema": {"b": "string"}})
        joined = left.equijoin(right, "j", comparison_prompt="c {{ left.k }} {{ right.k }}")

        schema = plan_schema(joined)
        assert schema["a"] == "string" and schema["b"] == "string"

    def test_join_collision_warns(self):
        left = from_list([{"k": 1}], name="l").map(
            "lm", prompt="x {{ input.k }}", output={"schema": {"a": "string"}})
        right = from_list([{"k": 2}], name="r").map(
            "rm", prompt="y {{ input.k }}", output={"schema": {"a": "integer"}})
        joined = left.equijoin(right, "j", comparison_prompt="c {{ left.k }} {{ right.k }}")

        plan = lift(joined._build_config(checkpoint=False))
        _, _, issues = propagate_schemas(plan)
        assert any(i.level == "warning" and "'a'" in i.message for i in issues)


class TestRemovedFieldTracking:
    def test_rewrite_resets_removal(self):
        # A field dropped then re-written is no longer "removed".
        frame = (
            from_list([{"raw": "x"}])
            .map("d", prompt="p {{ input.raw }}", output={"schema": {"s": "string"}},
                 drop_keys=["raw"])
            .map("rewrite", prompt="q {{ input.s }}", output={"schema": {"raw": "string"}})
            .code_filter("reads_raw", code="def transform(doc):\n    return doc['raw']")
        )
        from docetl.plan import validate

        issues = validate(lift(frame._build_config(checkpoint=False)))
        assert not any(i.level == "error" for i in issues)

    def test_unknown_writer_resets_removal(self):
        # code_map may re-add anything: no definite-removal error after it.
        frame = (
            from_list([{"raw": "x"}])
            .map("d", prompt="p {{ input.raw }}", output={"schema": {"s": "string"}},
                 drop_keys=["raw"])
            .code_map("cm", code="def transform(doc):\n    return {'raw': 1}")
            .code_filter("reads_raw", code="def transform(doc):\n    return doc['raw']")
        )
        from docetl.plan import validate

        issues = validate(lift(frame._build_config(checkpoint=False)))
        assert not any(i.level == "error" for i in issues)
