"""Operator plan-trait and field-read-extractor soundness tests."""

import pytest

from docetl.operations import get_operation
from docetl.operations.base import BaseOperation, Cardinality
from docetl.operations.code_operations import extract_doc_field_reads
from docetl.utils import extract_input_field_reads, extract_template_field_reads


class TestJinjaFieldReadExtractor:
    def test_simple_reads(self):
        assert extract_input_field_reads("{{ input.text }}") == {"text"}
        assert extract_input_field_reads('{{ input["a key"] }}') == {"a key"}

    def test_condition_reads_are_seen(self):
        # extract_jinja_variables misses these; the sound extractor must not.
        assert extract_input_field_reads(
            "{% if input.flag %}yes{% endif %} {{ input.text }}"
        ) == {"flag", "text"}

    def test_loop_over_field(self):
        assert extract_input_field_reads(
            "{% for t in input.tags %}{{ t }}{% endfor %}"
        ) == {"tags"}

    def test_nested_access_reads_top_field(self):
        assert extract_input_field_reads("{{ input.a.b }}") == {"a"}

    def test_whole_object_use_fails_closed(self):
        assert extract_input_field_reads("{{ input }}") is None
        assert extract_input_field_reads("{{ input | tojson }}") is None
        assert extract_input_field_reads("{% set d = input %}{{ d.x }}") is None

    def test_dynamic_subscript_fails_closed(self):
        assert extract_input_field_reads("{{ input[key] }}") is None

    def test_non_jinja_fails_closed(self):
        # The runtime appends the whole document to non-Jinja prompts.
        assert extract_input_field_reads("summarize this document") is None

    def test_non_string_fails_closed(self):
        assert extract_input_field_reads(None) is None
        assert extract_input_field_reads(42) is None

    def test_other_variables_ignored(self):
        assert extract_input_field_reads(
            "{{ input.x }} {{ other.y }} {% for i in items %}{{ i }}{% endfor %}"
        ) == {"x"}

    def test_custom_var(self):
        assert extract_input_field_reads(
            "{{ left.k }} vs {{ right.k }}", var="left"
        ) == {"k"}

    def test_template_variant_treats_plain_text_as_empty(self):
        # Auxiliary templates (gleaning) don't get the document appended.
        assert extract_template_field_reads("just words") == frozenset()
        assert extract_template_field_reads("{{ input.x }}") == {"x"}
        assert extract_template_field_reads("{{ input }}") is None


class TestCodeFieldReadExtractor:
    def test_subscript_and_get(self):
        code = 'def transform(doc):\n    return {"y": doc["x"] + doc.get("z", 0)}'
        assert extract_doc_field_reads(code) == {"x", "z"}

    def test_param_name_is_positional(self):
        code = 'def transform(d):\n    return {"y": d["x"]}'
        assert extract_doc_field_reads(code) == {"x"}

    def test_whole_doc_use_fails_closed(self):
        assert extract_doc_field_reads("def transform(doc):\n    return dict(doc)") is None
        assert extract_doc_field_reads("def transform(doc):\n    return {**doc}") is None
        assert (
            extract_doc_field_reads("def transform(doc):\n    d = doc\n    return {'y': d['x']}")
            is None
        )

    def test_dynamic_key_fails_closed(self):
        assert (
            extract_doc_field_reads("def transform(doc):\n    k = 'x'\n    return {'y': doc[k]}")
            is None
        )

    def test_iteration_fails_closed(self):
        assert (
            extract_doc_field_reads(
                "def transform(doc):\n    return {k: v for k, v in doc.items()}"
            )
            is None
        )

    def test_callable_def(self):
        def fn(doc):
            return {"n": doc["text"]}

        assert extract_doc_field_reads(fn) == {"text"}

    def test_unparseable_fails_closed(self):
        assert extract_doc_field_reads("not python {{{") is None
        assert extract_doc_field_reads(None) is None


def _op(op_type):
    return get_operation(op_type)


class TestMapTraits:
    BASE = {
        "name": "m",
        "type": "map",
        "prompt": "Summarize {{ input.text }}",
        "output": {"schema": {"summary": "string"}},
    }

    def test_plain_map(self):
        cls = _op("map")
        assert cls.cardinality(self.BASE) == Cardinality.ONE_TO_ONE
        assert cls.fields_read(self.BASE) == {"text"}
        assert cls.fields_written(self.BASE) == {"summary"}
        assert cls.is_llm(self.BASE)
        assert cls.is_row_local(self.BASE)
        assert cls.preserves_order(self.BASE)

    @pytest.mark.parametrize(
        "extra",
        [
            {"skip_on_error": True},
            {"limit": 5},  # truncates *inputs*
            {"validate": ["len(output['summary']) > 0"]},  # failures drop rows
            {"tools": [{"code": "def f(): pass", "function": {"name": "f"}}]},
        ],
    )
    def test_cardinality_downgrades(self, extra):
        cls = _op("map")
        assert cls.cardinality({**self.BASE, **extra}) == Cardinality.MANY_TO_MANY

    def test_non_jinja_prompt_reads_whole_row(self):
        cls = _op("map")
        cfg = {**self.BASE, "prompt": "summarize this"}
        assert cls.fields_read(cfg) is None

    def test_retriever_and_batch_prompt_read_whole_row(self):
        cls = _op("map")
        assert cls.fields_read({**self.BASE, "retriever": "r"}) is None
        assert cls.fields_read({**self.BASE, "batch_prompt": "b {{ inputs }}"}) is None

    def test_tools_make_writes_unknown(self):
        cls = _op("map")
        cfg = {**self.BASE, "tools": [{"code": "x", "function": {"name": "f"}}]}
        assert cls.fields_read(cfg) is None
        assert cls.fields_written(cfg) is None

    def test_calibrate_not_row_local(self):
        cls = _op("map")
        assert not cls.is_row_local({**self.BASE, "calibrate": True})

    def test_gleaning_reads_included(self):
        cls = _op("map")
        cfg = {
            **self.BASE,
            "gleaning": {
                "num_rounds": 1,
                "validation_prompt": "check against {{ input.reference }}",
            },
        }
        assert cls.fields_read(cfg) == {"text", "reference"}

    def test_drop_keys_count_as_writes(self):
        cls = _op("map")
        cfg = {**self.BASE, "drop_keys": ["raw"]}
        assert cls.fields_written(cfg) == {"summary", "raw"}

    def test_drop_keys_only_map_is_not_llm(self):
        cls = _op("map")
        cfg = {"name": "m", "type": "map", "drop_keys": ["raw"]}
        assert not cls.is_llm(cfg)
        assert cls.fields_read(cfg) == frozenset()

    def test_observability_key_is_written(self):
        cls = _op("map")
        cfg = {**self.BASE, "enable_observability": True}
        assert "_observability_m" in cls.fields_written(cfg)


class TestFilterTraits:
    BASE = {
        "name": "f",
        "type": "filter",
        "prompt": "Keep? {{ input.category }}",
        "output": {"schema": {"keep": "boolean"}},
    }

    def test_selection_even_with_row_droppers(self):
        cls = _op("filter")
        assert cls.cardinality(self.BASE) == Cardinality.SELECTION
        assert (
            cls.cardinality({**self.BASE, "skip_on_error": True})
            == Cardinality.SELECTION
        )

    def test_decision_key_counts_as_write(self):
        # The popped decision key removes any same-named input field.
        cls = _op("filter")
        assert cls.fields_written(self.BASE) == {"keep"}

    def test_short_explanation_is_written(self):
        cls = _op("filter")
        cfg = {
            **self.BASE,
            "output": {"schema": {"keep": "boolean", "_short_explanation": "string"}},
        }
        assert cls.fields_written(cfg) == {"keep", "_short_explanation"}

    def test_reads(self):
        cls = _op("filter")
        assert cls.fields_read(self.BASE) == {"category"}


class TestCodeOpTraits:
    def test_code_map(self):
        cls = _op("code_map")
        cfg = {"name": "c", "type": "code_map", "code": "def transform(doc):\n    return {'y': doc['x']}"}
        assert cls.cardinality(cfg) == Cardinality.ONE_TO_ONE
        assert cls.cardinality({**cfg, "limit": 2}) == Cardinality.MANY_TO_MANY
        assert cls.fields_read(cfg) == {"x"}
        assert cls.fields_written(cfg) is None  # arbitrary code
        assert cls.is_row_local(cfg) and cls.preserves_order(cfg)
        assert not cls.is_llm(cfg)

    def test_code_filter(self):
        cls = _op("code_filter")
        cfg = {"name": "c", "type": "code_filter", "code": "def transform(doc):\n    return doc['x'] > 1"}
        assert cls.cardinality(cfg) == Cardinality.SELECTION
        assert cls.fields_read(cfg) == {"x"}
        assert cls.fields_written(cfg) == frozenset()

    def test_code_reduce(self):
        cls = _op("code_reduce")
        cfg = {"name": "c", "type": "code_reduce", "code": "def transform(items):\n    return {}"}
        assert cls.cardinality(cfg) == Cardinality.MANY_TO_ONE
        assert cls.fields_read(cfg) is None


class TestSampleTraits:
    def test_first(self):
        cls = _op("sample")
        cfg = {"name": "s", "type": "sample", "method": "first", "samples": 3}
        assert cls.cardinality(cfg) == Cardinality.SELECTION
        assert cls.fields_read(cfg) == frozenset()
        assert cls.is_deterministic(cfg)
        assert cls.preserves_order(cfg)

    def test_stratified_first_not_order_preserving(self):
        cls = _op("sample")
        cfg = {
            "name": "s", "type": "sample", "method": "first",
            "samples": 3, "stratify_key": "grp",
        }
        assert not cls.preserves_order(cfg)
        assert cls.fields_read(cfg) == {"grp"}

    def test_uniform_determinism_needs_seed(self):
        cls = _op("sample")
        cfg = {"name": "s", "type": "sample", "method": "uniform", "samples": 3}
        assert not cls.is_deterministic(cfg)
        assert cls.is_deterministic({**cfg, "random_state": 42})
        assert not cls.preserves_order(cfg)

    def test_embedding_methods_fail_closed(self):
        cls = _op("sample")
        cfg = {
            "name": "s", "type": "sample", "method": "outliers",
            "method_kwargs": {"embedding_keys": ["text"], "std": 2},
        }
        assert cls.fields_read(cfg) is None
        assert cls.is_llm(cfg)


class TestStructuralOpTraits:
    def test_split(self):
        cls = _op("split")
        cfg = {
            "name": "sp", "type": "split", "split_key": "text",
            "method": "token_count", "method_kwargs": {"num_tokens": 10},
        }
        assert cls.cardinality(cfg) == Cardinality.ONE_TO_MANY
        assert cls.fields_read(cfg) == {"text"}
        assert cls.fields_written(cfg) == {"text_chunk", "sp_id", "sp_chunk_num"}
        assert not cls.is_deterministic(cfg)  # fresh uuid per document

    def test_unnest(self):
        cls = _op("unnest")
        cfg = {"name": "u", "type": "unnest", "unnest_key": "tags"}
        assert cls.cardinality(cfg) == Cardinality.ONE_TO_MANY
        assert cls.fields_read(cfg) == {"tags"}
        assert cls.fields_written(cfg) == {"tags"}

    def test_gather_not_row_local(self):
        cls = _op("gather")
        cfg = {
            "name": "g", "type": "gather", "content_key": "chunk",
            "doc_id_key": "doc_id", "order_key": "num",
        }
        assert cls.cardinality(cfg) == Cardinality.ONE_TO_ONE
        assert not cls.is_row_local(cfg)  # reads neighboring chunks
        assert cls.fields_read(cfg) == {"chunk", "doc_id", "num"}
        assert cls.fields_written(cfg) == {"chunk_rendered"}

    def test_rank_reorders(self):
        cls = _op("rank")
        cfg = {
            "name": "r", "type": "rank", "prompt": "p",
            "input_keys": ["text"], "direction": "desc",
        }
        assert cls.cardinality(cfg) == Cardinality.ONE_TO_ONE
        assert not cls.preserves_order(cfg)
        assert cls.fields_written(cfg) == {"_rank"}

    def test_add_uuid_not_deterministic(self):
        cls = _op("add_uuid")
        cfg = {"name": "ids", "type": "add_uuid"}
        assert cls.cardinality(cfg) == Cardinality.ONE_TO_ONE
        assert cls.fields_written(cfg) == {"ids_id"}
        assert not cls.is_deterministic(cfg)


class TestConservativeDefaults:
    def test_base_defaults_block_everything(self):
        cfg = {"name": "x", "type": "anything"}
        assert BaseOperation.cardinality(cfg) == Cardinality.MANY_TO_MANY
        assert BaseOperation.fields_read(cfg) is None
        assert BaseOperation.fields_written(cfg) is None
        assert not BaseOperation.is_llm(cfg)
        assert not BaseOperation.is_deterministic(cfg)
        assert not BaseOperation.is_row_local(cfg)
        assert not BaseOperation.preserves_order(cfg)

    def test_reduce_blocks_rewrites(self):
        cls = _op("reduce")
        cfg = {
            "name": "r", "type": "reduce", "reduce_key": "k",
            "prompt": "{{ inputs }}", "output": {"schema": {"s": "string"}},
        }
        assert cls.cardinality(cfg) == Cardinality.MANY_TO_ONE
        assert cls.fields_read(cfg) is None
        assert cls.fields_written(cfg) is None
