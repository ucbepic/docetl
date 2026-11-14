"""Tests for the Quail execution engine (docetl/engines/quail).

Three layers:

1. Prompt parsing, eligibility, and SQL compilation, with no server.
2. A fake Quail Server on ``pyarrow.flight.FlightServerBase``, which checks
   the client calls and how result rows map back to DocETL rows.
3. A real ``quail-server`` with Quail's oracle worker (no GPU), when
   ``QUAIL_REPO`` points to a Quail checkout built with ``tools/build.sh``.

``tests/quail_e2e/modal_e2e.py`` runs the engine against a real model on a
Modal H100 (not collected by pytest).
"""

from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.flight as flight
import pytest

from docetl.engines.quail import QuailUnsupported, check_eligible, compile_run, engine_for
from docetl.engines.quail.compile import OutputColumn, convert
from docetl.engines.quail.prompts import QuailPromptError, parse_prompt

ENGINE = {"type": "quail", "endpoint": "grpc://127.0.0.1:1"}

FILTER = {
    "name": "ending",
    "type": "filter",
    "prompt": "Does the review discuss the ending? {{ input.body }}",
    "output": {"schema": {"discusses_ending": "bool"}},
}
MAP = {
    "name": "label",
    "type": "map",
    "prompt": "Label the review: {{ input.body }}",
    "output": {"schema": {"kind": "enum[praise, complaint]", "summary": "string", "stars": "int"}},
}


# ── 1. Parsing, eligibility, compilation ─────────────────────────────


def test_parse_prompt_maps_fields_to_placeholders():
    prompt = parse_prompt("Compare {{ input.a }} with {{ input['b'] }}; is {{ input.a }} longer?")
    assert prompt.template == "Compare {0} with {1}; is {0} longer?"
    assert prompt.fields == ("a", "b")
    assert prompt.instruction() == "Compare [a] with [b]; is [a] longer?"


@pytest.mark.parametrize(
    "template, construct",
    [
        ("Summarize {{ input.body | upper }}", "a filter"),
        ("{% for c in input.comments %}{{ c }}{% endfor %}", "a for loop"),
        ("{% if input.lang == 'fr' %}Translate.{% endif %}{{ input.body }}", "an if block"),
        ("Read {{ document }}", "a variable other than input"),
        ("Return JSON like {\"a\": 1} for {{ input.body }}", "brace"),
        ("No fields at all", "reads no field"),
        ("Read {{ input['two words'] }}", "not a plain column name"),
    ],
)
def test_parse_prompt_names_the_construct_it_refuses(template, construct):
    with pytest.raises(QuailPromptError, match=construct):
        parse_prompt(template)


def test_parse_reduce_prompt_takes_one_loop_over_inputs():
    from docetl.engines.quail.prompts import parse_reduce_prompt

    prompt = parse_reduce_prompt(
        "Summarize these reviews.{% for r in inputs %}\nReview: {{ r.body }}{% endfor %}\nBe brief."
    )
    assert prompt.template == "Summarize these reviews.{0}\nBe brief."
    assert prompt.item == (("text", "\nReview: "), ("field", "body"))
    assert prompt.fields == ("body",)
    assert prompt.instruction() == "Summarize these reviews.[inputs] Be brief."
    for bad, reason in [
        ("{{ inputs }}", "outside it"),
        ("{% for r in inputs %}{{ r.body | upper }}{% endfor %}", "a filter"),
        ("{% for r in rows %}{{ r.body }}{% endfor %}", "for item in inputs"),
        ("{% for r in inputs %}{% if r.x %}{{ r.body }}{% endif %}{% endfor %}", "an if block"),
    ]:
        with pytest.raises(QuailPromptError, match=reason):
            parse_reduce_prompt(bad)


REDUCE = {
    "name": "per_movie",
    "type": "reduce",
    "reduce_key": "movie",
    "prompt": "Summarize the reviews of one movie.{% for r in inputs %}\n- {{ r.body }}{% endfor %}",
    "output": {"schema": {"verdict": "enum[good, bad]", "summary": "string"}},
}


def test_compile_run_groups_a_reduce_after_a_filter():
    compiled = compile_run([FILTER, REDUCE], "docs")
    assert compiled.grouped
    assert compiled.input_fields == ["body", "movie"]
    assert compiled.group_reads == (["movie"], ["body"])
    assert (
        "c2 AS (SELECT c1.`movie` AS `movie`, "
        "AI.CLASSIFY(PROMPT('Summarize the reviews of one movie.{0} Answer only the field verdict.', "
        "STRING_AGG(CONCAT('\\n- ', c1.`body`), '\\n' ORDER BY c1.`_docetl_id`)), ['good', 'bad']) AS `verdict`, "
        "AI.MAP(STRING_AGG(CONCAT('\\n- ', c1.`body`), '\\n' ORDER BY c1.`_docetl_id`), "
        "'Summarize the reviews of one movie.[inputs] Answer only the field summary.') AS `summary` "
        "FROM c1 GROUP BY c1.`movie`)"
    ) in compiled.sql
    assert compiled.sql.endswith("SELECT c2.`movie` AS `movie`, c2.`verdict` AS `verdict`, c2.`summary` AS `summary` FROM c2")


def test_compile_run_reduces_everything_into_one_group():
    compiled = compile_run([{**REDUCE, "reduce_key": "_all"}], "docs")
    assert "GROUP BY" not in compiled.sql
    assert compiled.group_reads == ([], ["body"])


def test_engine_for_resolves_pipeline_and_operation_settings():
    pipeline = {"engine": {"type": "quail", "endpoint": "grpc://h:1"}}
    assert engine_for({}, pipeline) == {
        "type": "quail",
        "endpoint": "grpc://h:1",
        "fallback": "litellm",
        "max_group_tokens": 24_000,
        "gpu_cost_per_hour": 0.0,
    }
    assert engine_for({"engine": "litellm"}, pipeline) is None
    assert engine_for({}, {}) is None
    assert engine_for({"engine": {"type": "quail", "endpoint": "grpc://o:2", "fallback": "strict"}}, {})[
        "fallback"
    ] == "strict"
    with pytest.raises(ValueError, match="endpoint"):
        engine_for({"engine": "quail"}, {})


@pytest.mark.parametrize(
    "change, reason",
    [
        ({"gleaning": {"num_rounds": 1}}, "gleaning"),
        ({"type": "equijoin"}, "type 'equijoin'"),
        ({"output": {"schema": {"keep": "string"}}}, "one boolean output"),
        ({"prompt": "{{ input.body | trim }}"}, "a filter"),
    ],
)
def test_check_eligible_refuses_operations_with_no_quail_form(change, reason):
    with pytest.raises(QuailUnsupported, match=reason):
        check_eligible({**FILTER, **change})


def test_compile_run_fuses_a_filter_and_a_map_into_one_query():
    compiled = compile_run([FILTER, MAP], "docs")
    assert compiled.input_fields == ["body"]
    assert compiled.dropped == ["discusses_ending"]
    assert compiled.sql == (
        "WITH c0 AS (SELECT * FROM `docs`),\n"
        "c1 AS (SELECT c0.`_docetl_id` AS `_docetl_id`, c0.`body` AS `body` FROM c0 "
        "WHERE AI.IF(PROMPT('Does the review discuss the ending? {0}', c0.`body`))),\n"
        "c2 AS (SELECT c1.`_docetl_id` AS `_docetl_id`, c1.`body` AS `body`, "
        "AI.CLASSIFY(PROMPT('Label the review: {0} Answer only the field kind.', c1.`body`), "
        "['praise', 'complaint']) AS `kind`, "
        "AI.MAP(c1.`body`, 'Label the review: [body] Answer only the field summary.') AS `summary`, "
        "TRY_CAST(AI.MAP(c1.`body`, 'Label the review: [body] Answer only the field stars. "
        "Answer with a number only.') AS DOUBLE) AS `stars` FROM c1)\n"
        "SELECT c2.`_docetl_id` AS `_docetl_id`, c2.`body` AS `body`, c2.`kind` AS `kind`, "
        "c2.`summary` AS `summary`, c2.`stars` AS `stars` FROM c2"
    )


def test_compile_run_reads_an_earlier_output_as_a_column():
    later = {
        "name": "short",
        "type": "filter",
        "prompt": "Is the summary short? {{ input.summary }}",
        "output": {"schema": {"short": "bool"}},
    }
    compiled = compile_run([MAP, later], "docs")
    assert compiled.input_fields == ["body"]
    assert "PROMPT('Is the summary short? {0}', c1.`summary`)" in compiled.sql


def test_compile_run_escapes_quotes_and_passes_max_tokens():
    op = {
        **MAP,
        "prompt": "Summarize the reviewer's text: {{ input.body }}",
        "output": {"schema": {"summary": "string"}},
        "litellm_completion_kwargs": {"max_tokens": 64},
    }
    sql = compile_run([op], "docs").sql
    assert "AI.MAP(c0.`body`, 'Summarize the reviewer\\'s text: [body]', 64)" in sql


def test_convert_turns_cells_into_schema_types():
    assert convert(4.0, OutputColumn("n", "integer")) == 4
    assert convert(None, OutputColumn("n", "integer")) is None
    assert convert('```json\n["a", "b"]\n```', OutputColumn("j", "json")) == ["a", "b"]
    assert convert("not json", OutputColumn("j", "json")) is None


def test_group_quail_runs_fuses_consecutive_operations_and_falls_back():
    from docetl.containers import _group_quail_runs

    ops = {
        "ending": FILTER,
        "label": MAP,
        "gleaned": {**MAP, "name": "gleaned", "gleaning": {"num_rounds": 1}},
        "plain": {**MAP, "name": "plain", "engine": "litellm"},
        "tail": {**MAP, "name": "tail"},
    }
    runner = SimpleNamespace(
        config={"engine": ENGINE},
        find_operation=lambda name: ops[name],
        console=SimpleNamespace(log=lambda *a, **k: None),
    )
    nodes = [SimpleNamespace(name=n) for n in ["ending", "label", "gleaned", "plain", "tail"]]
    groups = _group_quail_runs(runner, nodes)
    shape = [[n.name for n in g[0]] if isinstance(g, tuple) else g.name for g in groups]
    assert shape == [["ending", "label"], "gleaned", "plain", ["tail"]]

    runner.config = {"engine": {**ENGINE, "fallback": "strict"}}
    with pytest.raises(ValueError, match="strict.*gleaning"):
        _group_quail_runs(runner, nodes)


# ── 2. A fake Quail Server ────────────────────────────────────────────


class FakeQuailServer(flight.FlightServerBase):
    """Answers like Quail Server: keeps even rows, labels each kept row."""

    def __init__(self):
        super().__init__("grpc://127.0.0.1:0")
        self.tables: dict[str, pa.Table] = {}
        self.queries: list[dict] = []
        self.explained: list[str] = []
        self.sql_by_id: dict[str, str] = {}

    def do_put(self, context, descriptor, reader, writer):
        name = descriptor.path[0].decode()
        if name in self.tables:  # as Quail Server refuses a second upload
            raise flight.FlightServerError(f"Execution error: The table {name} already exists")
        self.tables[name] = reader.read_all()

    def do_action(self, context, action):
        body = json.loads(action.body.to_pybytes())
        if action.type == "explain":
            self.explained.append(body["sql"])
            return [flight.Result(b"plan")]
        if action.type == "submit":
            self.queries.append(body)
            self.sql_by_id[body.get("id", "q")] = body["sql"]
            status = {"id": body.get("id", "q"), "state": "succeeded", "revision": 1}
        else:
            status = {"id": body["id"], "state": "succeeded", "revision": 1}
        return [flight.Result(json.dumps(status).encode())]

    def do_get(self, context, ticket):
        query_id = json.loads(ticket.ticket.decode())["query_id"]
        name = re.search(r"FROM `([^`]+)`", self.sql_by_id[query_id]).group(1)
        table = self.tables[name]
        kept = [i for i in table.column("_docetl_id").to_pylist() if int(i) % 2 == 0]
        return flight.RecordBatchStream(
            pa.table(
                {
                    "c2._docetl_id": kept,
                    "c2.body": ["ignored"] * len(kept),
                    "kind": ["praise"] * len(kept),
                    "summary": [f"summary {int(i)}" for i in kept],
                    "stars": [4.0] * len(kept),
                }
            )
        )


@pytest.fixture
def fake_server():
    server = FakeQuailServer()
    thread = threading.Thread(target=server.serve, daemon=True)
    thread.start()
    yield server, f"grpc://127.0.0.1:{server.port}"
    server.shutdown()


def _pipeline(tmp_path, endpoint, ops, rows):
    source = tmp_path / "in.json"
    source.write_text(json.dumps(rows))
    return {
        "default_model": "gpt-4o-mini",
        "engine": {"type": "quail", "endpoint": endpoint},
        "datasets": {"reviews": {"type": "file", "path": str(source)}},
        "operations": ops,
        "pipeline": {
            "steps": [{"name": "s", "input": "reviews", "operations": [op["name"] for op in ops]}],
            "output": {"type": "file", "path": str(tmp_path / "out.json")},
        },
    }


def test_runner_sends_one_fused_query_and_maps_rows_back(tmp_path, fake_server):
    from docetl.runner import DSLRunner

    server, endpoint = fake_server
    rows = [{"id": i, "body": f"review {i}", "extra": {"k": i}} for i in range(4)]
    config = _pipeline(tmp_path, endpoint, [FILTER, MAP], rows)
    DSLRunner(config, max_threads=2).load_run_save()

    output = json.loads((tmp_path / "out.json").read_text())
    assert output == [
        {"id": 0, "body": "review 0", "extra": {"k": 0}, "kind": "praise", "summary": "summary 0", "stars": 4},
        {"id": 2, "body": "review 2", "extra": {"k": 2}, "kind": "praise", "summary": "summary 2", "stars": 4},
    ]
    (query,) = server.queries
    assert query["dialect"] == "BigQuery"
    assert query["id"].startswith("docetl-")
    (table,) = [t for name, t in server.tables.items() if not name.startswith("docetl_check_")]
    assert table.column_names == ["_docetl_id", "body"]
    # syntax_check planned the query against an empty table first, once
    assert len(server.explained) == 1

    # A rerun uploads the same table name again, which the server refuses;
    # the client treats that as done and gets the same rows.
    DSLRunner(config, max_threads=2).load_run_save()
    assert json.loads((tmp_path / "out.json").read_text()) == output
    assert len(server.queries) == 2 and server.queries[0]["id"] == server.queries[1]["id"]


# ── 3. A real quail-server with Quail's oracle worker ────────────────

QUAIL_REPO = os.environ.get("QUAIL_REPO")


@pytest.fixture
def quail_server(tmp_path):
    repo = Path(QUAIL_REPO)
    with socket.socket() as found:
        found.bind(("127.0.0.1", 0))
        port = found.getsockname()[1]
    python = repo / ".venv" / "bin" / "python"
    process = subprocess.Popen(
        [str(repo / ".venv" / "bin" / "quail-server"), "--model", "qwen3-4b-fp8",
         "--device", "h100-sxm", "--listen", f"127.0.0.1:{port}", "--data", str(tmp_path / "data")],
        env={**os.environ, "QUAIL_WORKER": f"{python} {repo / 'tests/helpers/oracle_worker.py'}"},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    endpoint = f"grpc://127.0.0.1:{port}"
    client = flight.FlightClient(endpoint)
    deadline = time.monotonic() + 60
    while True:
        try:
            list(client.list_actions())
            break
        except flight.FlightUnavailableError:
            assert process.poll() is None and time.monotonic() < deadline, "quail-server did not start"
            time.sleep(0.1)
    yield endpoint
    process.terminate()
    process.wait(timeout=10)


@pytest.mark.skipif(not QUAIL_REPO, reason="set QUAIL_REPO to a built Quail checkout")
def test_real_quail_server_runs_a_fused_filter_and_map(tmp_path, quail_server):
    from docetl.runner import DSLRunner

    rows = [{"id": i, "body": f"review {i}"} for i in range(4)]
    later = {
        "name": "short",
        "type": "filter",
        "prompt": "Is the reviewer's summary short? {{ input.summary }}",
        "output": {"schema": {"short": "bool"}},
    }
    config = _pipeline(tmp_path, quail_server, [FILTER, MAP, later], rows)
    DSLRunner(config, max_threads=2).load_run_save()

    output = json.loads((tmp_path / "out.json").read_text())
    # The oracle keeps the documents at even positions of each AI stage, so
    # the two filters keep fewer rows than came in.
    assert 0 < len(output) < len(rows)
    for row in output:
        assert set(row) == {"id", "body", "kind", "summary", "stars"}
        assert row["body"] == f"review {row['id']}"
        assert row["kind"] in ("praise", "complaint")

    # A rerun uploads the same table name again; the server's "already
    # exists" refusal counts as done, and the saved rows come back.
    DSLRunner(config, max_threads=2).load_run_save()
    assert json.loads((tmp_path / "out.json").read_text()) == output


@pytest.mark.skipif(not QUAIL_REPO, reason="set QUAIL_REPO to a built Quail checkout")
def test_real_quail_server_runs_a_fused_filter_map_and_reduce(tmp_path, quail_server):
    from docetl.runner import DSLRunner

    rows = [{"id": i, "movie": i % 2, "body": f"review {i}"} for i in range(8)]
    tag = {
        "name": "tag",
        "type": "map",
        "prompt": "Tag the review: {{ input.body }}",
        "output": {"schema": {"tag": "string"}},
    }
    reduce = {**REDUCE, "prompt": "Summarize.{% for r in inputs %}\n- {{ r.body }} ({{ r.tag }}){% endfor %}"}
    config = _pipeline(tmp_path, quail_server, [tag, reduce], rows)
    DSLRunner(config, max_threads=2).load_run_save()

    output = json.loads((tmp_path / "out.json").read_text())
    # One row per movie, with the integer key restored.
    assert sorted(row["movie"] for row in output) == [0, 1]
    for row in output:
        assert set(row) == {"movie", "verdict", "summary"}
        assert row["verdict"] in ("good", "bad")
        assert isinstance(row["summary"], str)


def _quail_op(engine, ops):
    from docetl.operations.quail import QuailOperation

    runner = SimpleNamespace(
        config={}, total_cost=0.0, _run_operation=lambda op, data: [{"ran": op["name"], "n": len(data)}]
    )
    return QuailOperation(
        runner,
        {"name": "q", "type": "quail", "engine": engine, "ops": ops},
        "gpt-4o-mini",
        1,
        console=SimpleNamespace(log=lambda *a, **k: None),
    )


def test_oversized_reduce_groups_fall_back_to_litellm_or_fail_strictly():
    engine = {**ENGINE, "fallback": "litellm", "max_group_tokens": 10}
    rows = [{"movie": 1, "body": "x" * 100}, {"movie": 2, "body": "short"}]
    op = _quail_op(engine, [REDUCE])
    assert op._oversized_group(rows, (["movie"], ["body"])) == 25
    assert op.execute(rows) == ([{"ran": "per_movie", "n": 2}], 0.0)

    strict = _quail_op({**engine, "fallback": "strict"}, [REDUCE])
    with pytest.raises(ValueError, match="max_group_tokens"):
        strict.execute(rows)


def test_quail_form_directive_retries_until_the_parser_accepts(monkeypatch):
    import docetl.reasoning_optimizer.directives.quail_form as quail_form

    answers = iter(
        [
            "Summarize {{ input.body | trim }}",  # still has a filter
            "Summarize the review {{ input.body }}.",
        ]
    )
    seen = []

    def fake_completion(model, messages, response_format):
        seen.append(messages[-1]["content"])
        content = json.dumps({"quail_prompt": next(answers)})
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            _hidden_params={"response_cost": 0.001},
        )

    monkeypatch.setattr(quail_form, "completion", fake_completion)
    op = {**MAP, "prompt": "{% if input.x %}Hi {% endif %}Summarize {{ input.body | trim }}"}
    directive = quail_form.QuailFormDirective()
    ops, _, cost = directive.instantiate([op], ["label"], "some-model")
    assert ops[0]["prompt"] == "Summarize the review {{ input.body }}."
    assert cost == pytest.approx(0.002)
    # the retry carried the parser's error back to the model
    assert "a filter" in seen[1]


def test_fused_run_checkpoints_under_its_last_operation(tmp_path, fake_server):
    from docetl.runner import DSLRunner

    server, endpoint = fake_server
    rows = [{"id": i, "body": f"review {i}"} for i in range(4)]
    config = _pipeline(tmp_path, endpoint, [FILTER, MAP], rows)
    config["pipeline"]["output"]["intermediate_dir"] = str(tmp_path / "intermediates")
    DSLRunner(config, max_threads=2).load_run_save()
    first = json.loads((tmp_path / "out.json").read_text())

    # The run saves its rows under its last operation's name and step hash.
    saved = json.loads((tmp_path / "intermediates" / "s" / "label.json").read_text())
    assert saved == first
    assert len(server.queries) == 1

    # A rerun loads the checkpoint and sends no query.
    DSLRunner(config, max_threads=2).load_run_save()
    assert json.loads((tmp_path / "out.json").read_text()) == first
    assert len(server.queries) == 1

    # Changing an operation in the run changes the hash, so the run reruns.
    changed = {**MAP, "prompt": "Label this review: {{ input.body }}"}
    config["operations"] = [FILTER, changed]
    DSLRunner(config, max_threads=2).load_run_save()
    assert len(server.queries) == 2


def test_bypass_cache_asks_quail_server_for_a_fresh_run(tmp_path, fake_server):
    from docetl.runner import DSLRunner

    server, endpoint = fake_server
    rows = [{"id": i, "body": f"review {i}"} for i in range(4)]
    config = {**_pipeline(tmp_path, endpoint, [FILTER, MAP], rows), "bypass_cache": True}
    DSLRunner(config, max_threads=2).load_run_save()
    DSLRunner(config, max_threads=2).load_run_save()
    ids = [query["id"] for query in server.queries]
    assert len(ids) == 2 and ids[0] != ids[1]
    assert all(i.startswith(ids[0][: len("docetl-") + 24]) for i in ids)

    # An operation-level bypass_cache does the same.
    server.queries.clear()
    config = _pipeline(tmp_path, endpoint, [FILTER, {**MAP, "bypass_cache": True}], rows)
    DSLRunner(config, max_threads=2).load_run_save()
    DSLRunner(config, max_threads=2).load_run_save()
    assert server.queries[0]["id"] != server.queries[1]["id"]


def test_quail_model_names_route_operations_and_price_gpu_time(tmp_path, fake_server):
    from docetl.runner import DSLRunner

    server, endpoint = fake_server
    rows = [{"id": i, "body": f"review {i}"} for i in range(4)]
    config = _pipeline(tmp_path, endpoint, [FILTER, MAP], rows)
    del config["engine"]
    config["default_model"] = "quail/qwen3-4b-fp8"
    # A huge price, so the fake server's milliseconds show up as cost.
    config["quail_models"] = {"qwen3-4b-fp8": {"endpoint": endpoint, "gpu_cost_per_hour": 3.6e6}}
    cost = DSLRunner(config, max_threads=2).load_run_save()
    assert len(server.queries) == 1
    assert cost > 0
    assert len(json.loads((tmp_path / "out.json").read_text())) == 2


def test_quail_model_names_fail_when_an_operation_has_no_quail_form(tmp_path):
    from docetl.containers import _group_quail_runs

    ops = {"gleaned": {**MAP, "name": "gleaned", "gleaning": {"num_rounds": 1}}, "hosted": {**MAP, "name": "hosted", "model": "gemini/gemini-2.5-flash"}}
    runner = SimpleNamespace(
        config={"default_model": "quail/qwen3-4b-fp8", "quail_models": {"qwen3-4b-fp8": {"endpoint": "grpc://h:1"}}},
        find_operation=lambda name: ops[name],
        console=SimpleNamespace(log=lambda *a, **k: None),
    )
    # An operation with its own hosted model runs on LiteLLM.
    (group,) = _group_quail_runs(runner, [SimpleNamespace(name="hosted")])
    assert group.name == "hosted"
    # A quail/ model cannot run on LiteLLM, so an ineligible operation fails.
    with pytest.raises(ValueError, match="strict.*gleaning"):
        _group_quail_runs(runner, [SimpleNamespace(name="gleaned")])
    with pytest.raises(ValueError, match="needs a Quail Server"):
        engine_for({"model": "quail/unknown"}, runner.config)
