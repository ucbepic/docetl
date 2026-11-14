"""A Quail Server client over Arrow Flight, with only ``pyarrow``.

Quail Server (``quail-server``) takes five Flight calls: ``do_put`` uploads a
table, the ``submit`` / ``status`` / ``explain`` actions queue, watch, and plan
a query, and ``do_get`` reads a finished query's rows. Talking to the server
directly keeps Quail's Python package and its pins out of DocETL.
"""

from __future__ import annotations

import json
import os
import time

import pyarrow as pa
import pyarrow.flight as flight

TOKEN_VARIABLE = "QUAIL_SERVER_TOKEN"
DIALECT = "BigQuery"
FINISHED = frozenset({"succeeded", "failed", "cancelled"})


class QuailQueryError(RuntimeError):
    """A query that Quail Server failed or cancelled."""


class QuailClient:
    """Flight calls to one Quail Server.

    Args:
        endpoint: The server's address, such as ``grpc://host:50051``.
        token: The server's token; ``QUAIL_SERVER_TOKEN`` when omitted.
    """

    def __init__(self, endpoint: str, token: str | None = None):
        self.endpoint = endpoint
        kwargs = {}
        if endpoint.startswith("grpc+tls://"):
            # A TLS endpoint behind a proxy that re-signs TLS needs the
            # proxy's CA; QUAIL_TLS_ROOT_CERTS, then SSL_CERT_FILE, name it.
            roots = os.environ.get("QUAIL_TLS_ROOT_CERTS") or os.environ.get("SSL_CERT_FILE")
            if roots and os.path.exists(roots):
                with open(roots, "rb") as f:
                    kwargs["tls_root_certs"] = f.read()
        self._flight = flight.FlightClient(endpoint, **kwargs)
        token = token if token is not None else os.environ.get(TOKEN_VARIABLE)
        self._options = flight.FlightCallOptions(
            headers=[(b"authorization", f"Bearer {token}".encode())] if token else []
        )

    def close(self) -> None:
        self._flight.close()

    def __enter__(self) -> "QuailClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _act(self, name: str, body) -> bytes:
        text = body if isinstance(body, str) else json.dumps(body)
        (answer,) = self._flight.do_action(
            flight.Action(name, text.encode()), self._options
        )
        return answer.body.to_pybytes()

    def upload(self, name: str, table: pa.Table, id_col: str) -> None:
        """Register ``table`` on the server as ``name``, keyed by ``id_col``."""
        descriptor = flight.FlightDescriptor.for_path(name, id_col)
        writer, _ = self._flight.do_put(descriptor, table.schema, self._options)
        try:
            writer.write_table(table)
        finally:
            writer.close()

    def submit(self, sql: str, query_id: str | None = None) -> dict:
        """Queue a query. The server runs one id once and keeps its rows."""
        body = {"sql": sql, "dialect": DIALECT}
        if query_id:
            body["id"] = query_id
        return json.loads(self._act("submit", body))

    def wait(self, status: dict, timeout_s: float | None = None) -> dict:
        """Wait until a submitted query finishes; return its final status."""
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        while status["state"] not in FINISHED:
            if deadline is not None and time.monotonic() > deadline:
                raise TimeoutError(f"Quail query {status['id']} did not finish")
            status = json.loads(
                self._act(
                    "status",
                    {"id": status["id"], "after": status.get("revision", 0), "wait_s": 30},
                )
            )
        return status

    def result(self, query_id: str) -> pa.Table:
        """Return the saved rows of a query that succeeded."""
        ticket = flight.Ticket(json.dumps({"query_id": query_id}).encode())
        return self._flight.do_get(ticket, self._options).read_all()

    def explain(self, sql: str) -> str:
        """Return the server's plan of a query, without running it."""
        return self._act("explain", {"sql": sql, "dialect": DIALECT}).decode()

    def run(self, sql: str, query_id: str | None = None) -> pa.Table:
        """Submit a query, wait for it, and return its rows."""
        status = self.wait(self.submit(sql, query_id))
        if status["state"] != "succeeded":
            raise QuailQueryError(
                f"Quail query {status['id']} {status['state']}: {status.get('error')}"
            )
        return self.result(status["id"])
