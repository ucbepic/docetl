from __future__ import annotations

import json
import os
import threading
from functools import lru_cache
from typing import Any

from jinja2 import Template

from .base import RetrievalResult, Retriever


class LanceDBRetriever(Retriever):
    """OSS LanceDB-backed retriever supporting FTS, vector, and hybrid search."""

    _index_lock = threading.Lock()
    _ensured = False

    def _connect(self):
        # Lazy import to avoid hard dependency during linting/tests without install
        try:
            import lancedb  # type: ignore
        except Exception as e:
            raise ImportError(
                "lancedb is required for LanceDBRetriever. Please add it to dependencies."
            ) from e
        index_dir = self.config["index_dir"]
        os.makedirs(index_dir, exist_ok=True)
        return lancedb.connect(index_dir)

    def _table_name(self) -> str:
        # Stable table name based on retriever name
        return f"docetl_{self.name}"

    def _iter_dataset_rows(self) -> list[dict]:
        """Load dataset referenced by this retriever."""
        dataset_name = self.config["dataset"]
        # Ensure datasets are loaded
        if not getattr(self.runner, "datasets", None):
            self.runner.load()
        if dataset_name not in self.runner.datasets:
            raise ValueError(
                f"Retriever '{self.name}' references unknown dataset '{dataset_name}'."
            )
        return self.runner.datasets[dataset_name].load()

    def _render_input_phrase(self, tmpl: str | None, input_obj: dict) -> str:
        if not tmpl:
            return ""
        try:
            return Template(tmpl).render(input=input_obj)
        except Exception:
            return ""

    def _batch_embed(self, texts: list[str]) -> list[list[float]]:
        model = self.config.get("embedding", {}).get("model")
        if not model:
            # If no embedding model, return empty vectors
            return []
        # Use DocETL embedding router + cache
        resp = self.runner.api.gen_embedding(model, json.dumps(texts))
        # litellm embedding response
        vectors = [d["embedding"] for d in resp["data"]]
        return vectors

    def _index_types(self) -> set[str]:
        idx = self.config.get("index_types", [])
        if isinstance(idx, str):
            idx = [idx]
        idx = set(idx or [])
        # Interpret hybrid as both
        if "hybrid" in idx:
            idx.update({"fts", "embedding"})
        return idx

    def ensure_index(self) -> None:
        build_policy = self.config.get("build_index", "if_missing")
        with self._index_lock:
            # Only build once per process; 'always' means rebuild once at startup
            if self._ensured:
                return

            db = self._connect()
            table_name = self._table_name()
            dataset_name = self.config.get("dataset", "<unknown>")
            idx_types = self._index_types()
            try:
                self.runner.console.log(
                    f"[cyan]LanceDB[/cyan] ensure_index: table='{table_name}', dataset='{dataset_name}', "
                    f"index_types={sorted(list(idx_types))}, build_policy='{build_policy}'"
                )
            except Exception:
                pass
            table = None
            try:
                existing_names = db.table_names()
            except Exception as exc:
                # Surface the real error instead of sidestepping
                raise RuntimeError(f"Failed to list LanceDB tables: {exc}") from exc
            if build_policy == "never":
                try:
                    self.runner.console.log(
                        f"[yellow]LanceDB[/yellow] skipping index build (build_index=never) for '{table_name}'"
                    )
                except Exception:
                    pass
                self._ensured = True
                return
            if table_name in existing_names:
                if build_policy == "always":
                    try:
                        self.runner.console.log(
                            f"[cyan]LanceDB[/cyan] dropping existing table '{table_name}' (build_index=always)"
                        )
                    except Exception:
                        pass
                    db.drop_table(table_name)
                else:
                    # if_missing and table exists -> nothing to do
                    try:
                        self.runner.console.log(
                            f"[green]LanceDB[/green] index exists for '{table_name}', skipping build (build_index=if_missing)"
                        )
                    except Exception:
                        pass
                    self._ensured = True
                    return

            if table is None:
                try:
                    rows = self._iter_dataset_rows()
                except Exception:
                    # Dataset may not be available yet (e.g., produced by a prior step)
                    try:
                        self.runner.console.log(
                            f"[yellow]LanceDB[/yellow] dataset '{dataset_name}' not available; "
                            f"deferring index build for '{table_name}'"
                        )
                    except Exception:
                        pass
                    return
                idx_types = self._index_types()

                # Build per-row phrases for FTS and Embedding
                fts_index_tmpl = self.config.get("fts", {}).get("index_phrase", None)
                emb_index_tmpl = self.config.get("embedding", {}).get(
                    "index_phrase", None
                )

                fts_texts: list[str] = []
                if "fts" in idx_types:
                    fts_texts = [
                        self._render_input_phrase(fts_index_tmpl, r) for r in rows
                    ]

                emb_texts: list[str] = []
                if "embedding" in idx_types:
                    emb_texts = [
                        self._render_input_phrase(emb_index_tmpl, r) for r in rows
                    ]
                    # If no template, fall back to fts texts
                    if not any(emb_texts) and fts_texts:
                        emb_texts = fts_texts
                # Compute embeddings for embedding index
                vectors: list[list[float]] = []
                if "embedding" in idx_types:
                    # Batch embeddings
                    batch = 128
                    for i in range(0, len(emb_texts), batch):
                        chunk = emb_texts[i : i + batch]
                        if not chunk:
                            continue
                        vectors.extend(self._batch_embed(chunk))

                # Construct records for LanceDB
                data = []
                for idx, r in enumerate(rows):
                    rec = {
                        "id": r.get("id", idx),
                        "text": (
                            fts_texts[idx]
                            if fts_texts
                            else (emb_texts[idx] if emb_texts else "")
                        ),
                    }
                    if vectors and idx < len(vectors):
                        rec["vector"] = vectors[idx]
                    # keep some fields for rendering
                    data.append(rec)

                # Create table
                table = db.create_table(table_name, data=data, mode="overwrite")
                try:
                    self.runner.console.log(
                        f"[green]LanceDB[/green] created table '{table_name}' with {len(data)} rows "
                        f"(fts={'fts' in idx_types}, embedding={'embedding' in idx_types})"
                    )
                except Exception:
                    pass

                # Create FTS index if fts enabled
                if "fts" in idx_types:
                    try:
                        table.create_fts_index("text")
                        try:
                            self.runner.console.log(
                                f"[green]LanceDB[/green] created FTS index on column 'text' for '{table_name}'"
                            )
                        except Exception:
                            pass
                    except Exception:
                        # Index may already exist or backend unavailable; continue
                        try:
                            self.runner.console.log(
                                f"[yellow]LanceDB[/yellow] FTS index creation skipped/failed for '{table_name}'"
                            )
                        except Exception:
                            pass

            self._ensured = True
            try:
                self.runner.console.log(
                    f"[green]LanceDB[/green] index ready for '{table_name}'"
                )
            except Exception:
                pass

    def _render_query_phrase(self, tmpl: str | None, context: dict[str, Any]) -> str:
        if not tmpl:
            return ""
        try:
            return Template(tmpl).render(**context)
        except Exception:
            return ""

    def _select_mode(self) -> str:
        qmode = self.config.get("query", {}).get("mode", None)
        if qmode:
            return qmode
        # Auto: hybrid if both index types present
        idx = self._index_types()
        if "fts" in idx and "embedding" in idx:
            return "hybrid"
        if "fts" in idx:
            return "fts"
        if "embedding" in idx:
            return "embedding"
        return "fts"

    def _reranker(self):
        # Lazy import and instantiate RRFReranker when needed
        try:
            from lancedb.rerankers import RRFReranker  # type: ignore

            rr = RRFReranker()
            return rr
        except Exception:
            return None

    def _limit_and_format(self, rows: list[dict]) -> list[dict]:
        top_k = int(self.config.get("query", {}).get("top_k", 5))
        return rows[:top_k]

    def _fetch(self, context: dict[str, Any]) -> list[dict]:
        db = self._connect()
        table = db.open_table(self._table_name())
        mode = self._select_mode()
        top_k = int(self.config.get("query", {}).get("top_k", 5))

        # Build queries
        text_query = self._render_query_phrase(
            self.config.get("fts", {}).get("query_phrase", None), context
        )
        vector_query: list[float] | None = None
        emb_qtext = self._render_query_phrase(
            self.config.get("embedding", {}).get("query_phrase", None), context
        )
        if emb_qtext:
            vecs = self._batch_embed([emb_qtext])
            vector_query = vecs[0] if vecs else None

        # Execute
        try:
            if mode == "hybrid" and text_query and vector_query is not None:
                q = (
                    table.search(query_type="hybrid")
                    .vector(vector_query)
                    .text(text_query)
                )
                rr = self._reranker()
                if rr:
                    q = q.rerank(rr)
                df = q.limit(top_k).to_pandas()
            elif mode == "fts" and text_query:
                # FTS-only
                q = table.search(text_query, query_type="fts", fts_columns="text")
                df = q.limit(top_k).to_pandas()
            elif mode == "embedding" and vector_query is not None:
                q = table.search(vector_query)
                df = q.limit(top_k).to_pandas()
            else:
                return []
        except Exception:
            # Fallbacks
            try:
                q = (
                    table.search(text_query, query_type="fts", fts_columns="text")
                    if text_query
                    else table.search(vector_query)
                )
                df = q.limit(top_k).to_pandas()
            except Exception:
                return []

        # Convert to list of dicts
        rows = df.to_dict(orient="records") if hasattr(df, "to_dict") else []
        return self._limit_and_format(rows)

    def _render_docs(self, docs: list[dict]) -> str:
        # Minimal, opinionated rendering: bullets of text (truncated)
        max_chars = 1000
        lines: list[str] = []
        for i, d in enumerate(docs, start=1):
            snippet = str(d.get("text", ""))[:max_chars]
            if snippet:
                lines.append(f"- [{i}] {snippet}")
        return "\n".join(lines)

    @staticmethod
    @lru_cache(maxsize=256)
    def _cache_key(name: str, ctx_key: str) -> str:
        return f"{name}:{ctx_key}"

    def _ctx_fingerprint(self, context: dict[str, Any]) -> str:
        # Make a stable string key from context, using only parts used by templates
        if "input" in context:
            base = context["input"]
        elif "reduce_key" in context:
            base = {
                "reduce_key": context.get("reduce_key", {}),
                "inputs_len": len(context.get("inputs", [])),
            }
        else:
            base = context
        try:
            return json.dumps(base, sort_keys=True, ensure_ascii=False)[:2000]
        except Exception:
            return str(base)[:2000]

    def retrieve(self, context: dict[str, Any]) -> RetrievalResult:
        self.ensure_index()
        # Build a fingerprint (reserved for future caching)
        _ = self._ctx_fingerprint(context)
        docs = self._fetch(context)
        rendered = self._render_docs(docs) if docs else "No extra context available."
        meta = {"retriever": self.name, "num_docs": len(docs)}
        return RetrievalResult(docs=docs, rendered_context=rendered, meta=meta)
