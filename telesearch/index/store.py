"""LanceDB-backed vector + full-text store with hybrid retrieval.

LanceDB is an embedded, file-based vector database (no server to run). We store
each chunk's embedding alongside its text and metadata, build a BM25 full-text
index on the text, and fuse vector and keyword results with Reciprocal Rank
Fusion so that both semantic and exact-keyword matches surface.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import lancedb
import numpy as np
import pyarrow as pa

TABLE_NAME = "chunks"


# String columns carried on every chunk. The trailing block (collection_id …
# lang) was added for multi-tenant / multi-source indexing and search-time
# filtering; older indexes built before this simply don't have them and are read
# back transparently (missing values default to "").
_STR_FIELDS = (
    "chunk_id",
    "chat",
    "sender",
    "date_str",
    "modality",
    "content",
    "media_path",
    "extra",
    "collection_id",
    "source_kind",
    "doc_id",
    "lang",
)


def _schema(dim: int) -> pa.Schema:
    fields = [pa.field("message_id", pa.int64()), pa.field("timestamp", pa.int64())]
    fields += [pa.field(name, pa.string()) for name in _STR_FIELDS]
    fields.append(pa.field("vector", pa.list_(pa.float32(), dim)))
    return pa.schema(fields)


def _table_names(db) -> list[str]:
    """Return existing table names as a plain ``list[str]`` across LanceDB versions.

    Newer LanceDB returns a ``ListTablesResponse`` from ``list_tables()`` whose
    membership test never matches (iterating it yields ``(key, value)`` tuples),
    so we read its ``.tables`` attribute. We prefer ``list_tables()`` because
    ``table_names()`` is deprecated, falling back to it on very old versions.
    """
    resp = db.list_tables()
    tables = getattr(resp, "tables", None)
    if tables is not None:
        return list(tables)
    if isinstance(resp, (list, tuple)):
        return list(resp)
    try:  # pragma: no cover - very old LanceDB
        return list(db.table_names())
    except AttributeError:  # pragma: no cover
        return list(resp)


class VectorStore:
    def __init__(self, db_path: str | Path, dim: int):
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.dim = dim
        self.db = lancedb.connect(str(self.db_path))
        if TABLE_NAME in _table_names(self.db):
            self.table = self.db.open_table(TABLE_NAME)
        else:
            self.table = self.db.create_table(TABLE_NAME, schema=_schema(dim))

    def _coerce_rows(self, rows: list[dict[str, Any]], vectors: np.ndarray) -> list[dict]:
        """Align rows to the table's actual schema before inserting.

        This keeps inserts working against both fresh indexes (full schema) and
        pre-existing indexes created before the multi-tenant columns were added:
        unknown columns are dropped and missing string columns are filled with
        ``""`` so the payload always matches the table on disk.
        """
        cols = set(self.table.schema.names)
        payload = []
        for row, vec in zip(rows, vectors):
            r = {k: v for k, v in row.items() if k in cols}
            for name in _STR_FIELDS:
                if name in cols:
                    r[name] = r.get(name) or ""
            r["vector"] = vec.astype(np.float32).tolist()
            payload.append(r)
        return payload

    def add(self, rows: list[dict[str, Any]], vectors: np.ndarray) -> None:
        """Append rows with their (already L2-normalized) vectors."""
        if not rows:
            return
        self.table.add(self._coerce_rows(rows, vectors))

    def build_fts(self) -> None:
        """(Re)build the BM25 full-text index over the content column."""
        self.table.create_fts_index("content", replace=True, use_tantivy=False)

    def count(self) -> int:
        return self.table.count_rows()

    def existing_message_ids(self, collection_id: str | None = None) -> set[int]:
        """Return message ids already present (for resumable builds).

        When ``collection_id`` is given, resume is scoped to that collection so
        multiple sources sharing one workspace (and thus overlapping message-id
        spaces) don't shadow each other. On indexes created before the
        ``collection_id`` column existed, the filter is ignored (all ids).
        """
        if self.count() == 0:
            return set()
        try:
            tbl = self.table.to_arrow()
            if collection_id is not None and "collection_id" in tbl.schema.names:
                import pyarrow.compute as pc

                tbl = tbl.filter(pc.equal(tbl.column("collection_id"), collection_id))
            return set(tbl.column("message_id").to_pylist())
        except Exception:
            return set()

    def drop(self) -> None:
        """Delete all indexed data (for a full rebuild)."""
        self.table = self.db.create_table(
            TABLE_NAME, schema=_schema(self.dim), mode="overwrite"
        )

    def delete_modalities(
        self,
        modalities: list[str] | tuple[str, ...],
        collection_id: str | None = None,
    ) -> int:
        """Delete all chunks of the given modalities; return rows removed.

        Used by the text-only reindex to refresh just the text-derived chunks
        (``text`` + ``conversation``) without touching the expensive media
        chunks (captions, transcripts, OCR, documents). When ``collection_id``
        is given the delete is scoped to that collection so refreshing one
        source doesn't wipe sibling sources in the same workspace.
        """
        if not modalities:
            return 0
        before = self.count()
        mods = ", ".join(f"'{m}'" for m in modalities)
        where = f"modality IN ({mods})"
        if collection_id is not None:
            safe = collection_id.replace("'", "''")
            where += f" AND collection_id = '{safe}'"
        self.table.delete(where)
        return before - self.count()

    def _apply_where(self, search, where: str | None):
        """Attach a (pre)filter to a LanceDB search across versions."""
        if not where:
            return search
        try:
            return search.where(where, prefilter=True)
        except TypeError:  # pragma: no cover - older LanceDB without prefilter kw
            return search.where(where)

    def _vector_search(self, query_vec: np.ndarray, k: int, where: str | None = None) -> list[dict]:
        search = self.table.search(query_vec.astype(np.float32))
        return self._apply_where(search, where).limit(k).to_list()

    def _fts_search(self, query_text: str, k: int, where: str | None = None) -> list[dict]:
        try:
            search = self.table.search(query_text, query_type="fts")
            return self._apply_where(search, where).limit(k).to_list()
        except Exception:
            # FTS index may not exist yet.
            return []

    def fetch_around(
        self,
        chat: str,
        message_ids: list[int],
        before: int,
        after: int,
        modalities: tuple[str, ...] | None = None,
        limit: int = 2000,
    ) -> list[dict]:
        """Return chunks in the message-id neighbourhood of ``message_ids``.

        Used for retrieval-time *context expansion*: given the message ids of
        the best matches, pull the surrounding messages (same chat) so an answer
        can be grounded in the conversation around a hit, not just the single
        matched line. Telegram message ids are (near-)sequential in time, so an
        id range is a good proxy for "the messages just before/after this one".
        Results are returned in chronological order with the vector dropped.
        """
        if not message_ids or (before <= 0 and after <= 0):
            return []
        safe_chat = chat.replace("'", "''")
        ranges = " OR ".join(
            f"(message_id >= {mid - before} AND message_id <= {mid + after})"
            for mid in message_ids
        )
        where = f"chat = '{safe_chat}' AND ({ranges})"
        if modalities:
            mods = ", ".join(f"'{m}'" for m in modalities)
            where += f" AND modality IN ({mods})"
        try:
            rows = self.table.search().where(where).limit(limit).to_list()
        except Exception:
            return []
        for r in rows:
            r.pop("vector", None)
        rows.sort(
            key=lambda r: (
                r.get("timestamp", 0),
                r.get("message_id", 0),
                r.get("chunk_id", ""),
            )
        )
        return rows

    def hybrid_search(
        self,
        query_text: str,
        query_vec: np.ndarray,
        k: int = 10,
        candidates: int = 50,
        rrf_k: int = 60,
        where: str | None = None,
    ) -> list[dict]:
        """Reciprocal Rank Fusion of vector and full-text results.

        ``where`` is an optional LanceDB filter expression applied *before*
        retrieval (prefilter), so structured constraints — date range, modality,
        source, collection, sender — narrow the candidate pool instead of being
        applied to an already-truncated top-k.
        """
        vec_hits = self._vector_search(query_vec, candidates, where)
        fts_hits = self._fts_search(query_text, candidates, where)

        scores: dict[str, float] = {}
        records: dict[str, dict] = {}

        for rank, hit in enumerate(vec_hits):
            cid = hit["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank + 1)
            records[cid] = hit
        for rank, hit in enumerate(fts_hits):
            cid = hit["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank + 1)
            records.setdefault(cid, hit)

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        results = []
        for cid, score in ranked[:k]:
            rec = dict(records[cid])
            rec["score"] = score
            rec.pop("vector", None)
            results.append(rec)
        return results
