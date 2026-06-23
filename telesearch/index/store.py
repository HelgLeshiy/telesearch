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


def _schema(dim: int) -> pa.Schema:
    return pa.schema(
        [
            pa.field("chunk_id", pa.string()),
            pa.field("message_id", pa.int64()),
            pa.field("chat", pa.string()),
            pa.field("sender", pa.string()),
            pa.field("timestamp", pa.int64()),
            pa.field("date_str", pa.string()),
            pa.field("modality", pa.string()),
            pa.field("content", pa.string()),
            pa.field("media_path", pa.string()),
            pa.field("extra", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), dim)),
        ]
    )


class VectorStore:
    def __init__(self, db_path: str | Path, dim: int):
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.dim = dim
        self.db = lancedb.connect(str(self.db_path))
        if TABLE_NAME in self.db.list_tables().tables:
            self.table = self.db.open_table(TABLE_NAME)
        else:
            self.table = self.db.create_table(TABLE_NAME, schema=_schema(dim))

    def add(self, rows: list[dict[str, Any]], vectors: np.ndarray) -> None:
        """Append rows with their (already L2-normalized) vectors."""
        if not rows:
            return
        payload = []
        for row, vec in zip(rows, vectors):
            r = dict(row)
            r["media_path"] = r.get("media_path") or ""
            r["vector"] = vec.astype(np.float32).tolist()
            payload.append(r)
        self.table.add(payload)

    def build_fts(self) -> None:
        """(Re)build the BM25 full-text index over the content column."""
        self.table.create_fts_index("content", replace=True, use_tantivy=False)

    def count(self) -> int:
        return self.table.count_rows()

    def existing_message_ids(self) -> set[int]:
        """Return the set of message ids already present (for resumable builds)."""
        if self.count() == 0:
            return set()
        try:
            ids = self.table.to_arrow().column("message_id").to_pylist()
            return set(ids)
        except Exception:
            return set()

    def drop(self) -> None:
        """Delete all indexed data (for a full rebuild)."""
        self.table = self.db.create_table(
            TABLE_NAME, schema=_schema(self.dim), mode="overwrite"
        )

    def _vector_search(self, query_vec: np.ndarray, k: int) -> list[dict]:
        return (
            self.table.search(query_vec.astype(np.float32))
            .limit(k)
            .to_list()
        )

    def _fts_search(self, query_text: str, k: int) -> list[dict]:
        try:
            return (
                self.table.search(query_text, query_type="fts")
                .limit(k)
                .to_list()
            )
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
    ) -> list[dict]:
        """Reciprocal Rank Fusion of vector and full-text results."""
        vec_hits = self._vector_search(query_vec, candidates)
        fts_hits = self._fts_search(query_text, candidates)

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
