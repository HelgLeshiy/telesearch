"""Build a topic knowledge-graph from indexed chunks (G1+G2).

Pipeline (all on CPU, numpy + scikit-learn):

1. cluster the chunk embeddings into topics (HDBSCAN, falling back to KMeans);
   noise points are attached to their nearest topic so every chunk has a home;
2. label each topic with salient keywords via class-based TF-IDF (c-TF-IDF);
3. lay topics out in 2D from their centroids, blending semantic position with
   date so temporally-close topics sit nearer (UMAP used if installed, else PCA);
4. size each node by how many chunks it holds (more knowledge -> bigger node);
5. connect topics whose centroids are similar (cosine >= threshold), capping
   per-node degree to avoid a hairball.

The result is a JSON-able ``{"nodes", "edges", "meta"}`` dict ready to persist
and render.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass

import numpy as np

_WORD_RE = re.compile(r"[^\W\d_]{3,}", re.UNICODE)

# Small multilingual-ish stopword set (EN + common RU) for keyword cleanup.
_STOPWORDS = {
    "the", "and", "for", "you", "that", "this", "with", "have", "are", "was",
    "but", "not", "they", "his", "her", "she", "him", "from", "what", "out",
    "can", "all", "get", "got", "just", "like", "about", "would", "your",
    "там", "это", "как", "что", "так", "его", "она", "они", "вот", "был",
    "для", "уже", "тебя", "меня", "если", "только", "когда", "если",
    "replying", "msg", "photo", "video", "caption", "transcript",
}


@dataclass
class GraphParams:
    min_cluster_size: int = 3
    max_topics: int = 40
    edge_threshold: float = 0.45  # min centroid cosine for an edge
    max_edges_per_node: int = 4
    date_weight: float = 0.25  # 0 = pure semantics, 1 = pure date in layout
    keywords_per_topic: int = 6
    samples_per_topic: int = 5
    random_state: int = 0

    def hash(self, collections: list[str] | None) -> str:
        payload = json.dumps(
            {"params": asdict(self), "collections": sorted(collections or [])},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _tokenize(text: str) -> list[str]:
    return [w for w in (m.group().lower() for m in _WORD_RE.finditer(text))
            if w not in _STOPWORDS]


def _cluster(x: np.ndarray, params: GraphParams) -> np.ndarray:
    """Return a cluster label per row; noise (-1) reassigned to nearest centroid."""
    from sklearn.cluster import HDBSCAN, KMeans

    n = len(x)
    labels: np.ndarray
    try:
        mcs = max(2, min(params.min_cluster_size, n - 1))
        labels = HDBSCAN(min_cluster_size=mcs, metric="euclidean", copy=True).fit_predict(x)
    except Exception:
        labels = np.full(n, -1)

    distinct = {int(c) for c in labels if c != -1}
    if len(distinct) < 2:
        # Fall back to KMeans with a sqrt-ish topic count.
        k = int(np.clip(round(np.sqrt(max(n, 1) / 2)), 2, min(params.max_topics, n)))
        labels = KMeans(n_clusters=k, n_init=10, random_state=params.random_state).fit_predict(x)
        return labels

    # Reassign noise points to the nearest cluster centroid.
    centroids = {c: x[labels == c].mean(axis=0) for c in distinct}
    cids = list(centroids)
    cmat = np.vstack([centroids[c] for c in cids])
    noise_idx = np.where(labels == -1)[0]
    if len(noise_idx):
        sims = x[noise_idx] @ cmat.T
        nearest = sims.argmax(axis=1)
        for i, j in zip(noise_idx, nearest):
            labels[i] = cids[j]
    return labels


def _keywords(contents: list[str], cluster_ids: list[int], labels: np.ndarray,
              top_n: int) -> dict[int, list[str]]:
    """Class-based TF-IDF: one pseudo-document per cluster, top terms each."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    docs = {c: [] for c in cluster_ids}
    for content, lab in zip(contents, labels):
        docs[int(lab)].append(content)
    cluster_docs = [" ".join(_tokenize(" ".join(docs[c]))) for c in cluster_ids]
    if not any(cluster_docs):
        return {c: [] for c in cluster_ids}
    try:
        vec = TfidfVectorizer(max_features=4000, token_pattern=r"\S+")
        tfidf = vec.fit_transform(cluster_docs)
        terms = np.array(vec.get_feature_names_out())
    except ValueError:
        return {c: [] for c in cluster_ids}
    out: dict[int, list[str]] = {}
    for row, c in enumerate(cluster_ids):
        arr = tfidf[row].toarray().ravel()
        if arr.any():
            top = arr.argsort()[::-1][:top_n]
            out[c] = [terms[i] for i in top if arr[i] > 0]
        else:
            out[c] = []
    return out


def _layout(centroids: np.ndarray, mean_ts: np.ndarray, params: GraphParams) -> np.ndarray:
    """2D coordinates per topic, blending semantics with date proximity."""
    n = len(centroids)
    if n == 1:
        return np.array([[0.0, 0.0]])

    # Augment centroids with a scaled date column so close-in-time topics attract.
    feats = centroids.copy()
    if params.date_weight > 0 and np.ptp(mean_ts) > 0:
        ts = (mean_ts - mean_ts.min()) / np.ptp(mean_ts)
        scale = params.date_weight * float(np.linalg.norm(feats, axis=1).mean())
        feats = np.hstack([feats * (1 - params.date_weight), (ts * scale)[:, None]])

    coords = None
    try:  # UMAP if available (preferred), else PCA.
        import umap  # type: ignore

        coords = umap.UMAP(
            n_components=2, n_neighbors=min(15, n - 1), random_state=params.random_state
        ).fit_transform(feats)
    except Exception:
        from sklearn.decomposition import PCA

        coords = PCA(n_components=2, random_state=params.random_state).fit_transform(feats)

    coords = np.asarray(coords, dtype=float)
    # Normalize to [-1, 1] for a stable client-side viewport.
    span = np.ptp(coords, axis=0)
    span[span == 0] = 1.0
    coords = 2 * (coords - coords.min(axis=0)) / span - 1
    return coords


def build_graph(
    rows: list[dict],
    *,
    params: GraphParams | None = None,
    collections: list[str] | None = None,
) -> dict:
    """Build a topic graph from chunk ``rows`` (each with vector/content/...)."""
    params = params or GraphParams()
    items = [r for r in rows if r.get("vector") and (r.get("content") or "").strip()]
    n = len(items)
    meta = {
        "n_chunks": n,
        "n_topics": 0,
        "params_hash": params.hash(collections),
        "collections": collections or [],
    }
    if n == 0:
        return {"nodes": [], "edges": [], "meta": meta}

    x = np.array([r["vector"] for r in items], dtype="float32")
    # Ensure L2-normalized for cosine via dot product.
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    x = x / norms
    contents = [r.get("content", "") for r in items]
    timestamps = np.array([float(r.get("timestamp") or 0) for r in items])

    if n < max(3, params.min_cluster_size):
        labels = np.zeros(n, dtype=int)  # single topic when too small to cluster
    else:
        labels = _cluster(x, params)

    cluster_ids = sorted(int(c) for c in set(labels.tolist()))
    # If clustering produced too many topics, merge smallest into nearest later;
    # for now cap by keeping the largest ``max_topics`` clusters.
    if len(cluster_ids) > params.max_topics:
        sizes = {c: int((labels == c).sum()) for c in cluster_ids}
        keep = set(sorted(cluster_ids, key=lambda c: sizes[c], reverse=True)[: params.max_topics])
        centroids_keep = {c: x[labels == c].mean(axis=0) for c in keep}
        kc = list(centroids_keep)
        kmat = np.vstack([centroids_keep[c] for c in kc])
        for i, lab in enumerate(labels):
            if int(lab) not in keep:
                labels[i] = kc[int((x[i] @ kmat.T).argmax())]
        cluster_ids = sorted(keep)

    keywords = _keywords(contents, cluster_ids, labels, params.keywords_per_topic)

    centroids = np.vstack([x[labels == c].mean(axis=0) for c in cluster_ids])
    cnorm = centroids / np.clip(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-9, None)
    mean_ts = np.array([timestamps[labels == c].mean() for c in cluster_ids])
    coords = _layout(cnorm, mean_ts, params)

    nodes = []
    index_by_cluster = {c: np.where(labels == c)[0] for c in cluster_ids}
    for pos, c in enumerate(cluster_ids):
        idx = index_by_cluster[c]
        # Representative chunks = closest to the centroid.
        sims = x[idx] @ cnorm[pos]
        order = idx[np.argsort(sims)[::-1]]
        sample = order[: params.samples_per_topic]
        kw = keywords.get(c, [])
        ts_vals = timestamps[idx][timestamps[idx] > 0]
        label = ", ".join(kw[:3]) if kw else f"topic {pos + 1}"
        nodes.append({
            "id": pos,
            "label": label,
            "keywords": kw,
            "size": int(len(idx)),
            "x": float(coords[pos][0]),
            "y": float(coords[pos][1]),
            "mean_ts": float(mean_ts[pos]),
            "date_from": int(ts_vals.min()) if len(ts_vals) else 0,
            "date_to": int(ts_vals.max()) if len(ts_vals) else 0,
            "sample_chunk_ids": [items[i].get("chunk_id", "") for i in sample],
            "sample_contents": [items[i].get("content", "")[:280] for i in sample],
        })

    # Edges by centroid cosine, capped per node to avoid a hairball.
    edges = []
    if len(cluster_ids) >= 2:
        sim = cnorm @ cnorm.T
        ts_norm = (mean_ts - mean_ts.min()) / (np.ptp(mean_ts) or 1.0)
        kept: set[tuple[int, int]] = set()
        for i in range(len(cluster_ids)):
            order = np.argsort(sim[i])[::-1]
            deg = 0
            for j in order:
                if j == i or sim[i][j] < params.edge_threshold:
                    continue
                a, b = (i, int(j)) if i < j else (int(j), i)
                if (a, b) in kept:
                    continue
                date_prox = 1.0 - abs(ts_norm[i] - ts_norm[j])
                weight = float(0.8 * sim[i][j] + 0.2 * date_prox)
                edges.append({"source": a, "target": b, "weight": round(weight, 4)})
                kept.add((a, b))
                deg += 1
                if deg >= params.max_edges_per_node:
                    break

    meta["n_topics"] = len(nodes)
    return {"nodes": nodes, "edges": edges, "meta": meta}
