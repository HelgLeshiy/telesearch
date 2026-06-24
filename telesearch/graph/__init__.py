"""Knowledge-graph construction over indexed embeddings (Phase 3, G1+G2).

Turns a workspace's chunks into a navigable topic graph: cluster the existing
embeddings into topics, label each topic with its salient keywords (c-TF-IDF),
lay topics out in 2D, and connect related topics by centroid similarity and
date proximity. Built entirely on numpy + scikit-learn (already required by the
embedding stack); UMAP and an LLM are optional enhancements.
"""

from .build import GraphParams, build_graph

__all__ = ["build_graph", "GraphParams"]
