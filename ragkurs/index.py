"""Qdrant-Index (lokaler Modus, kein Server noetig) mit Dense-, Sparse- und Hybrid-Suche.

Hybrid = Dense + Sparse parallel (prefetch) -> Fusion (RRF oder DBSF) -> Top-k.
Filter (Rollen, Status, Abteilung) werden VOR der Vektorsuche angewendet (Pre-Filtering).
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Literal

from qdrant_client import QdrantClient, models

from .chunking import Chunk
from .config import settings
from .llm import embedding_dim, get_embedder
from .sparse import SimpleBM25Encoder

Fusion = Literal["rrf", "dbsf"]


@dataclass
class Hit:
    chunk: Chunk
    score: float
    rank: int
    source: str = "dense"  # dense | sparse | hybrid | rerank
    extra: dict = field(default_factory=dict)

    @property
    def doc_id(self) -> str:
        return self.chunk.doc_id

    def __repr__(self) -> str:
        return f"Hit(#{self.rank} {self.chunk.chunk_id} score={self.score:.3f} via {self.source})"


def _is_local(client: QdrantClient) -> bool:
    return type(getattr(client, "_client", None)).__name__ == "QdrantLocal"


def _point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def build_filter(
    user_roles: list[str] | None = None,
    status: str | None = "current",
    department: str | None = None,
    doc_ids: list[str] | None = None,
) -> models.Filter | None:
    """Payload-Filter. `user_roles` -> Chunk muss fuer eine der Rollen ODER 'all' freigegeben sein."""
    must: list[models.Condition] = []
    if user_roles is not None:
        must.append(models.FieldCondition(key="access", match=models.MatchAny(any=list({*user_roles, "all"}))))
    if status:
        must.append(models.FieldCondition(key="status", match=models.MatchValue(value=status)))
    if department:
        must.append(models.FieldCondition(key="department", match=models.MatchValue(value=department)))
    if doc_ids:
        must.append(models.FieldCondition(key="doc_id", match=models.MatchAny(any=doc_ids)))
    return models.Filter(must=must) if must else None


class HybridIndex:
    def __init__(self, collection: str = "aurelia", location: str | None = None, client: QdrantClient | None = None):
        self.collection = collection
        loc = location or settings.qdrant_location
        self.client = client or (QdrantClient(":memory:") if loc == ":memory:" else QdrantClient(path=loc))
        self.embedder = get_embedder()
        self.sparse = SimpleBM25Encoder()
        self.chunks: dict[str, Chunk] = {}
        self.stats: dict = {}

    # ------------------------------------------------------------------ build
    def build(self, chunks: list[Chunk], batch_size: int = 64) -> "HybridIndex":
        t0 = time.perf_counter()
        dim = embedding_dim()
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            self.collection,
            vectors_config={"dense": models.VectorParams(size=dim, distance=models.Distance.COSINE)},
            sparse_vectors_config={"sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)},
        )
        if not _is_local(self.client):  # Payload-Indizes gibt es nur beim Qdrant-Server
            for key in ("access", "status", "department", "doc_id"):
                self.client.create_payload_index(self.collection, key, models.PayloadSchemaType.KEYWORD)
        self.sparse.fit([c.text for c in chunks])
        self.chunks = {c.chunk_id: c for c in chunks}
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            dense = self.embedder.embed_documents([c.text for c in batch])
            points = [
                models.PointStruct(
                    id=_point_id(c.chunk_id),
                    vector={"dense": d, "sparse": self.sparse.encode_document(c.text)},
                    payload={"chunk_id": c.chunk_id, "text": c.text, **c.metadata},
                )
                for c, d in zip(batch, dense)
            ]
            self.client.upsert(self.collection, points)
        self.stats = {"chunks": len(chunks), "build_seconds": round(time.perf_counter() - t0, 2), "dense_dim": dim}
        return self

    def _to_hits(self, points, source: str) -> list[Hit]:
        hits = []
        for rank, p in enumerate(points, start=1):
            chunk = self.chunks.get(p.payload["chunk_id"]) or Chunk(
                p.payload["chunk_id"], p.payload["doc_id"], p.payload["text"],
                {k: v for k, v in p.payload.items() if k not in {"chunk_id", "text"}},
            )
            hits.append(Hit(chunk=chunk, score=float(p.score), rank=rank, source=source))
        return hits

    # ----------------------------------------------------------------- search
    def search_dense(self, query: str, k: int = 5, query_filter: models.Filter | None = None) -> list[Hit]:
        vec = self.embedder.embed_query(query)
        res = self.client.query_points(self.collection, query=vec, using="dense", limit=k, query_filter=query_filter, with_payload=True)
        return self._to_hits(res.points, "dense")

    def search_sparse(self, query: str, k: int = 5, query_filter: models.Filter | None = None) -> list[Hit]:
        sv = self.sparse.encode_query(query)
        res = self.client.query_points(self.collection, query=sv, using="sparse", limit=k, query_filter=query_filter, with_payload=True)
        return self._to_hits(res.points, "sparse")

    def search_hybrid(
        self, query: str, k: int = 5, prefetch_k: int = 20, fusion: Fusion = "rrf", query_filter: models.Filter | None = None
    ) -> list[Hit]:
        """Dense + Sparse parallel (je prefetch_k Kandidaten), dann Fusion durch Qdrant."""
        vec = self.embedder.embed_query(query)
        sv = self.sparse.encode_query(query)
        fusion_enum = models.Fusion.RRF if fusion == "rrf" else models.Fusion.DBSF
        res = self.client.query_points(
            self.collection,
            prefetch=[
                models.Prefetch(query=vec, using="dense", limit=prefetch_k, filter=query_filter),
                models.Prefetch(query=sv, using="sparse", limit=prefetch_k, filter=query_filter),
            ],
            query=models.FusionQuery(fusion=fusion_enum),
            limit=k,
            with_payload=True,
        )
        return self._to_hits(res.points, "hybrid")

    def search(self, query: str, mode: str = "dense", k: int = 5, **kw) -> list[Hit]:
        if mode == "dense":
            return self.search_dense(query, k, kw.get("query_filter"))
        if mode == "sparse":
            return self.search_sparse(query, k, kw.get("query_filter"))
        if mode == "hybrid":
            return self.search_hybrid(query, k, kw.get("prefetch_k", 20), kw.get("fusion", "rrf"), kw.get("query_filter"))
        raise ValueError(mode)

    def count(self) -> int:
        return self.client.count(self.collection).count


# ---------------------------------------------------------------------------
# Fusion "von Hand" - zur Veranschaulichung auf der Folie und in Lab 2
# ---------------------------------------------------------------------------
def rrf_fuse(*rankings: list[Hit], k: int = 60, top_k: int = 5) -> list[Hit]:
    """Reciprocal Rank Fusion: score = sum 1/(k + rank). Robust, keine Score-Normalisierung noetig."""
    scores: dict[str, float] = {}
    chunks: dict[str, Chunk] = {}
    for ranking in rankings:
        for h in ranking:
            scores[h.chunk.chunk_id] = scores.get(h.chunk.chunk_id, 0.0) + 1.0 / (k + h.rank)
            chunks[h.chunk.chunk_id] = h.chunk
    ordered = sorted(scores.items(), key=lambda kv: -kv[1])[:top_k]
    return [Hit(chunks[cid], s, r, "hybrid") for r, (cid, s) in enumerate(ordered, start=1)]
