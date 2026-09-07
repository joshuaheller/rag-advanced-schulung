"""Semantic Cache: aehnliche Fragen bekommen die gecachte Antwort (spart LLM-Calls und Latenz)."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

from .llm import get_embedder
from .pipeline import RAGPipeline, RunResult


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


@dataclass
class CacheEntry:
    query: str
    vector: list[float]
    result: RunResult
    roles_key: str
    created: float = field(default_factory=time.time)


class SemanticCache:
    """Cache-Key = Embedding der Frage. Treffer, wenn Cosine >= threshold UND gleiche Rollen (ACL!)."""

    def __init__(self, threshold: float = 0.92, ttl_seconds: float = 3600):
        self.threshold = threshold
        self.ttl = ttl_seconds
        self.entries: list[CacheEntry] = []
        self.embedder = get_embedder()
        self.stats = {"hits": 0, "misses": 0, "saved_ms": 0.0}

    @staticmethod
    def _roles_key(roles: list[str] | None) -> str:
        return ",".join(sorted(roles or ["all"]))

    def lookup(self, query: str, roles: list[str] | None = None) -> tuple[RunResult | None, float]:
        vec = self.embedder.embed_query(query)
        rk = self._roles_key(roles)
        best, best_sim = None, 0.0
        now = time.time()
        for e in self.entries:
            if e.roles_key != rk or now - e.created > self.ttl:
                continue
            sim = cosine(vec, e.vector)
            if sim > best_sim:
                best, best_sim = e, sim
        if best and best_sim >= self.threshold:
            return best.result, best_sim
        return None, best_sim

    def store(self, query: str, roles: list[str] | None, result: RunResult) -> None:
        self.entries.append(CacheEntry(query, self.embedder.embed_query(query), result, self._roles_key(roles)))


class CachedPipeline:
    """Wrapper: erst Cache, dann Pipeline."""

    def __init__(self, pipeline: RAGPipeline, cache: SemanticCache | None = None):
        self.pipeline = pipeline
        self.cache = cache or SemanticCache()

    def run(self, query: str, user_roles: list[str] | None = None) -> RunResult:
        t0 = time.perf_counter()
        cached, sim = self.cache.lookup(query, user_roles)
        if cached is not None:
            self.cache.stats["hits"] += 1
            self.cache.stats["saved_ms"] += cached.trace.get("t_total_ms", 0)
            trace = dict(cached.trace) | {"cache": "hit", "cache_similarity": round(sim, 3), "t_total_ms": (time.perf_counter() - t0) * 1000, "cost_usd": 0.0}
            return RunResult(query, cached.answer, cached.hits, trace)
        self.cache.stats["misses"] += 1
        res = self.pipeline.run(query, user_roles)
        res.trace["cache"] = "miss"
        res.trace["cache_similarity"] = round(sim, 3)
        self.cache.store(query, user_roles, res)
        return res
