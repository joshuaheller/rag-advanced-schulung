"""Die konfigurierbare RAG-Pipeline - das zentrale Objekt aller Labs.

    pipe = RAGPipeline(index, PipelineConfig(retrieval="hybrid", rerank="fast"))
    result = pipe.run("Wie viele Urlaubstage habe ich?", user_roles=["employee"])
    result.answer.text, result.hits, result.trace

Jeder Schritt schreibt Zeiten und Zwischenergebnisse in `result.trace` - das ist die
"eingebaute Observability", die in Lab 1 (Debugging) und Lab 7 (Monitoring) genutzt wird.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal

from .chunking import ParentStore
from .config import settings
from .generate import Answer, Reorder, answer as generate_answer
from .index import Hit, HybridIndex, build_filter, rrf_fuse
from .llm import get_llm

Retrieval = Literal["dense", "sparse", "hybrid"]
RerankKind = Literal["fast", "quality", "colbert", "llm"] | None
QueryTransform = Literal["rewrite", "multi", "decompose", "hyde", "stepback"] | None


@dataclass
class PipelineConfig:
    retrieval: Retrieval = "dense"
    k: int = 5                      # Treffer, die in den Kontext gehen
    prefetch_k: int = 20            # Kandidaten je Retriever (hybrid) bzw. vor dem Reranker
    fusion: Literal["rrf", "dbsf"] = "rrf"
    query_transform: QueryTransform = None
    rerank: RerankKind = None
    parent_expand: bool = False     # Kinder suchen, Eltern liefern (Parent-Child-Chunks)
    reorder: Reorder = "none"
    max_context_chars: int = 6000
    status_filter: str | None = "current"   # nur aktuelle Dokumente
    enforce_acl: bool = True        # Pre-Filter nach Rollen
    name: str = ""

    def label(self) -> str:
        if self.name:
            return self.name
        parts = [self.retrieval]
        if self.query_transform:
            parts.append(self.query_transform)
        if self.rerank:
            parts.append(f"rerank:{self.rerank}")
        if self.parent_expand:
            parts.append("parent")
        if self.reorder != "none":
            parts.append(self.reorder)
        return "+".join(parts)


@dataclass
class RunResult:
    query: str
    answer: Answer
    hits: list[Hit]
    trace: dict = field(default_factory=dict)

    @property
    def retrieved_doc_ids(self) -> list[str]:
        return list(dict.fromkeys(h.doc_id for h in self.hits))

    def show(self, n_chars: int = 300) -> None:
        print(f"Frage : {self.query}")
        print(f"Antwort: {self.answer.text}\n")
        for h in self.hits:
            print(f"  #{h.rank} {h.score:.3f} {h.chunk.chunk_id} [{h.chunk.section[:60]}]")
            print("     " + h.chunk.text[:n_chars].replace("\n", " ") + ("..." if len(h.chunk.text) > n_chars else ""))
        t = self.trace
        print(f"\nZeit: retrieval {t.get('t_retrieval_ms', 0):.0f} ms | rerank {t.get('t_rerank_ms', 0):.0f} ms | "
              f"generation {t.get('t_generation_ms', 0):.0f} ms | tokens {self.answer.usage}")


class RAGPipeline:
    def __init__(self, index: HybridIndex, config: PipelineConfig | None = None, parent_store: ParentStore | None = None, llm=None):
        self.index = index
        self.config = config or PipelineConfig()
        self.parent_store = parent_store
        self._llm = llm
        self.log: list[RunResult] = []  # alle Laeufe (fuer Lab 7: Auswertung wie in Produktion)

    @property
    def llm(self):
        if self._llm is None:
            self._llm = get_llm()
        return self._llm

    # ----------------------------------------------------------- Teilschritte
    def _transform(self, query: str, trace: dict) -> list[str]:
        from . import query as qt

        kind = self.config.query_transform
        if kind is None:
            return [query]
        t0 = time.perf_counter()
        if kind == "rewrite":
            queries = [qt.rewrite(query, self.llm)]
        elif kind == "multi":
            queries = qt.multi_query(query, 3, self.llm)
        elif kind == "decompose":
            queries = qt.decompose(query, self.llm)
        elif kind == "hyde":
            queries = [qt.hyde(query, self.llm)]
        elif kind == "stepback":
            queries = [query, qt.step_back(query, self.llm)]
        else:
            raise ValueError(kind)
        trace["t_transform_ms"] = (time.perf_counter() - t0) * 1000
        trace["queries"] = queries
        return queries

    def _retrieve(self, queries: list[str], user_roles: list[str] | None, trace: dict) -> list[Hit]:
        cfg = self.config
        flt = build_filter(user_roles=user_roles if cfg.enforce_acl else None, status=cfg.status_filter)
        n = cfg.prefetch_k if (cfg.rerank or len(queries) > 1) else cfg.k
        t0 = time.perf_counter()
        rankings = [self.index.search(q, mode=cfg.retrieval, k=n, prefetch_k=cfg.prefetch_k, fusion=cfg.fusion, query_filter=flt) for q in queries]
        hits = rankings[0] if len(rankings) == 1 else rrf_fuse(*rankings, top_k=n)
        trace["t_retrieval_ms"] = (time.perf_counter() - t0) * 1000
        trace["candidates"] = [(h.chunk.chunk_id, round(h.score, 4)) for h in hits]
        trace["filter"] = str(flt) if flt else None
        return hits

    def _rerank(self, query: str, hits: list[Hit], trace: dict) -> list[Hit]:
        if not self.config.rerank:
            return hits[: self.config.k]
        from .rerank import get_reranker

        t0 = time.perf_counter()
        out = get_reranker(self.config.rerank).rerank(query, hits, top_k=self.config.k)
        trace["t_rerank_ms"] = (time.perf_counter() - t0) * 1000
        trace["reranked"] = [(h.chunk.chunk_id, round(h.score, 4), h.extra.get("pre_rank")) for h in out]
        return out

    def _expand(self, hits: list[Hit]) -> list[Hit]:
        if not (self.config.parent_expand and self.parent_store):
            return hits
        out, seen = [], set()
        for h in hits:
            pid = h.chunk.metadata.get("parent_id")
            if not pid or pid in seen:
                if not pid:
                    out.append(h)
                continue
            seen.add(pid)
            parent = self.parent_store.get(pid)
            out.append(Hit(parent, h.score, len(out) + 1, "parent", {"child_id": h.chunk.chunk_id}))
        return out

    # ------------------------------------------------------------------ run
    def retrieve_only(self, query: str, user_roles: list[str] | None = None) -> RunResult:
        """Retrieval + Rerank ohne Generierung (fuer Retrieval-Metriken, spart LLM-Kosten)."""
        trace: dict = {"config": self.config.label()}
        queries = self._transform(query, trace)
        hits = self._expand(self._rerank(query, self._retrieve(queries, user_roles, trace), trace))
        return RunResult(query, Answer("", hits, "", {}), hits, trace)

    def run(self, query: str, user_roles: list[str] | None = None) -> RunResult:
        trace: dict = {"config": self.config.label(), "user_roles": user_roles}
        t_all = time.perf_counter()
        queries = self._transform(query, trace)
        hits = self._expand(self._rerank(query, self._retrieve(queries, user_roles, trace), trace))
        t0 = time.perf_counter()
        ans = generate_answer(query, hits, llm=self.llm, max_chars=self.config.max_context_chars, reorder=self.config.reorder)
        trace["t_generation_ms"] = (time.perf_counter() - t0) * 1000
        trace["t_total_ms"] = (time.perf_counter() - t_all) * 1000
        trace["usage"] = ans.usage
        trace["cost_usd"] = estimate_cost(ans.usage)
        trace["context_chars"] = len(ans.context)
        result = RunResult(query, ans, ans.sources, trace)
        self.log.append(result)
        return result


def estimate_cost(usage: dict) -> float:
    return round(usage.get("input_tokens", 0) / 1e6 * settings.price_in + usage.get("output_tokens", 0) / 1e6 * settings.price_out, 6)
