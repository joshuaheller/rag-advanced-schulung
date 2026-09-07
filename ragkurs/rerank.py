"""Reranker: Cross-Encoder (schnell / Qualitaet), Late Interaction (ColBERT), LLM-Listwise.

Alle Modelle sind Open Source und werden beim ersten Aufruf von Hugging Face geladen
(kein Account noetig). Der LLM-Reranker nutzt die OpenAI-API.
"""
from __future__ import annotations

import json
import re
import time
from typing import Protocol

from .config import settings
from .index import Hit


class Reranker(Protocol):
    name: str

    def rerank(self, query: str, hits: list[Hit], top_k: int = 5) -> list[Hit]: ...


def _rescored(hits: list[Hit], scores: list[float], top_k: int, name: str) -> list[Hit]:
    paired = sorted(zip(hits, scores), key=lambda p: -p[1])[:top_k]
    return [
        Hit(chunk=h.chunk, score=float(s), rank=r, source="rerank", extra={**h.extra, "pre_rank": h.rank, "pre_score": h.score, "reranker": name})
        for r, (h, s) in enumerate(paired, start=1)
    ]


class CrossEncoderReranker:
    """Cross-Encoder: Query und Passage werden GEMEINSAM durch das Modell geschickt -> ein Relevanz-Score."""

    def __init__(self, model_name: str | None = None, max_length: int = 512):
        from sentence_transformers import CrossEncoder

        self.name = model_name or settings.reranker_fast
        self.model = CrossEncoder(self.name, max_length=max_length)

    def rerank(self, query: str, hits: list[Hit], top_k: int = 5) -> list[Hit]:
        if not hits:
            return []
        scores = self.model.predict([(query, h.chunk.text) for h in hits]).tolist()
        return _rescored(hits, scores, top_k, self.name)


class ColbertReranker:
    """Late Interaction: Token-Embeddings von Query und Passage, MaxSim-Summe. (Bibliothek `rerankers`)"""

    def __init__(self, model_name: str | None = None):
        from rerankers import Reranker as _RR

        self.name = model_name or settings.colbert_model
        self.model = _RR(self.name, model_type="colbert", verbose=0)

    def rerank(self, query: str, hits: list[Hit], top_k: int = 5) -> list[Hit]:
        if not hits:
            return []
        res = self.model.rank(query=query, docs=[h.chunk.text for h in hits], doc_ids=list(range(len(hits))))
        scores = [0.0] * len(hits)
        for r in res.results:
            scores[int(r.doc_id)] = float(r.score)
        return _rescored(hits, scores, top_k, self.name)


_LLM_RERANK_PROMPT = """Du bist ein Such-Reranker. Ordne die Passagen nach Relevanz fuer die Frage.
Frage: {query}

Passagen:
{passages}

Antworte NUR mit einer JSON-Liste der Passagen-Nummern, relevanteste zuerst, z. B. [3, 1, 2]."""


class LLMReranker:
    """Listwise Reranking per LLM - teuer und langsam, aber stark bei komplexen Kriterien."""

    def __init__(self, llm=None):
        from .llm import get_llm

        self.llm = llm or get_llm()
        self.name = f"llm:{getattr(self.llm, 'model_name', 'chat')}"

    def rerank(self, query: str, hits: list[Hit], top_k: int = 5) -> list[Hit]:
        if not hits:
            return []
        passages = "\n\n".join(f"[{i + 1}] {h.chunk.text[:800]}" for i, h in enumerate(hits))
        raw = self.llm.invoke(_LLM_RERANK_PROMPT.format(query=query, passages=passages)).content
        m = re.search(r"\[[\d,\s]+\]", raw)
        order = json.loads(m.group(0)) if m else list(range(1, len(hits) + 1))
        scores = [0.0] * len(hits)
        for pos, num in enumerate(order):
            if 1 <= int(num) <= len(hits):
                scores[int(num) - 1] = float(len(hits) - pos)
        return _rescored(hits, scores, top_k, self.name)


class LexicalReranker:
    """Offline-Ersatz (FAKE_EMBEDDINGS=1): Token-Ueberlappung statt Modell. Nur fuer Tests ohne Internet."""

    def __init__(self, name: str = "offline-lexical"):
        self.name = name

    def rerank(self, query: str, hits: list[Hit], top_k: int = 5) -> list[Hit]:
        from .sparse import tokenize

        q = set(tokenize(query))
        scores = [len(q & set(tokenize(h.chunk.text))) / (len(q) or 1) for h in hits]
        return _rescored(hits, scores, top_k, self.name)


_registry: dict[str, object] = {}


def get_reranker(kind: str):
    """'fast' | 'quality' | 'colbert' | 'llm' (Instanzen werden gecacht - Modelle laden dauert)."""
    if kind not in _registry:
        if settings.fake_embeddings and kind != "llm":
            _registry[kind] = LexicalReranker(f"offline-{kind}")
        elif kind == "fast":
            _registry[kind] = CrossEncoderReranker(settings.reranker_fast)
        elif kind == "quality":
            _registry[kind] = CrossEncoderReranker(settings.reranker_quality)
        elif kind == "colbert":
            try:
                _registry[kind] = ColbertReranker()
            except Exception as e:  # noqa: BLE001  (rerankers <-> transformers 5.x: bekannter Konflikt)
                raise RuntimeError(
                    "ColBERT-Reranker konnte nicht geladen werden (Bibliothekskonflikt rerankers/transformers). "
                    "Im Kurs: diesen Reranker ueberspringen - das Konzept steht auf den Folien. "
                    f"Original: {type(e).__name__}: {e}"
                ) from e
        elif kind == "llm":
            _registry[kind] = LLMReranker()
        else:
            raise ValueError(f"Unbekannter Reranker: {kind}")
    return _registry[kind]


def timed_rerank(reranker, query: str, hits: list[Hit], top_k: int = 5) -> tuple[list[Hit], float]:
    t0 = time.perf_counter()
    out = reranker.rerank(query, hits, top_k)
    return out, (time.perf_counter() - t0) * 1000.0
