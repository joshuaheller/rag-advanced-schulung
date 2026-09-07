"""LLM- und Embedding-Zugriff (OpenAI via LangChain) mit Embedding-Cache und Offline-Fallback."""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from .config import settings

# ---------------------------------------------------------------------------
# Chat-LLM
# ---------------------------------------------------------------------------
_llm_cache: dict[str, object] = {}


class OfflineChatModel(BaseChatModel):
    """Offline-Ersatz fuer Tests ohne API-Key (FAKE_EMBEDDINGS=1): liefert plausible Platzhalter."""

    model_name: str = "offline"

    @property
    def _llm_type(self) -> str:
        return "offline"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        text = messages[-1].content if messages else ""
        if '"question"' in text and '"ground_truth"' in text:
            out = '{"question": "Offline-Testfrage?", "ground_truth": "Offline-Referenzantwort."}'
        elif '"verdaechtig"' in text:
            out = '{"verdaechtig": true, "grund": "Offline-Modus"}'
        elif '"correct"' in text:
            out = '{"correct": false, "reason": "Offline-Modus - kein LLM verfuegbar"}'
        elif "JSON-Liste der Passagen" in text:
            out = "[1, 2, 3, 4, 5]"
        elif "JSON-Liste" in text:
            out = '["Variante A der Frage", "Variante B der Frage"]'
        else:
            out = "Offline-Antwort: kein LLM verfuegbar (FAKE_EMBEDDINGS=1) [1]"
        n_in = max(1, len(text) // 4)
        msg = AIMessage(content=out, usage_metadata={"input_tokens": n_in, "output_tokens": 12, "total_tokens": n_in + 12})
        return ChatResult(generations=[ChatGeneration(message=msg)])


def get_llm(model: str | None = None, **kwargs):
    """ChatOpenAI-Instanz (gecacht). `model=None` -> settings.chat_model. Offline: OfflineChatModel."""
    if settings.fake_embeddings:
        return _llm_cache.setdefault("offline", OfflineChatModel())
    from langchain_openai import ChatOpenAI

    settings.require_openai()
    model = model or settings.chat_model
    key = model + json.dumps(kwargs, sort_keys=True)
    if key not in _llm_cache:
        params = dict(model=model, api_key=settings.openai_api_key)
        if settings.reasoning_effort and model.startswith("gpt-5"):
            params["reasoning_effort"] = settings.reasoning_effort
        params.update(kwargs)
        _llm_cache[key] = ChatOpenAI(**params)
    return _llm_cache[key]


def get_judge_llm():
    return get_llm(settings.judge_model)


def usage_of(message) -> dict:
    """Token-Verbrauch aus einer LangChain-AIMessage extrahieren (robust gegen fehlende Felder)."""
    u = getattr(message, "usage_metadata", None) or {}
    return {"input_tokens": int(u.get("input_tokens", 0)), "output_tokens": int(u.get("output_tokens", 0))}


# ---------------------------------------------------------------------------
# Embeddings mit Disk-Cache (spart Kosten bei wiederholten Notebook-Laeufen)
# ---------------------------------------------------------------------------
class CachedEmbeddings(Embeddings):
    def __init__(self, inner: Embeddings, model_name: str, cache_dir: Path | None = None):
        self.inner = inner
        self.model_name = model_name
        self.path = (cache_dir or settings.cache_dir) / f"embeddings_{re.sub(r'[^a-z0-9]+', '_', model_name.lower())}.jsonl"
        self._mem: dict[str, list[float]] = {}
        self.stats = {"hits": 0, "misses": 0}
        if self.path.exists():
            with self.path.open(encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        rec = json.loads(line)
                        self._mem[rec["k"]] = rec["v"]

    def _key(self, text: str) -> str:
        return hashlib.sha1((self.model_name + "\x00" + text).encode("utf-8")).hexdigest()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        keys = [self._key(t) for t in texts]
        missing = [(k, t) for k, t in zip(keys, texts) if k not in self._mem]
        self.stats["hits"] += len(texts) - len(missing)
        self.stats["misses"] += len(missing)
        if missing:
            vecs = self.inner.embed_documents([t for _, t in missing])
            with self.path.open("a", encoding="utf-8") as f:
                for (k, _), v in zip(missing, vecs):
                    self._mem[k] = v
                    f.write(json.dumps({"k": k, "v": v}) + "\n")
        return [self._mem[k] for k in keys]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class HashingFakeEmbeddings(Embeddings):
    """Offline-Ersatz: Bag-of-Words-Hashing. Lexikalisch aehnliche Texte liegen nah beieinander.

    Nur fuer Smoke-Tests ohne API-Key - KEINE semantische Suche.
    """

    def __init__(self, dim: int = 256):
        self.dim = dim

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for tok in re.findall(r"\w+", text.lower()):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0 if (h >> 40) % 2 else -1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


_embedder = None


def get_embedder() -> Embeddings:
    """Embedding-Modell (gecacht). Bei FAKE_EMBEDDINGS=1 ohne API."""
    global _embedder
    if _embedder is None:
        if settings.fake_embeddings:
            _embedder = HashingFakeEmbeddings()
        else:
            from langchain_openai import OpenAIEmbeddings

            settings.require_openai()
            inner = OpenAIEmbeddings(model=settings.embed_model, api_key=settings.openai_api_key)
            _embedder = CachedEmbeddings(inner, settings.embed_model)
    return _embedder


def embedding_dim() -> int:
    return 256 if settings.fake_embeddings else settings.embed_dim
