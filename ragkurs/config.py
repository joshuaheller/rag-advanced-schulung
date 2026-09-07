"""Zentrale Konfiguration des Kurs-Pakets `ragkurs`.

Alle Einstellungen kommen aus Umgebungsvariablen bzw. der Datei `.env` im Repo-Root
(siehe `.env.example`). Nichts davon erfordert einen Account ausser dem OpenAI-Key.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

# Telemetrie der verwendeten Open-Source-Bibliotheken abschalten (keine Daten nach aussen)
os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    # --- OpenAI ---
    openai_api_key: str | None = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    chat_model: str = field(default_factory=lambda: os.getenv("OPENAI_CHAT_MODEL", "gpt-5.6-luna"))
    judge_model: str = field(default_factory=lambda: os.getenv("OPENAI_JUDGE_MODEL", "gpt-5.6-terra"))
    embed_model: str = field(default_factory=lambda: os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small"))
    embed_dim: int = field(default_factory=lambda: int(os.getenv("OPENAI_EMBED_DIM", "1536")))
    # "none" = kein Reasoning (schnell, guenstig). Leer lassen, falls das Modell den Parameter nicht kennt.
    reasoning_effort: str | None = field(default_factory=lambda: os.getenv("OPENAI_REASONING_EFFORT", "none") or None)

    # --- Offline-Modus fuer Tests ohne API-Key (Fake-Embeddings, kein LLM) ---
    fake_embeddings: bool = field(default_factory=lambda: _bool("FAKE_EMBEDDINGS", False))

    # --- Pfade ---
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("RAG_DATA_DIR", str(ROOT / "data"))))
    cache_dir: Path = field(default_factory=lambda: Path(os.getenv("RAG_CACHE_DIR", str(ROOT / ".cache"))))
    # Qdrant laeuft im Python-Prozess ("lokaler Modus"). ":memory:" = fluechtig, Pfad = persistent.
    qdrant_location: str = field(default_factory=lambda: os.getenv("QDRANT_LOCATION", ":memory:"))

    # --- Open-Source-Modelle (werden beim ersten Aufruf von Hugging Face geladen) ---
    reranker_fast: str = field(default_factory=lambda: os.getenv("RERANKER_FAST", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"))
    reranker_quality: str = field(default_factory=lambda: os.getenv("RERANKER_QUALITY", "BAAI/bge-reranker-v2-m3"))
    colbert_model: str = field(default_factory=lambda: os.getenv("COLBERT_MODEL", "answerdotai/answerai-colbert-small-v1"))

    # --- Preise (USD pro 1 Mio. Tokens) fuer Kostenschaetzung, anpassbar per .env ---
    price_in: float = field(default_factory=lambda: float(os.getenv("PRICE_INPUT_PER_MTOK", "0.20")))
    price_out: float = field(default_factory=lambda: float(os.getenv("PRICE_OUTPUT_PER_MTOK", "1.20")))
    price_embed: float = field(default_factory=lambda: float(os.getenv("PRICE_EMBED_PER_MTOK", "0.02")))

    @property
    def corpus_dir(self) -> Path:
        return self.data_dir / "corpus"

    @property
    def golden_path(self) -> Path:
        return self.data_dir / "golden" / "golden_set.jsonl"

    @property
    def poison_dir(self) -> Path:
        return self.data_dir / "poison"

    def require_openai(self) -> None:
        if self.fake_embeddings:
            return
        if not self.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY fehlt. Bitte `.env` anlegen (siehe .env.example) oder FAKE_EMBEDDINGS=1 setzen."
            )


settings = Settings()
settings.cache_dir.mkdir(parents=True, exist_ok=True)
