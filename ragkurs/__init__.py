"""ragkurs - Begleitpaket zur Schulung "RAG in der Praxis - Advanced" (IT-Schulungen.com / TAISC).

Module:
    config    Einstellungen (.env), Modelle, Pfade
    loading   Korpus laden (Markdown + PDF, pypdf/docling)
    chunking  Chunking-Strategien (fixed, headings, parent-child, contextual)
    llm       LLM & Embeddings (OpenAI, Cache, Offline-Fake)
    sparse    BM25-Encoder fuer Sparse-Vektoren
    index     Qdrant-Index: dense / sparse / hybrid + Filter
    rerank    Cross-Encoder, ColBERT, LLM-Reranker
    query     Query-Transformationen
    generate  Context Packing, Re-Ordering, Antwort mit Quellen
    pipeline  RAGPipeline + PipelineConfig + Trace
    eval      Golden Set, Metriken, Judge, Failure-Klassifikation, Ragas
    security  Injection-Scan, Ingest-Gate, ACL-Leak-Test
    cache     Semantic Cache
    tracing   Phoenix-Observability
"""
from .chunking import Chunk, chunk_by_headings, chunk_fixed, chunk_parent_child
from .config import settings
from .index import HybridIndex, build_filter
from .loading import Document, load_corpus
from .pipeline import PipelineConfig, RAGPipeline, RunResult

__all__ = [
    "settings", "Document", "load_corpus", "Chunk", "chunk_fixed", "chunk_by_headings", "chunk_parent_child",
    "HybridIndex", "build_filter", "PipelineConfig", "RAGPipeline", "RunResult",
]
