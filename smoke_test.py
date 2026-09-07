"""Schneller Funktionstest der Kursumgebung. Aufruf: python smoke_test.py  (offline: FAKE_EMBEDDINGS=1)

Prueft: Imports, Korpus, Chunking, Qdrant-Index (dense/sparse/hybrid), Filter, Golden Set.
Mit gesetztem OPENAI_API_KEY zusaetzlich: ein Embedding- und ein LLM-Aufruf.
"""
import sys
import time

t0 = time.time()
errors = []

def check(label, fn):
    try:
        out = fn()
        print(f"[ok]   {label}" + (f" -> {out}" if out is not None else ""))
    except Exception as e:  # noqa: BLE001
        errors.append(label)
        print(f"[FAIL] {label}: {type(e).__name__}: {e}")


def imports():
    import langchain_core, langchain_openai, qdrant_client, pandas  # noqa: F401
    import ragkurs  # noqa: F401
    return "ragkurs, langchain, qdrant-client, pandas"

check("Imports", imports)

from ragkurs import settings  # noqa: E402
print(f"       Modelle: chat={settings.chat_model} judge={settings.judge_model} embed={settings.embed_model} fake={settings.fake_embeddings}")

def corpus():
    from ragkurs import load_corpus
    docs = load_corpus()
    assert len(docs) >= 20, "zu wenige Dokumente"
    return f"{len(docs)} Dokumente"

check("Korpus laden (Markdown + PDF via pypdf)", corpus)

def golden():
    from ragkurs.eval import load_golden
    return f"{len(load_golden())} Golden-Fragen"

check("Golden Set", golden)

def index():
    from ragkurs import HybridIndex, PipelineConfig, RAGPipeline, chunk_by_headings, load_corpus
    docs = load_corpus()
    idx = HybridIndex().build(chunk_by_headings(docs))
    q = "Wie viele Urlaubstage habe ich?"
    for mode in ("dense", "sparse", "hybrid"):
        hits = idx.search(q, mode=mode, k=3)
        assert hits, f"keine Treffer im Modus {mode}"
    pipe = RAGPipeline(idx, PipelineConfig(retrieval="hybrid"))
    res = pipe.retrieve_only(q, user_roles=["employee"])
    return f"{idx.count()} Chunks, Top-Dokument: {res.retrieved_doc_ids[0]}"

check("Qdrant-Index + Hybrid-Suche + ACL-Filter", index)

if settings.openai_api_key and not settings.fake_embeddings:
    def llm():
        from ragkurs.llm import get_llm
        return get_llm().invoke("Antworte nur mit dem Wort OK.").content.strip()[:20]
    check("OpenAI Chat-Aufruf", llm)
else:
    print("[skip] OpenAI-Aufruf (kein Key oder FAKE_EMBEDDINGS=1)")

def optional_models():
    import importlib
    missing = [m for m in ("sentence_transformers", "rerankers", "docling", "ragas", "deepeval", "phoenix") if importlib.util.find_spec(m) is None]
    return "alle installiert" if not missing else f"nicht installiert: {missing} (nur fuer spaetere Labs noetig)"

check("Optionale Pakete", optional_models)

print(f"\n{'Setup OK' if not errors else 'Setup FEHLERHAFT: ' + ', '.join(errors)}  ({time.time() - t0:.1f}s)")
sys.exit(1 if errors else 0)
