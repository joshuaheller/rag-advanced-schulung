# %% [markdown]
# # Lab 3 – Reranking & Relevance Optimization
#
# **Ziel (30 Min):** Cross-Encoder und Late-Interaction-Reranker in die Pipeline einbauen und die
# Trade-off-Tabelle **Qualität × Latenz × Kosten** mit echten Zahlen füllen.
#
# **Warum ein zweiter Schritt?** Ein Bi-Encoder (Embedding) komprimiert Frage und Passage *getrennt* in je einen
# Vektor – schnell, aber grob. Ein Cross-Encoder sieht Frage **und** Passage *gemeinsam* und kann Feinheiten
# bewerten („Kündigung *durch* den Arbeitnehmer“ vs. „*durch* den Arbeitgeber“). Er ist zu langsam für den ganzen
# Korpus, aber ideal für die Top-20 bis Top-100 Kandidaten. Typisches Muster: **Recall@50 → Precision@5**.
#
# **Modelle in diesem Lab (alle Open Source, lokal, ohne Account):**
#
# | Kürzel | Modell | Typ | Größe |
# |---|---|---|---|
# | `fast` | cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 | Cross-Encoder, multilingual | ~470 MB |
# | `quality` | BAAI/bge-reranker-v2-m3 | Cross-Encoder, multilingual, 8k Kontext | ~2,2 GB |
# | `colbert` | answerdotai/answerai-colbert-small-v1 | Late Interaction (Token-MaxSim), primär Englisch | ~130 MB |
# | `llm` | Chat-Modell (listwise) | LLM-Reranker | API |

# %%
import sys, os, time
_root = os.getcwd()                      # Repo-Root finden (Notebook liegt in labs/ oder labs/solutions/)
while not os.path.isdir(os.path.join(_root, "ragkurs")):
    _root = os.path.dirname(_root)
sys.path.insert(0, _root); os.chdir(_root)
import pandas as pd
pd.set_option("display.max_colwidth", 80)

from ragkurs import load_corpus, chunk_by_headings, HybridIndex, RAGPipeline, PipelineConfig
from ragkurs.eval import load_golden, evaluate_retrieval, retrieval_summary, compare_retrieval
from ragkurs.rerank import get_reranker, timed_rerank

docs = load_corpus()
index = HybridIndex(collection="lab3").build(chunk_by_headings(docs))
golden = load_golden()

# %% [markdown]
# ## Teil A – Walkthrough
#
# ### A1 Reranker laden (erster Aufruf lädt das Modell – bei vorbereiteten VMs aus dem Cache)

# %%
t0 = time.perf_counter(); fast = get_reranker("fast"); print(f"fast geladen in {time.perf_counter()-t0:.1f}s")

# %% [markdown]
# ### A2 Vorher / Nachher an einer Near-Miss-Frage

# %%
q = "Mit welcher Frist kann Aurelia einem Mitarbeiter kündigen, der seit 9 Jahren im Unternehmen ist?"
candidates = index.search_hybrid(q, k=20, prefetch_k=30)
print("Hybrid Top-5 (vor Reranking):")
for h in candidates[:5]:
    print(f"  #{h.rank} {h.score:.3f} {h.chunk.chunk_id}  {h.chunk.section[:70]}")

reranked, ms = timed_rerank(fast, q, candidates, top_k=5)
print(f"\nNach Cross-Encoder ({ms:.0f} ms für {len(candidates)} Kandidaten):")
for h in reranked:
    print(f"  #{h.rank} {h.score:.3f} (vorher #{h.extra['pre_rank']}) {h.chunk.chunk_id}  {h.chunk.section[:70]}")

# %% [markdown]
# ### A3 Reranker in der Pipeline: Recall@20 → Precision@5

# %%
pipes = [
    RAGPipeline(index, PipelineConfig(retrieval="hybrid", k=5, prefetch_k=20, name="hybrid")),
    RAGPipeline(index, PipelineConfig(retrieval="hybrid", k=5, prefetch_k=20, rerank="fast", name="hybrid+rerank:fast")),
]
compare_retrieval(pipes, golden, user_roles=["employee"])

# %% [markdown]
# ### A4 Wie viele Kandidaten braucht der Reranker?
#
# Das `prefetch_k` ist der wichtigste Kostenhebel: Cross-Encoder-Latenz wächst *linear* mit der Kandidatenzahl.

# %%
rows = []
for pk in (5, 10, 20, 40):
    p = RAGPipeline(index, PipelineConfig(retrieval="hybrid", k=5, prefetch_k=pk, rerank="fast", name=f"pf{pk}"))
    s = retrieval_summary(evaluate_retrieval(p, golden, ["employee"]))
    rows.append({"prefetch_k": pk, **{k: v for k, v in s.items() if k != "config"}})
pd.DataFrame(rows)

# %% [markdown]
# ## Teil B – Aufgaben
#
# ### B1 Die Trade-off-Tabelle füllen
# Vergleicht die Reranker `fast`, `quality`, `colbert` (und optional `llm`) auf Hybrid-Top-20:
# Hit-Rate, MRR, **p50/p95-Latenz pro Anfrage** und geschätzte **Kosten pro 1.000 Anfragen**
# (lokale Modelle: CPU-Zeit; LLM: Tokens). Die Latenzen stehen in `t_retrieval_ms` des Retrieval-DataFrames.
#
# Hinweis: `quality` (2,2 GB) braucht auf CPU spürbar länger – das *ist* der Punkt der Übung.

# %%
# === LOESUNG START ===
# HINWEIS: evaluate_retrieval(...)["t_retrieval_ms"].quantile(0.95) ; get_reranker("quality") lädt das große Modell
results = []
for kind in ("fast", "quality", "colbert"):
    t0 = time.perf_counter(); get_reranker(kind); load_s = time.perf_counter() - t0
    p = RAGPipeline(index, PipelineConfig(retrieval="hybrid", k=5, prefetch_k=20, rerank=kind, name=f"rerank:{kind}"))  # ? Pipeline mit rerank=kind
    df = evaluate_retrieval(p, golden, ["employee"])
    s = retrieval_summary(df)
    results.append({
        "reranker": kind, "hit_rate": s["hit_rate"], "mrr": s["mrr"],
        "p50_ms": round(df["t_retrieval_ms"].quantile(0.5)),  # ? Median der Spalte t_retrieval_ms
        "p95_ms": round(df["t_retrieval_ms"].quantile(0.95)),  # ? 95 %-Quantil der Spalte t_retrieval_ms
        "modell_laden_s": round(load_s, 1),
    })
tradeoff = pd.DataFrame(results)
tradeoff
# === LOESUNG ENDE ===

# %% [markdown]
# ### B2 Kaskade bauen
# Baut eine **zweistufige** Kaskade von Hand: Hybrid Top-40 → `fast` Top-10 → `quality` Top-5.
# Vergleicht Qualität und Latenz mit `quality` direkt auf Top-40. Wann lohnt sich die Kaskade?

# %%
# === LOESUNG START ===
# HINWEIS: reranker.rerank(query, hits, top_k) lässt sich verketten; Zeit mit time.perf_counter() messen
from ragkurs.eval import retrieval_row
fast_r, qual_r = get_reranker("fast"), get_reranker("quality")

def cascade(q, roles=("employee",)):
    from ragkurs.index import build_filter
    flt = build_filter(user_roles=list(roles))
    c = index.search_hybrid(q, k=40, prefetch_k=40, query_filter=flt)
    return qual_r.rerank(q, fast_r.rerank(q, c, top_k=10), top_k=5)  # ? erst fast auf Top-10, dann quality auf Top-5

def direct(q, roles=("employee",)):
    from ragkurs.index import build_filter
    flt = build_filter(user_roles=list(roles))
    return qual_r.rerank(q, index.search_hybrid(q, k=40, prefetch_k=40, query_filter=flt), top_k=5)  # ? quality direkt auf die 40 Kandidaten

from ragkurs.pipeline import RunResult
from ragkurs.generate import Answer
rows = []
for name, fn in (("kaskade fast->quality", cascade), ("quality direkt", direct)):
    for g in golden:
        if not g["source_docs"]:
            continue
        t0 = time.perf_counter(); hits = fn(g["question"]); ms = (time.perf_counter() - t0) * 1000
        rr = retrieval_row(RunResult(g["question"], Answer("", hits, "", {}), hits, {"t_retrieval_ms": ms}), g, 5)
        rows.append({"variante": name, "hit": rr["hit"], "rr": rr["rr"], "ms": ms})
pd.DataFrame(rows).groupby("variante").agg(hit_rate=("hit", "mean"), mrr=("rr", "mean"), p50_ms=("ms", "median"), p95_ms=("ms", lambda s: s.quantile(0.95))).round(3)
# === LOESUNG ENDE ===

# %% [markdown]
# ### B3 Reranker-Auswahl für *euren* Fall
# Füllt für euer eigenes Projekt aus (Markdown): Sprache der Dokumente, typische Chunk-Länge, Anfragen pro Tag,
# Latenzbudget, darf ein API-Reranker (Cohere/Jina) Daten sehen? → Welcher Reranker, welches `prefetch_k`?

# %% [markdown]
# **Unsere Wahl:** ...

# %% [markdown]
# ### B4 (Bonus) LLM als Reranker
# `rerank="llm"` sortiert listwise per Chat-Modell. Messt Qualität und Latenz auf 10 Fragen. Wann ist das die
# richtige Wahl (Stichwort: komplexe Relevanzkriterien, geringe Anfragezahl)?

# %%
# === LOESUNG START ===
# HINWEIS: PipelineConfig(rerank="llm", prefetch_k=10) ; nur golden[:10] verwenden (Kosten/Zeit)
p = RAGPipeline(index, PipelineConfig(retrieval="hybrid", k=5, prefetch_k=10, rerank="llm", name="rerank:llm"))  # ? rerank="llm", prefetch_k=10
retrieval_summary(evaluate_retrieval(p, golden[:10], ["employee"]))
# === LOESUNG ENDE ===

# %% [markdown]
# ## Teil C – Debrief
#
# 1. Wie viel Hit-Rate bringt der Reranker – und wie viel Latenz kostet er? Ist das Verhältnis akzeptabel?
# 2. ColBERT ist primär englisch trainiert. Was habt ihr auf deutschen Texten beobachtet? Was folgt daraus für die Modellwahl?
# 3. Anti-Pattern-Check: Reranker auf nur 5 Kandidaten – warum bringt das fast nichts?
#
# **Merksatz:** Der Reranker repariert die Reihenfolge, nicht den Recall. Was der Retriever nicht liefert,
# kann der Reranker nicht nach vorne holen.
