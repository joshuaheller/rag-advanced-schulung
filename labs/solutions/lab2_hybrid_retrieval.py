# %% [markdown]
# # Lab 2 – Fortgeschrittene Retrieval-Strategien: Hybrid Search, Fusion, Query-Transformationen
#
# **Ziel (35 Min):** Dense- und Sparse-Retrieval kombinieren, Score-Fusion verstehen und Query-Transformationen
# messbar bewerten – nicht glauben, sondern gegen das Golden Set messen.
#
# **Kernidee:** Dense-Embeddings verstehen *Bedeutung* („Kündigungsfrist“ ≈ „wie lange muss ich bleiben“),
# BM25 versteht *exakte Terme* („AX-200“, „E-101“, „§ 4“). Produktive Systeme brauchen beides.

# %%
import sys, os
_root = os.getcwd()                      # Repo-Root finden (Notebook liegt in labs/ oder labs/solutions/)
while not os.path.isdir(os.path.join(_root, "ragkurs")):
    _root = os.path.dirname(_root)
sys.path.insert(0, _root); os.chdir(_root)
import pandas as pd
pd.set_option("display.max_colwidth", 80)

from ragkurs import load_corpus, chunk_by_headings, HybridIndex, RAGPipeline, PipelineConfig
from ragkurs.eval import load_golden, evaluate_retrieval, retrieval_summary, compare_retrieval
from ragkurs.sparse import SimpleBM25Encoder, tokenize

docs = load_corpus()
chunks = chunk_by_headings(docs)          # ab jetzt struktur-erhaltendes Chunking (Details in Lab 4)
index = HybridIndex(collection="lab2").build(chunks)
golden = load_golden()
print(index.stats)

# %% [markdown]
# ## Teil A – Walkthrough
#
# ### A1 BM25 in 3 Minuten
#
# Sparse-Vektor = Wörterbuch `{Term-ID: Gewicht}`. Gewicht steigt mit der Termhäufigkeit im Chunk, aber
# **gesättigt** (k1) und **längennormalisiert** (b). Seltene Terme bekommen über die IDF mehr Gewicht – die
# rechnet Qdrant serverseitig (`Modifier.IDF`).

# %%
text = "Bei Fehlercode E-101 (Spindelübertemperatur) Programm anhalten und Spindel 20 Minuten abkühlen lassen. E-101 tritt bei der AX-200 auf."
print(tokenize(text))
index.sparse.explain(text)[:8]

# %% [markdown]
# ### A2 Dense vs. Sparse vs. Hybrid an einer Near-Miss-Frage

# %%
q = "Alle wie viele Betriebsstunden muss bei der AX-200 das Spindelöl gewechselt werden?"
for mode in ("dense", "sparse", "hybrid"):
    hits = index.search(q, mode=mode, k=5)
    print(f"{mode:7s}", [(h.doc_id.replace('prod-handbuch-', ''), round(h.score, 3)) for h in hits])

# %% [markdown]
# Überraschung: BM25 setzt hier das **AX-300**-Handbuch auf Platz 1 – weil dort steht „kürzere Intervalle gegenüber
# der AX-200“ und „Spindel“ mehrfach vorkommt. Exakte Terme helfen nur, wenn das *falsche* Dokument sie nicht auch
# enthält. Hybrid mischt beide Listen; ob AX-200 oder AX-300 oben landet, entscheidet hier die Fusion. Merkt euch
# die Frage – in Lab 3 löst sie der Reranker, in Lab 4 der Breadcrumb im Chunk.

# %% [markdown]
# ### A3 Reciprocal Rank Fusion – von Hand
#
# `score(d) = Σ 1 / (k + rank_i(d))` mit k = 60. Keine Normalisierung der Roh-Scores nötig – deshalb ist RRF
# der robuste Standard. Alternativen (DBSF, gewichtetes α) sind sinnvoll, wenn ein Retriever systematisch besser ist.

# %%
from ragkurs.index import rrf_fuse

dense_hits = index.search_dense(q, k=10)
sparse_hits = index.search_sparse(q, k=10)
fused = rrf_fuse(dense_hits, sparse_hits, k=60, top_k=5)
for h in fused:
    d_rank = next((x.rank for x in dense_hits if x.chunk.chunk_id == h.chunk.chunk_id), "-")
    s_rank = next((x.rank for x in sparse_hits if x.chunk.chunk_id == h.chunk.chunk_id), "-")
    print(f"#{h.rank} rrf={h.score:.4f}  dense-Rang={d_rank:>2}  sparse-Rang={s_rank:>2}  {h.chunk.chunk_id}")

# %% [markdown]
# ### A4 Messen statt glauben: drei Pipelines gegen das Golden Set

# %%
pipes = [
    RAGPipeline(index, PipelineConfig(retrieval="dense", k=5, name="dense")),
    RAGPipeline(index, PipelineConfig(retrieval="sparse", k=5, name="sparse/BM25")),
    RAGPipeline(index, PipelineConfig(retrieval="hybrid", k=5, prefetch_k=20, fusion="rrf", name="hybrid/RRF")),
]
compare_retrieval(pipes, golden, user_roles=["employee"])

# %% [markdown]
# Bei 42 Dokumenten ist die Hit-Rate@5 fast gesättigt – alle drei finden das richtige Dokument fast immer *irgendwo*
# in den Top-5. Der Unterschied steckt in **p@1** und **MRR**: Wo steht die richtige Quelle? Deshalb unten beide
# Kennzahlen je Fragetyp.

# %%
# Nach Fragetyp: Hit-Rate@5 und p@1 (richtige Quelle auf Platz 1)
frames = {p.config.label(): evaluate_retrieval(p, golden, ["employee"]) for p in pipes}
pd.concat({
    "hit@5": pd.DataFrame({name: df.groupby("type")["hit"].mean() for name, df in frames.items()}),
    "p@1":   pd.DataFrame({name: df.assign(p1=df["first_rank"] == 1).groupby("type")["p1"].mean() for name, df in frames.items()}),
}, axis=1).round(2)

# %% [markdown]
# ### A5 Query-Transformationen
#
# | Technik | Was passiert | Kosten |
# |---|---|---|
# | Rewrite | Frage → präzise Suchanfrage | 1 LLM-Call |
# | Multi-Query | n Varianten, Ergebnisse per RRF fusioniert | 1 LLM-Call + n Suchen |
# | Decomposition | Multi-Hop-Frage → Teilfragen | 1 LLM-Call + n Suchen |
# | HyDE | hypothetische Antwort wird eingebettet | 1 LLM-Call (mehr Tokens) |
# | Step-back | zusätzlich allgemeinere Frage | 1 LLM-Call + 2 Suchen |

# %%
from ragkurs import query as qt

frage = "Ich bin seit 4 Monaten bei Aurelia. Darf ich schon 3 Tage pro Woche mobil arbeiten und den vollen Jahresurlaub nehmen?"
print("Rewrite  :", qt.rewrite(frage))
print("Multi    :", qt.multi_query(frage, n=3))
print("Decompose:", qt.decompose(frage))
print("Step-back:", qt.step_back(frage))
print("HyDE     :", qt.hyde(frage)[:300], "...")

# %% [markdown]
# ## Teil B – Aufgaben
#
# ### B1 Fusion-Parameter: DBSF und prefetch_k
# Vergleicht `fusion="rrf"` mit `fusion="dbsf"` und `prefetch_k` ∈ {10, 20, 50}. Ändert sich die Hit-Rate?
# Was kostet ein größeres `prefetch_k` (Latenz, später Reranker-Kosten)?

# %%
# === LOESUNG START ===
# HINWEIS: compare_retrieval([...], golden, ["employee"]) mit mehreren PipelineConfig-Varianten
variants = [
    RAGPipeline(index, PipelineConfig(retrieval="hybrid", k=5, prefetch_k=pk, fusion=f, name=f"hybrid/{f}/pf{pk}"))  # ? PipelineConfig mit fusion=f und prefetch_k=pk
    for f in ("rrf", "dbsf") for pk in (10, 20, 50)
]
compare_retrieval(variants, golden, user_roles=["employee"])
# === LOESUNG ENDE ===

# %% [markdown]
# ### B2 Query-Transformationen messen
# Baut Hybrid-Pipelines mit `query_transform` ∈ {None, "rewrite", "multi", "decompose", "hyde"} und vergleicht
# Hit-Rate **und** Latenz. Achtet besonders auf die Typen `multi-hop` und `near-miss`.
# (Jede Transformation kostet LLM-Calls – 64 Fragen × 5 Varianten sind ok, bei größeren Sets sampeln.)

# %%
# === LOESUNG START ===
# HINWEIS: PipelineConfig(retrieval="hybrid", query_transform="multi", ...) ; nach Typ: df.groupby("type")["hit"].mean()
qt_pipes = [
    RAGPipeline(index, PipelineConfig(retrieval="hybrid", k=5, prefetch_k=20, query_transform=t, name=f"hybrid+{t or 'none'}"))  # ? PipelineConfig mit query_transform=t
    for t in (None, "rewrite", "multi", "decompose", "hyde")
]
summary = compare_retrieval(qt_pipes, golden, user_roles=["employee"])
display(summary)
by_type = pd.DataFrame({p.config.label(): evaluate_retrieval(p, golden, ["employee"]).groupby("type")["hit"].mean() for p in qt_pipes}).round(2)
by_type
# === LOESUNG ENDE ===

# %% [markdown]
# ### B3 Metadaten-Filter schlägt Embedding-Tuning
# Die Frage „Wie hoch ist der maximale Hotelpreis laut *aktueller* Reisekostenrichtlinie?“ hat eine Stale-Data-Falle.
# Vergleicht `status_filter="current"` (Standard) mit `status_filter=None`. Was passiert bei Sparse-Suche?
# Diskutiert: Welche Metadaten müsstet ihr in *eurem* Korpus pflegen, damit solche Filter möglich sind?

# %%
# === LOESUNG START ===
# HINWEIS: PipelineConfig(status_filter=None) vs. Standard; retrieve_only(...).retrieved_doc_ids
q18 = next(g for g in golden if g["id"] == "g18")["question"]
for sf in ("current", None):
    p = RAGPipeline(index, PipelineConfig(retrieval="sparse", k=3, status_filter=sf))  # ? Pipeline mit status_filter=sf
    print(f"status_filter={sf!s:8s} ->", p.retrieve_only(q18, ["employee"]).retrieved_doc_ids)
# === LOESUNG ENDE ===

# %% [markdown]
# ### B4 (Bonus) Contextual Retrieval
# Anthropic (2024): Ein LLM schreibt pro Chunk einen Kontextsatz („Dieser Abschnitt stammt aus der
# Reisekostenrichtlinie 2026 und regelt Hotelkosten …“), der vor den Chunk gestellt und mit-eingebettet wird.
# Kostet einen LLM-Call pro Chunk (hier ~150) – probiert es für die HR-Dokumente aus und messt den Effekt.

# %%
# === LOESUNG START ===
# HINWEIS: from ragkurs.chunking import add_contextual_prefix ; nur Chunks mit department == "HR" anreichern
from ragkurs.chunking import add_contextual_prefix
hr_chunks = [c for c in chunks if c.metadata.get("department") == "HR"]
other = [c for c in chunks if c.metadata.get("department") != "HR"]
ctx_chunks = add_contextual_prefix(hr_chunks, docs) + other  # ? HR-Chunks anreichern, Rest unveraendert anhaengen
ctx_index = HybridIndex(collection="lab2ctx").build(ctx_chunks)  # ? neuen Index bauen
hr_golden = [g for g in golden if any(s.startswith("hr-") for s in g["source_docs"])]
compare_retrieval(
    [RAGPipeline(index, PipelineConfig(retrieval="hybrid", name="hybrid")),
     RAGPipeline(ctx_index, PipelineConfig(retrieval="hybrid", name="hybrid+contextual"))],
    hr_golden, ["employee"],
)
# === LOESUNG ENDE ===

# %% [markdown]
# ## Teil C – Debrief
#
# 1. Welche Technik hat auf *diesem* Korpus am meisten gebracht? Hättet ihr das vorher gewettet? Und welche Kennzahl
#    hat den Unterschied überhaupt sichtbar gemacht – Hit@5 oder p@1/MRR?
# 2. Multi-Query kostet 3–4× Retrieval und einen LLM-Call. Für welche Fragetypen lohnt sich das – und wie würdet ihr
#    das in Produktion entscheiden (Routing)?
# 3. ARAGOG (2024) fand: HyDE + Rerank helfen, Multi-Query oft nicht. Deckt sich das mit euren Zahlen?
#
# **Merksatz:** Hybrid + Metadaten-Filter ist die Grundausstattung. Query-Transformationen sind Werkzeuge für
# bestimmte Fragetypen, kein Standard für alle Anfragen.
