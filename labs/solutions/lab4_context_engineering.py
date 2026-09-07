# %% [markdown]
# # Lab 4 – Context Engineering & Knowledge Structuring
#
# **Ziel (Tag 1: 25 Min + Tag 2: 20 Min):** Verstehen, dass die Qualität eines RAG-Systems *vor* dem Embedding
# entschieden wird – beim Parsen, Chunken und Packen des Kontexts.
#
# **Teil 1 (Tag 1):** Parsing (pypdf vs. Docling), struktur-erhaltendes Chunking, Tabellen, Chunk-Größe
# **Teil 2 (Tag 2):** Parent-Child-Retrieval, Context Packing, Re-Ordering, Long-Context vs. RAG

# %%
import sys, os, time
_root = os.getcwd()                      # Repo-Root finden (Notebook liegt in labs/ oder labs/solutions/)
while not os.path.isdir(os.path.join(_root, "ragkurs")):
    _root = os.path.dirname(_root)
sys.path.insert(0, _root); os.chdir(_root)
import pandas as pd
pd.set_option("display.max_colwidth", 90)

from ragkurs import load_corpus, chunk_fixed, chunk_by_headings, chunk_parent_child, HybridIndex, RAGPipeline, PipelineConfig
from ragkurs.eval import load_golden, evaluate_retrieval, retrieval_summary, compare_retrieval, run_golden, e2e_summary

golden = load_golden()
tabellen = [g for g in golden if g["type"] == "tabelle"]

# %% [markdown]
# ## Teil A – Walkthrough (Tag 1)
#
# ### A1 Parsing entscheidet: pypdf vs. Docling
#
# Docling (IBM, MIT-Lizenz) erkennt Layout, Überschriften und Tabellen und liefert Markdown. Der erste Aufruf
# lädt die Modelle (~500 MB) – auf vorbereiteten VMs sind sie im Cache.

# %%
from ragkurs.loading import load_document
from pathlib import Path

pdf = Path("data/corpus/prod-handbuch-ax200.pdf")
naive = load_document(pdf, pdf_parser="pypdf")
t0 = time.perf_counter(); structured = load_document(pdf, pdf_parser="docling"); print(f"Docling: {time.perf_counter()-t0:.1f}s")

print("=== pypdf ===");   print(naive.text[900:1500])
print("\n=== Docling ==="); print(structured.text[900:1900])

# %% [markdown]
# ### A2 Der Korpus mit Docling – und was das mit dem Chunking macht

# %%
docs_pypdf = load_corpus(pdf_parser="pypdf")
docs_docling = load_corpus(pdf_parser="docling")     # nur die 5 PDFs laufen durch Docling, Markdown bleibt Markdown

for name, docs in (("pypdf", docs_pypdf), ("docling", docs_docling)):
    ch = chunk_by_headings(docs)
    pdf_chunks = [c for c in ch if c.doc_id == "prod-handbuch-ax200"]
    print(f"{name:8s}: {len(ch)} Chunks gesamt, AX-200-Handbuch: {len(pdf_chunks)} Chunks, Abschnitte: {[c.section.split(' > ')[-1][:25] for c in pdf_chunks]}")

# %%
# Ein Tabellen-Chunk bleibt als Ganzes erhalten - mit Breadcrumb:
wart = next(c for c in chunk_by_headings(docs_docling) if c.doc_id == "prod-handbuch-ax200" and "Wartung" in c.section)
print(wart.text[:900])

# %% [markdown]
# ### A3 Der Effekt auf Tabellenfragen (nur Retrieval, ohne LLM)
#
# Achtung, Aha-Moment in die andere Richtung: Auf **Dokument-Ebene** (Hit-Rate = richtiges Dokument in den Top-5)
# findet auch pypdf die Tabellendokumente – die Wörter stehen ja noch drin, nur die Zuordnung Zelle↔Spalte ist weg.
# Der Unterschied zeigt sich in **p@1/MRR** (welcher Chunk steht oben), in der Lesbarkeit des Kontexts und – bei
# komplexeren Tabellen als unseren – in der Antwort selbst. Vergleicht die Kontexte aus A1 noch einmal.

# %%
variants = {
    "pypdf+fixed":      chunk_fixed(docs_pypdf, 800, 100),
    "pypdf+headings":   chunk_by_headings(docs_pypdf),
    "docling+headings": chunk_by_headings(docs_docling),
}
pipes = [RAGPipeline(HybridIndex(collection=n.replace("+", "_")).build(ch), PipelineConfig(retrieval="hybrid", k=5, name=n)) for n, ch in variants.items()]
compare_retrieval(pipes, tabellen, user_roles=["employee"])

# %% [markdown]
# ## Teil B – Aufgaben (Tag 1)
#
# ### B1 Chunk-Größe sweepen
# Variiert `max_chars` in `chunk_by_headings` (400, 800, 1500, 3000) auf den Docling-Dokumenten.
# Tragt Hit-Rate, Anzahl Chunks und durchschnittliche Kontextlänge (Zeichen der Top-5) auf.
# Wo liegt der Sweet Spot – und warum ist „größer“ nicht automatisch besser?

# %%
# === LOESUNG START ===
# HINWEIS: Kontextlänge = sum(len(h.chunk.text) for h in pipe.retrieve_only(q).hits)
rows = []
for mc in (400, 800, 1500, 3000):
    ch = chunk_by_headings(docs_docling, max_chars=mc)  # ? Chunking mit max_chars=mc
    p = RAGPipeline(HybridIndex(collection=f"mc{mc}").build(ch), PipelineConfig(retrieval="hybrid", k=5))
    s = retrieval_summary(evaluate_retrieval(p, golden, ["employee"]))
    ctx = [sum(len(h.chunk.text) for h in p.retrieve_only(g["question"], ["employee"]).hits) for g in golden[:15]]  # ? Zeichen der Top-5-Chunks je Frage summieren
    rows.append({"max_chars": mc, "chunks": len(ch), "hit_rate": s["hit_rate"], "mrr": s["mrr"], "avg_kontext_zeichen": round(sum(ctx) / len(ctx))})
pd.DataFrame(rows)
# === LOESUNG ENDE ===

# %% [markdown]
# ### B2 End-to-End auf Tabellenfragen
# Lasst `run_golden` (mit Judge) auf den 17 Tabellenfragen laufen – einmal `pypdf+fixed`, einmal `docling+headings`.
# Wie ändert sich die **Antwort-Korrektheit** (nicht nur die Hit-Rate)? Wenn beide gleichauf liegen: Lest euch zwei
# Antworten samt Kontext an – ein aktuelles Modell rekonstruiert zweispaltige Tabellen aus dem pypdf-Zeilensalat oft
# korrekt. Was passiert bei einer Tabelle mit fünf Spalten, bei Fußnoten, bei verbundenen Zellen?

# %%
# === LOESUNG START ===
# HINWEIS: run_golden(pipe, tabellen, ["employee"], verbose=False) -> e2e_summary(df)
res = {}
for p in (pipes[0], pipes[2]):
    df = run_golden(p, tabellen, ["employee"], verbose=False)  # ? End-to-End-Lauf mit Judge auf den Tabellenfragen
    res[p.config.label()] = e2e_summary(df)
pd.DataFrame(res).T
# === LOESUNG ENDE ===

# %% [markdown]
# ### B3 Breadcrumb-Ablation
# `chunk_by_headings(docs, breadcrumb=False)` entfernt die `[Dokument > Abschnitt]`-Zeile. Messt den Effekt auf
# `near-miss`-Fragen. Warum hilft der Breadcrumb gerade dort?

# %%
# === LOESUNG START ===
# HINWEIS: near_miss = [g for g in golden if g["type"] == "near-miss"]  # ? nur Fragen vom Typ near-miss
near_miss = [g for g in golden if g["type"] == "near-miss"]
compare_retrieval(
    [RAGPipeline(HybridIndex(collection="bc1").build(chunk_by_headings(docs_docling, breadcrumb=True)), PipelineConfig(retrieval="hybrid", name="mit breadcrumb")),
     RAGPipeline(HybridIndex(collection="bc0").build(chunk_by_headings(docs_docling, breadcrumb=False)), PipelineConfig(retrieval="hybrid", name="ohne breadcrumb"))],
    near_miss, ["employee"],
)
# === LOESUNG ENDE ===

# %% [markdown]
# ---
# ## Teil A – Walkthrough (Tag 2)
#
# ### A4 Parent-Child: klein suchen, groß liefern
#
# Kleine Chunks (~350 Zeichen) treffen präziser, große Chunks (ganze Abschnitte) geben dem LLM den nötigen Kontext.
# Also: Kinder indexieren, beim Antworten den Elternabschnitt liefern.

# %%
children, parents = chunk_parent_child(docs_docling, parent_chars=1500, child_chars=350)
print(len(children), "Kinder,", len(parents.parents), "Eltern")
pc_index = HybridIndex(collection="parent_child").build(children)

pc = RAGPipeline(pc_index, PipelineConfig(retrieval="hybrid", k=5, prefetch_k=20, parent_expand=True, name="parent-child"), parent_store=parents)
r = pc.retrieve_only("Wie hoch ist die Verpflegungspauschale bei 24 Stunden Abwesenheit?", ["employee"])
for h in r.hits[:3]:
    print(h.source, h.chunk.chunk_id, "| Kind:", h.extra.get("child_id"), "|", len(h.chunk.text), "Zeichen")

# %%
flat = RAGPipeline(HybridIndex(collection="flat").build(chunk_by_headings(docs_docling)), PipelineConfig(retrieval="hybrid", k=5, prefetch_k=20, name="headings-flat"))
compare_retrieval([flat, pc], golden, ["employee"])

# %% [markdown]
# ### A5 Context Packing und Re-Ordering
#
# `build_context` dedupliziert (Kinder desselben Elternteils), hält ein Zeichenbudget ein und kann die Reihenfolge
# ändern: **Lost in the Middle** (Liu et al. 2023) – Modelle nutzen Anfang und Ende des Kontexts besser als die Mitte.

# %%
from ragkurs.generate import build_context, lost_in_the_middle_reorder

hits = flat.retrieve_only("Welche Reaktionszeit gilt für einen Incident der Stufe S2?", ["employee"]).hits
print("Original    :", [h.rank for h in hits])
print("Re-Ordering :", [h.rank for h in lost_in_the_middle_reorder(hits)])
ctx, used = build_context(hits, max_chars=2500, reorder="lost_in_middle")
print(len(ctx), "Zeichen,", len(used), "Chunks verwendet")

# %% [markdown]
# ## Teil B – Aufgaben (Tag 2)
#
# ### B4 Re-Ordering messen
# Vergleicht end-to-end (Judge) `reorder="none"` vs. `"lost_in_middle"` bei **k = 10** auf 15 Fragen.
# Bei k=5 ist der Effekt meist klein – warum? Und wenn er auch bei k=10 nicht messbar ist: Was sagt das über
# aktuelle Modelle vs. die Modelle der Studie von 2023 – und über die Aussagekraft von 15 Fragen?

# %%
# === LOESUNG START ===
# HINWEIS: PipelineConfig(retrieval="hybrid", k=10, prefetch_k=25, reorder=..., max_context_chars=12000)
idx_flat = flat.index
out = {}
for ro in ("none", "lost_in_middle"):
    p = RAGPipeline(idx_flat, PipelineConfig(retrieval="hybrid", k=10, prefetch_k=25, reorder=ro, max_context_chars=12000, name=f"k10+{ro}"))  # ? k=10, reorder=ro, max_context_chars=12000
    out[p.config.label()] = e2e_summary(run_golden(p, golden[:15], ["employee"], verbose=False))
pd.DataFrame(out).T
# === LOESUNG ENDE ===

# %% [markdown]
# ### B5 Long-Context statt RAG?
# Der gesamte Korpus hat ~70.000 Zeichen (~19.000 Tokens) – das passt locker in ein modernes Kontextfenster.
# Baut eine „Alles-in-den-Kontext“-Variante: ein Hit pro Dokument, `max_chars` sehr groß, und vergleicht auf
# 10 Fragen **Korrektheit, Latenz und Kosten** mit der RAG-Pipeline. Wann kippt die Rechnung (Stichworte:
# Korpusgröße, Anfragen/Tag, Context Rot, Zugriffsrechte)?

# %%
# === LOESUNG START ===
# HINWEIS: from ragkurs.index import Hit ; from ragkurs.chunking import Chunk ; generate.answer(q, hits, max_chars=10**7)
from ragkurs.index import Hit
from ragkurs.chunking import Chunk
from ragkurs.generate import answer as gen_answer
from ragkurs.eval import judge_correctness
from ragkurs.pipeline import estimate_cost

all_hits = [Hit(Chunk(d.doc_id, d.doc_id, d.text, {"title": d.title, "version": d.metadata.get("version", "")}), 1.0, i + 1) for i, d in enumerate(docs_docling) if d.metadata.get("status") == "current"]
rows = []
for g in golden[:10]:
    t0 = time.perf_counter(); a = gen_answer(g["question"], all_hits, max_chars=10**7); ms = (time.perf_counter() - t0) * 1000  # ? gen_answer mit ALLEN Dokumenten als Hits und sehr grossem max_chars, Zeit messen
    j = judge_correctness(g["question"], a.text, g["ground_truth"])  # ? Judge auf die Antwort
    rows.append({"id": g["id"], "correct": j["correct"], "ms": round(ms), "input_tokens": a.usage.get("input_tokens"), "cost_usd": estimate_cost(a.usage)})
lc = pd.DataFrame(rows)
rag = run_golden(flat, golden[:10], ["employee"], verbose=False)
pd.DataFrame({
    "long-context": {"accuracy": lc["correct"].mean(), "avg_ms": lc["ms"].mean(), "avg_input_tokens": lc["input_tokens"].mean(), "cost_total": lc["cost_usd"].sum()},
    "rag (hybrid, k=5)": {"accuracy": rag["correct"].mean(), "avg_ms": rag["t_total_ms"].mean(), "avg_input_tokens": None, "cost_total": rag["cost_usd"].sum()},
}).T
# === LOESUNG ENDE ===

# %% [markdown]
# ## Teil C – Debrief
#
# 1. Welcher Schritt hat auf Tabellenfragen mehr gebracht – Docling oder das Chunking? Was folgt daraus für die
#    Reihenfolge, in der ihr ein System optimiert? (Und: Welcher Effekt war überhaupt messbar – Hit@5, p@1, Korrektheit?)
# 2. Parent-Child erhöht die Kontextlänge. Wo ist die Grenze, ab der Context Noise den Gewinn auffrisst?
# 3. Long-Context: Bei welcher Korpusgröße und Anfragezahl würdet ihr komplett auf Retrieval verzichten?
#    (Und was ist mit Zugriffsrechten?)
#
# **Merksatz:** Garbage in, garbage out gilt für RAG doppelt: Was das Parsing zerstört, findet kein Embedding wieder.
