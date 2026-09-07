# %% [markdown]
# # Lab 0 – Umgebungs-Check und Baseline-Pipeline
#
# **Ziel (15 Min):** Alle haben eine laufende Umgebung, kennen den Korpus und haben eine *naive* RAG-Pipeline
# gesehen, die wir in den folgenden Labs Schritt für Schritt verbessern.
#
# **Ablauf**
# 1. Teil A – Walkthrough (Trainer führt vor, alle führen mit aus)
# 2. Teil B – Aufgaben
# 3. Teil C – Debrief-Fragen
#
# > Alle Labs nutzen das Paket `ragkurs` im Repo. Der Code ist bewusst kurz und lesbar – bei Fragen einfach
# > `??` an eine Funktion hängen, z. B. `chunk_fixed??`.

# %% [markdown]
# ## Teil A – Walkthrough
#
# ### A1 Umgebung prüfen

# %%
import sys, os
_root = os.getcwd()                      # Repo-Root finden (Notebook liegt in labs/ oder labs/solutions/)
while not os.path.isdir(os.path.join(_root, "ragkurs")):
    _root = os.path.dirname(_root)
sys.path.insert(0, _root); os.chdir(_root)  # Repo-Root in den Pfad (Notebook liegt in labs/)

from ragkurs import settings
print("Chat-Modell :", settings.chat_model)
print("Judge-Modell:", settings.judge_model)
print("Embeddings  :", settings.embed_model, f"({settings.embed_dim} Dim.)")
print("API-Key     :", "vorhanden" if settings.openai_api_key else "FEHLT -> .env prüfen")
print("Offline-Modus (FAKE_EMBEDDINGS):", settings.fake_embeddings)

# %% [markdown]
# ### A2 Der Korpus: „Aurelia Maschinenbau GmbH“
#
# 25 Dokumente eines fiktiven Maschinenbauers: HR-Richtlinien, IT-Richtlinien, Produkthandbücher, Preisliste,
# Compliance – teils Markdown, teils **nur als PDF** (mit Tabellen). Dazu zwei Gesetzesauszüge (BUrlG, ArbZG).
#
# Eingebaute Fallen, die produktive Systeme genauso haben:
# - **Near-Misses**: AX-200 vs. AX-300 (ähnliche Handbücher, andere Zahlen)
# - **Stale Data**: Reisekostenrichtlinie 2024 (ersetzt) neben 2026 (gültig)
# - **Zugriffsrechte**: Gehaltsbänder (nur HR/Management), Rabattrichtlinie (nur Vertrieb)
# - **Multi-Hop**: Antworten, die zwei Dokumente brauchen (Richtlinie + Gesetz)
# - **Tabellen**: Fristen, Preise, Fehlercodes stehen in Tabellen – in PDFs geht die Struktur beim naiven Parsen verloren

# %%
from ragkurs import load_corpus
from ragkurs.loading import corpus_overview

docs = load_corpus()            # Standard: PDFs mit pypdf (schnell, aber Tabellen zerfallen)
corpus_overview(docs)

# %%
# So sieht ein PDF nach pypdf aus - Tabellenzellen stehen untereinander, die Zuordnung ist weg:
pdf_doc = next(d for d in docs if d.doc_id == "hr-reisekosten-2026")
print(pdf_doc.text[600:1300])

# %% [markdown]
# ### A3 Das Golden Set
#
# 50 Fragen mit Referenzantwort, Quelldokument(en) und Typ. Damit messen wir in jedem Lab, ob eine Änderung
# wirklich hilft – statt „sieht gut aus“ (LGTM@few).

# %%
from ragkurs.eval import load_golden
import pandas as pd

golden = load_golden()
pd.DataFrame(golden)["type"].value_counts()

# %%
pd.DataFrame(golden)[["id", "type", "question", "source_docs"]].head(8)

# %% [markdown]
# ### A4 Die naive Baseline
#
# Fixed-Size-Chunks (800 Zeichen) → Dense-Embeddings → Top-5 → Prompt → Antwort. Genau das, was die meisten
# Teams als ersten Prototyp bauen.

# %%
from ragkurs import chunk_fixed, HybridIndex, RAGPipeline, PipelineConfig

chunks = chunk_fixed(docs, chunk_size=800, chunk_overlap=100)
print(len(chunks), "Chunks")
print(chunks[10])
print(chunks[10].text[:300])

# %%
index = HybridIndex(collection="baseline").build(chunks)   # Qdrant im Prozess, kein Server
index.stats

# %%
baseline = RAGPipeline(index, PipelineConfig(retrieval="dense", k=5, name="baseline"))
result = baseline.run("Wie viele Urlaubstage haben Vollzeitmitarbeitende pro Jahr?", user_roles=["employee"])
result.show()

# %% [markdown]
# `result.trace` enthält alles, was wir später fürs Debugging brauchen: Kandidaten mit Scores, Zeiten pro Stufe,
# Tokens, Kosten.

# %%
{k: v for k, v in result.trace.items() if k != "candidates"}

# %% [markdown]
# ## Teil B – Aufgaben
#
# ### B1 Drei Fragen selbst stellen
# Stellt der Baseline drei Fragen aus dem Golden Set – eine `faktisch`, eine `near-miss`, eine `negativ` –
# und notiert: Ist die Antwort korrekt? Stand die richtige Quelle im Kontext?

# %%
# === LOESUNG START ===
# HINWEIS: golden ist eine Liste von Dicts; result.retrieved_doc_ids zeigt die Dokumente im Kontext.
for typ in ("faktisch", "near-miss", "negativ"):
    item = next(g for g in golden if g["type"] == typ)
    r = baseline.run(item["question"], user_roles=["employee"])
    print(f"\n=== {typ} | {item['id']} ===")
    print("Frage    :", item["question"])
    print("Antwort  :", r.answer.text)
    print("Referenz :", item["ground_truth"])
    print("Quellen im Kontext:", r.retrieved_doc_ids, "| erwartet:", item["source_docs"])
# === LOESUNG ENDE ===

# %% [markdown]
# ### B2 Retrieval-Trefferquote der Baseline messen
# Nutzt `evaluate_retrieval` (kein LLM-Call, nur Retrieval) und `retrieval_summary`. Wie hoch ist die Hit-Rate@5?

# %%
from ragkurs.eval import evaluate_retrieval, retrieval_summary

# === LOESUNG START ===
# HINWEIS: evaluate_retrieval(pipeline, golden, user_roles) -> DataFrame; retrieval_summary(df) -> Kennzahlen
df_base = evaluate_retrieval(baseline, golden, user_roles=["employee"])
retrieval_summary(df_base)
# === LOESUNG ENDE ===

# %%
# Welche Fragetypen scheitern am häufigsten?
df_base.groupby("type")["hit"].mean().sort_values()

# %% [markdown]
# ## Teil C – Debrief
#
# 1. Welche Fragetypen findet die Baseline schlecht – und warum vermutlich?
# 2. Was fehlt dieser Pipeline, um produktionsreif zu sein? (Sammelt Stichworte – das ist die Agenda der zwei Tage.)
#
# **Merksatz:** Ohne Golden Set ist jede Verbesserung eine Vermutung.
