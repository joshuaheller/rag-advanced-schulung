# %% [markdown]
# # Lab 1 – Advanced RAG Failure Analysis
#
# **Ziel (30 Min):** Fehler eines RAG-Systems *systematisch* finden und klassifizieren – statt einzelne
# Antworten anzustarren. Am Ende habt ihr eine Fehler-Landkarte der Baseline und wisst, welcher Hebel
# (Retrieval? Chunking? Generierung?) den größten Effekt verspricht.
#
# **Failure-Taxonomie, die wir verwenden**
#
# | Klasse | Bedeutung | Typischer Hebel |
# |---|---|---|
# | `retrieval:missing_evidence` | Quelle gar nicht in Top-k | Chunking, Hybrid Search, Query-Transformation |
# | `retrieval:semantic_near_miss` | *ähnliches*, aber falsches Dokument (AX-300 statt AX-200, 2024 statt 2026) | Hybrid (exakte Terme), Metadaten-Filter, Reranking |
# | `retrieval:partial_evidence` | Multi-Hop: nur ein Teil der Quellen gefunden | Decomposition, Multi-Query, größeres k + Rerank |
# | `generation:context_noise` | Quelle da, aber weit hinten / von Rauschen verdeckt | Reranking, Re-Ordering, weniger k |
# | `generation:wrong_answer` | Quelle vorne, Modell antwortet trotzdem falsch | Prompt, Context Packing, stärkeres Modell |
# | `generation:over_refusal` | Evidenz da, Modell verweigert | Prompt, Kontextformat |
# | `halluzination_statt_absage` | Frage unbeantwortbar, System antwortet trotzdem | Prompt, Score-Schwelle, Judge |

# %%
import sys, os
_root = os.getcwd()                      # Repo-Root finden (Notebook liegt in labs/ oder labs/solutions/)
while not os.path.isdir(os.path.join(_root, "ragkurs")):
    _root = os.path.dirname(_root)
sys.path.insert(0, _root); os.chdir(_root)
import pandas as pd
pd.set_option("display.max_colwidth", 80)

from ragkurs import load_corpus, chunk_fixed, HybridIndex, RAGPipeline, PipelineConfig
from ragkurs.eval import load_golden, run_golden, e2e_summary, failure_breakdown, judge_correctness

docs = load_corpus()
index = HybridIndex(collection="baseline").build(chunk_fixed(docs, 800, 100))
baseline = RAGPipeline(index, PipelineConfig(retrieval="dense", k=5, name="baseline"))
golden = load_golden()

# %% [markdown]
# ## Teil A – Walkthrough
#
# ### A1 Eine Antwort sezieren
#
# Der wichtigste Debugging-Schritt ist immer derselbe: **War die Evidenz im Kontext?**
# Wenn nein → Retrieval-Fehler. Wenn ja → Generierungsfehler. Alles andere kommt danach.

# %%
item = next(g for g in golden if g["id"] == "g33")   # Spindelöl AX-200 (Near-Miss-Falle)
r = baseline.run(item["question"], user_roles=["employee"])
r.show(n_chars=200)
print("\nErwartete Quelle(n):", item["source_docs"])

# %% [markdown]
# ### A2 LLM-as-a-Judge: korrekt oder nicht?
#
# Wir bewerten **binär** (korrekt / nicht korrekt) mit Begründung – binäre Labels lassen sich gegen menschliche
# Urteile kalibrieren, 1–10-Skalen nicht (Hamel Husain). Der Judge ist ein etwas stärkeres Modell als der Generator.

# %%
judge_correctness(item["question"], r.answer.text, item["ground_truth"])

# %% [markdown]
# ### A3 Der ganze Lauf: 20 Fragen, automatisch klassifiziert
#
# `run_golden` führt Retrieval + Generierung + Judge aus und schlägt per Heuristik eine Fehlerklasse vor.
# **Die Heuristik ist ein Vorschlag** – im Lab prüft ihr sie nach. (Kosten: ~2 LLM-Calls pro Frage.)

# %%
subset = golden[:20]
df = run_golden(baseline, subset, user_roles=["employee"])
e2e_summary(df)

# %%
failure_breakdown(df)

# %%
df[df["failure"] != "ok"][["id", "type", "failure", "question", "retrieved_docs", "judge_reason"]]

# %% [markdown]
# ## Teil B – Aufgaben
#
# ### B1 Heuristik nachprüfen
# Nehmt euch **drei** als fehlerhaft klassifizierte Fragen und prüft manuell:
# Stimmt die Klasse? Schaut euch dazu die Top-5-Kandidaten und die Antwort an (`baseline.run(...).show()`).
# Korrigiert die Spalte `failure` im DataFrame, wo nötig.

# %%
# === LOESUNG START ===
# HINWEIS: df.loc[df.id == "g12", "failure"] = "generation:wrong_answer"  # Beispiel für eine Korrektur
for fid in df[df["failure"] != "ok"]["id"].head(3):
    it = next(g for g in golden if g["id"] == fid)
    print("\n" + "=" * 100)
    print(fid, "| Heuristik:", df.loc[df.id == fid, "failure"].item())
    baseline.run(it["question"], user_roles=["employee"]).show(n_chars=150)
    print("Referenz:", it["ground_truth"])
# Beispielkorrektur (an eure Beobachtung anpassen):
# df.loc[df.id == "g12", "failure"] = "generation:wrong_answer"
# === LOESUNG ENDE ===

# %% [markdown]
# ### B2 Fehler clustern: Fehlerklasse × Fragetyp
# Erstellt eine Kreuztabelle **Fehlerklasse × Fragetyp** und beantwortet:
# Welche Kombination ist am häufigsten? Welche *eine* Maßnahme würde die meisten Fehler auf einmal beheben?

# %%
# === LOESUNG START ===
# HINWEIS: pd.crosstab(df["failure"], df["type"]) oder failure_breakdown(df)
ct = pd.crosstab(df["failure"], df["type"], margins=True)
display(ct)
# Retrieval- vs. Generierungsfehler in Summe:
df["failure"].str.split(":").str[0].value_counts()
# === LOESUNG ENDE ===

# %% [markdown]
# ### B3 Die Kosten der Fehler
# Nicht jeder Fehler ist gleich schlimm. Ordnet die Fehlerklassen nach *Schaden im Unternehmen*:
# Eine falsche Kündigungsfrist vs. eine verweigerte Antwort zur Kernarbeitszeit.
# Welche Klasse würdet ihr zuerst angehen – und welche Metrik überwacht ihr dafür in Produktion?
#
# *(Diskussion, kein Code – 3 Stichpunkte ins Markdown darunter.)*

# %% [markdown]
# **Eure Priorisierung:**
# 1. ...
# 2. ...
# 3. ...

# %% [markdown]
# ### B4 (Bonus) Retrieval-Fehler isolieren
# Für alle Retrieval-Fehler: An welcher Position stand die richtige Quelle, wenn man **k = 20** statt 5 nimmt?
# Wenn sie bei k=20 dabei ist, hilft ein Reranker (Lab 3). Wenn nicht, muss das Retrieval selbst besser werden (Lab 2/4).

# %%
# === LOESUNG START ===
# HINWEIS: PipelineConfig(retrieval="dense", k=20) und retrieve_only(); first_rank per retrieval_row()
from ragkurs.eval import retrieval_row
wide = RAGPipeline(index, PipelineConfig(retrieval="dense", k=20))
rows = []
for fid in df[df["failure"].str.startswith("retrieval")]["id"]:
    it = next(g for g in golden if g["id"] == fid)
    rr = retrieval_row(wide.retrieve_only(it["question"], ["employee"]), it, 20)
    rows.append({"id": fid, "type": it["type"], "rank_bei_k20": rr["first_rank"], "recall_k20": rr["recall"]})
pd.DataFrame(rows)
# === LOESUNG ENDE ===

# %% [markdown]
# ## Teil C – Debrief
#
# 1. Wie groß ist der Anteil Retrieval- vs. Generierungsfehler? (Erfahrungswert in Produktion: 60–80 % Retrieval.)
# 2. Welche Fehler wären mit „mehr Kontext“ (größeres k) behoben – und was kostet das?
# 3. Welche drei Hebel nehmen wir uns für heute vor?
#
# **Merksatz:** Erst klassifizieren, dann optimieren. Wer ohne Fehlerklassen optimiert, dreht an zufälligen Schrauben.
