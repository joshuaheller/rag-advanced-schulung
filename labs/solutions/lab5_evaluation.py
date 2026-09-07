# %% [markdown]
# # Lab 5 – Evaluation & Testing von RAG-Systemen
#
# **Ziel (45 Min):** Ein Golden Set erweitern (manuell + synthetisch), Ragas-Metriken (Context Precision/Recall, Faithfulness,
# Answer Relevancy) auf zwei Pipelines vergleichen,
# einen LLM-Judge kalibrieren und die Evaluation als **Regressionstest in CI/CD** verankern.
#
# **Warum klassische Metriken nicht reichen:** BLEU/ROUGE messen Wortüberlappung – „30 Tage“ und „30 Arbeitstage
# Erholungsurlaub“ sind für sie verschieden, „30 Tage“ und „20 Tage“ fast gleich. Hit@k misst nur das Retrieval.
# Wir brauchen Metriken auf drei Kanten des **RAG-Dreiecks**:
#
# | Kante | Frage | Ragas-Metrik |
# |---|---|---|
# | Kontext ↔ Frage | Ist das Gefundene relevant? Ist alles Nötige da? | Context Precision, Context Recall |
# | Antwort ↔ Kontext | Ist die Antwort durch den Kontext gedeckt? | Faithfulness |
# | Antwort ↔ Frage | Beantwortet sie die Frage? | Answer Relevancy |

# %%
import sys, os, json, time, subprocess
_root = os.getcwd()                      # Repo-Root finden (Notebook liegt in labs/ oder labs/solutions/)
while not os.path.isdir(os.path.join(_root, "ragkurs")):
    _root = os.path.dirname(_root)
sys.path.insert(0, _root); os.chdir(_root)
import pandas as pd
pd.set_option("display.max_colwidth", 90)

from ragkurs import load_corpus, chunk_fixed, chunk_by_headings, HybridIndex, RAGPipeline, PipelineConfig
from ragkurs.eval import load_golden, run_golden, e2e_summary, failure_breakdown, judge_correctness
from ragkurs.metrics import RagMetrics, metrics_summary

docs = load_corpus(pdf_parser="docling")
golden = load_golden()

baseline = RAGPipeline(HybridIndex(collection="base").build(chunk_fixed(load_corpus(), 800, 100)), PipelineConfig(retrieval="dense", k=5, name="baseline"))
best = RAGPipeline(HybridIndex(collection="best").build(chunk_by_headings(docs)), PipelineConfig(retrieval="hybrid", k=5, prefetch_k=20, rerank="fast", name="hybrid+rerank"))

# %% [markdown]
# ## Teil A – Walkthrough
#
# ### A1 End-to-End-Lauf beider Pipelines (unser eigener Judge)

# %%
subset = golden[:15]
df_base = run_golden(baseline, subset, ["employee"], verbose=False)
df_best = run_golden(best, subset, ["employee"], verbose=False)
pd.DataFrame([e2e_summary(df_base), e2e_summary(df_best)]).set_index("config")

# %% [markdown]
# ### A2 Ragas-Metriken auf denselben Läufen
#
# `RagMetrics` implementiert die vier Ragas-Metriken nach den Original-Definitionen (`ragkurs/metrics.py`, ~100 Zeilen –
# lest sie, dann wisst ihr genau, was der Judge pro Metrik tut). Die Kontexte kommen aus `pipeline.log`, es wird
# nichts neu generiert. Kosten: ~6 Judge-Calls pro Frage.
#
# > Das Original-Paket `ragas` pinnt ältere LangChain-/OpenAI-Versionen und kollidiert mit dem Kurs-Stack –
# > deshalb der Nachbau. Wer Ragas selbst nutzen will: `requirements-optional.txt` in einem eigenen venv.

# %%
rm = RagMetrics()
m_base = rm.evaluate(df_base, baseline, verbose=False)
m_best = rm.evaluate(df_best, best, verbose=False)
pd.DataFrame([metrics_summary(m_base), metrics_summary(m_best)]).set_index("config")

# %%
# Pro Frage - wo genau liegt der Unterschied?
m_best[["id", "type", "correct", "context_precision", "context_recall", "faithfulness", "answer_relevancy"]].round(2)

# %% [markdown]
# ### A3 Den Judge kalibrieren
#
# Ein LLM-Judge ist nur so gut wie seine Übereinstimmung mit Domänenexperten. Vorgehen (Hamel Husain):
# 1. Mensch labelt eine Stichprobe binär (korrekt / nicht korrekt)
# 2. Judge labelt dieselbe Stichprobe
# 3. Precision/Recall des Judges **gegen den Menschen** berechnen – nicht nur „Agreement“
# 4. Judge-Prompt anpassen, bis die Fehler des Judges akzeptabel sind

# %%
# Der Trainer hat 8 Antworten der Baseline manuell gelabelt (1 = korrekt). Vergleicht mit dem Judge:
human_labels = {row["id"]: None for _, row in df_base.head(8).iterrows()}   # -> im Lab gemeinsam füllen
df_base.head(8)[["id", "question", "answer", "ground_truth", "correct"]]

# %% [markdown]
# ### A4 Synthetische Testfragen erzeugen
#
# Aus zufälligen Chunks lässt das LLM Frage/Antwort-Paare erzeugen (Single-Hop) bzw. aus zwei Chunks Multi-Hop-Fragen.
# Ragas macht das Gleiche mit einem Knowledge Graph (`ragkurs.eval_synth.ragas_testset`, optional).
# Synthetische Fragen sind ein **Startpunkt**, kein Ersatz für echte Nutzerfragen – sie werden anschließend
# von Experten geprüft (`reviewed: false` → Review-Pflicht) und aussortiert.

# %%
from ragkurs.eval_synth import generate_synthetic_testset

synth = generate_synthetic_testset(docs[:8], n=6)     # wenige Dokumente, damit es schnell geht (~1-2 Min)
synth[["question", "ground_truth", "source_docs"]]

# %% [markdown]
# ### A5 Evaluation als Test: DeepEval + pytest + GitHub Actions
#
# `tests/test_rag_quality.py` enthält drei Test-Ebenen:
# 1. **Deterministisch, kostenlos:** Retrieval-Hit-Rate mit BM25 ≥ Schwelle (läuft ohne API-Key)
# 2. **Regression:** End-to-End-Korrektheit auf einem festen Golden-Subset ≥ Schwelle (LLM-Kosten, nur bei gesetztem Key)
# 3. **DeepEval-Metriken:** Faithfulness und Answer Relevancy pro Testfall mit `assert_test`
#
# `.github/workflows/rag-eval.yml` führt (1) bei jedem Push aus und (2)+(3) nur, wenn das Secret `OPENAI_API_KEY`
# gesetzt ist – Eval als Quality Gate pro Pull Request.

# %%
print(open("tests/test_rag_quality.py").read()[:2500])

# %%
# Ebene 1 lokal ausführen (ohne Kosten):
print(subprocess.run([sys.executable, "-m", "pytest", "tests", "-q", "-k", "retrieval", "--no-header"], capture_output=True, text=True, cwd=".").stdout[-1500:])

# %% [markdown]
# ## Teil B – Aufgaben
#
# ### B1 Golden Set erweitern
# Schreibt **fünf eigene Fragen** zum Korpus (mindestens eine unbeantwortbare, eine Multi-Hop) als JSONL nach
# `data/golden/golden_set_team.jsonl` – Format wie `golden_set.jsonl`. Die Liste unten ist eine Vorlage: **ersetzt die fünf
# Beispiele durch eure eigenen Fragen** (Antwort im Korpus nachschlagen!) und lasst `best` darauf laufen.

# %%
# === LOESUNG START ===
# HINWEIS: Felder: id, type, question, ground_truth, source_docs, answerable ; load_golden(path=...)
team = [
    {"id": "t01", "type": "faktisch", "question": "Wie viele Stunden beträgt die regelmäßige Wochenarbeitszeit?", "ground_truth": "38,5 Stunden für Vollzeitbeschäftigte.", "source_docs": ["hr-arbeitszeit"], "answerable": True},
    {"id": "t02", "type": "tabelle", "question": "Welche Reaktionszeit gilt telefonisch mit AureliaCare?", "ground_truth": "2 Stunden (ohne AureliaCare 8 Stunden).", "source_docs": ["prod-garantiebedingungen"], "answerable": True},
    {"id": "t03", "type": "multi-hop", "question": "Was kostet AureliaCare für eine AX-300 Plus pro Jahr?", "ground_truth": "2,5 % von 298.000 € = 7.450 € pro Jahr.", "source_docs": ["sales-preisliste-2026"], "answerable": True},
    {"id": "t04", "type": "negativ", "question": "Wie hoch ist der Zuschuss zur betrieblichen Altersvorsorge?", "ground_truth": "Dazu enthält die Wissensbasis keine Information.", "source_docs": [], "answerable": False},
    {"id": "t05", "type": "near-miss", "question": "Wie schwer ist die AX-300 Plus?", "ground_truth": "ca. 6.300 kg (AX-300: 5.800 kg).", "source_docs": ["prod-handbuch-ax300"], "answerable": True},
]
with open("data/golden/golden_set_team.jsonl", "w", encoding="utf-8") as f:
    for t in team:
        f.write(json.dumps(t, ensure_ascii=False) + "\n")
df_team = run_golden(best, load_golden(path="data/golden/golden_set_team.jsonl"), ["employee"])  # ? eigenes Set laden und mit best ausfuehren
df_team[["id", "type", "correct", "failure", "answer"]]
# === LOESUNG ENDE ===

# %% [markdown]
# ### B2 Judge kalibrieren
# Füllt `human_labels` (True/False) für die 8 Antworten aus A3 gemeinsam aus und berechnet Precision, Recall
# und Agreement des Judges gegenüber euren Labels. Wo irrt der Judge – zu streng oder zu großzügig?
# Passt danach den Judge-Prompt (`ragkurs/eval.py`, `_JUDGE_PROMPT`) in *einem* Punkt an und messt erneut.

# %%
# === LOESUNG START ===
# HINWEIS: TP = Judge True & Mensch True ; precision = TP / Judge-True ; recall = TP / Mensch-True
human_labels = {k: True for k in human_labels}         # <- im Lab durch echte Urteile ersetzen
sub = df_base.head(8).copy()
sub["human"] = sub["id"].map(human_labels)
tp = ((sub["correct"]) & (sub["human"])).sum()  # ? True Positives: Judge korrekt UND Mensch korrekt
precision = tp / max(1, sub["correct"].sum())  # ? TP / alle Judge-korrekt
recall = tp / max(1, sub["human"].sum())  # ? TP / alle Mensch-korrekt
agreement = (sub["correct"] == sub["human"]).mean()
print(f"Judge vs. Mensch: precision={precision:.2f} recall={recall:.2f} agreement={agreement:.2f}")
sub[sub["correct"] != sub["human"]][["id", "answer", "ground_truth", "judge_reason"]]
# === LOESUNG ENDE ===

# %% [markdown]
# ### B3 Regressions-Gate bauen
# Schreibt eine Funktion `regression_gate(df_candidate, df_reference, max_drop=0.05)`, die `False` liefert,
# wenn die Korrektheit um mehr als 5 Prozentpunkte fällt **oder** eine bisher korrekte Frage jetzt falsch ist.
# Testet sie mit `df_base` (Kandidat) gegen `df_best` (Referenz).

# %%
# === LOESUNG START ===
# HINWEIS: Beide DataFrames per "id" mergen; Regression = vorher correct, jetzt nicht
def regression_gate(df_candidate, df_reference, max_drop=0.05):
    m = df_reference[["id", "correct"]].merge(df_candidate[["id", "correct"]], on="id", suffixes=("_ref", "_cand"))  # ? beide DataFrames ueber id mergen, Suffixe _ref/_cand
    drop = m["correct_ref"].mean() - m["correct_cand"].mean()  # ? Differenz der Korrektheits-Mittelwerte
    regressed = m[(m["correct_ref"]) & (~m["correct_cand"])]["id"].tolist()  # ? IDs: vorher korrekt, jetzt nicht
    ok = drop <= max_drop and not regressed
    print(f"Korrektheit ref={m['correct_ref'].mean():.2f} cand={m['correct_cand'].mean():.2f} drop={drop:+.2f} | Regressionen: {regressed} -> {'PASS' if ok else 'FAIL'}")
    return ok

regression_gate(df_base, df_best)
# === LOESUNG ENDE ===

# %% [markdown]
# ### B4 CI-Strategie entwerfen (Markdown)
# Für euer Projekt: Welche Tests laufen bei jedem PR, welche nächtlich, welche vor Releases? Wie begrenzt ihr die
# Kosten (Sampling, Caching, kleines Modell als Judge)? Wer pflegt das Golden Set und wie kommen Produktionsfragen hinein?

# %% [markdown]
# **Unsere CI-Strategie:**
# - PR: ...
# - Nightly: ...
# - Release: ...

# %% [markdown]
# ## Teil C – Debrief
#
# 1. Faithfulness war hoch, unser Judge sagte trotzdem „falsch“ – wie geht das? (Faithfulness prüft nur Antwort↔Kontext,
#    nicht Antwort↔Wahrheit. Falscher Kontext → treue, aber falsche Antwort.)
# 2. Wie viele Golden-Fragen braucht ihr, damit ein 5-Prozentpunkte-Unterschied kein Rauschen ist? (Faustregel: ≥ 100 für 5 pp)
# 3. Was passiert mit eurem Golden Set, wenn sich die Dokumente ändern (Reisekosten 2027)?
#
# **Merksatz:** Eval ist kein Projekt-Meilenstein, sondern Infrastruktur – wie Unit-Tests.
