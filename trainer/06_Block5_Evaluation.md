# Block 5 – Evaluation & Testing (10:15–12:00, Lab 5: 45 Min)

## Kernbotschaft
Eval ist Infrastruktur, kein Meilenstein. Drei Ebenen: billige deterministische Tests bei jedem Commit, LLM-Judge-
Regression auf festem Golden-Subset pro PR, Online-Monitoring in Produktion. Und: Der Judge wird gegen Menschen
kalibriert, sonst misst man Rauschen.

## Konzepte in Erklär-Sprache

**Warum klassische Metriken nicht reichen (Folie 5.1):** BLEU/ROUGE zählen Wortüberlappung: „30 Arbeitstage
Erholungsurlaub“ vs. „30 Tage“ = schlecht, „30 Tage“ vs. „20 Tage“ = fast perfekt. Hit@k misst nur, ob die Quelle da
war. „LGTM@few“ (Hamel Husain): drei Beispiele anschauen und ‚sieht gut aus‘ sagen – das ist der Ist-Zustand vieler Teams.

**Drei Ebenen (Folie 5.2, Hamel):** 1) Unit-Tests/Assertions – deterministisch, kostenlos (Retrieval-Hit-Rate,
ACL-Leak, Superseded-Ausschluss: `tests/test_rag_quality.py` Ebene 1). 2) Modell- und Human-Eval – Judge auf Golden Set
(Ebene 2/3). 3) A/B-Tests in Produktion.

**RAG-Dreieck (Folie 5.5):** Frage – Kontext – Antwort, drei Kanten, vier Metriken:
- *Context Precision*: Anteil relevanter Chunks in den Top-k, rang-gewichtet („steht das Relevante vorne?“).
- *Context Recall*: Anteil der Referenz-Aussagen, die der Kontext abdeckt („ist alles Nötige da?“).
- *Faithfulness*: Anteil der Antwort-Aussagen, die aus dem Kontext folgen („erfindet das Modell?“).
- *Answer Relevancy*: Beantwortet die Antwort die Frage? (Rückgenerierte Fragen vs. Originalfrage, Cosine.)
**Wichtig:** Faithfulness hoch + Antwort falsch ist möglich (treu zum falschen Kontext). Korrektheit braucht die
Referenz (unser Judge).

**Ragas (Folie 5.3/5.4):** Definiert diese Metriken; `ragkurs/metrics.py` implementiert sie 1:1 nach (100 Zeilen),
weil das Ragas-Paket LangChain/OpenAI-Versionen pinnt, die mit dem Kurs-Stack kollidieren. Sag das offen: „In
Produktion nehmt ihr Ragas oder DeepEval – aber wenn ihr die 100 Zeilen gelesen habt, wisst ihr, was die tun.“

**LLM-as-a-Judge (Folie 5.6, Hamel):** binäre Labels, Begründung, gegen Domänenexperten kalibrieren, Precision/Recall
des Judges messen (nicht nur Agreement – bei 90 % korrekten Antworten hat ein Judge, der immer „korrekt“ sagt, 90 %
Agreement). Judge = stärkeres Modell als Generator; Prompt iterieren.

**Golden Dataset (Folie 5.7/5.8):** Quellen: echte Nutzerfragen (Logs!), Experten, synthetisch (Bootstrapping).
Fragetypen abdecken (faktisch, Tabelle, Multi-Hop, Near-Miss, unbeantwortbar, ACL). 50–200 Fragen. Versioniert wie
Code. Synthetische Fragen (`eval_synth.py`, Ragas TestsetGenerator) → immer `reviewed: false` bis ein Mensch drüber war.

**CI/CD (Folie 5.9/5.10):** `pytest` + DeepEval `assert_test` + GitHub Actions. Kosten begrenzen: festes Subset (10
Fragen), Sampling, Embedding-Cache, günstiger Generator, Judge nur wo nötig. Flaky-Tests: Schwellen statt exakter
Werte, `flaky=True`/Retries.

**Offline vs. Online (Folie 5.11):** Offline = Golden Set vor dem Release; Online = Stichproben aus Produktion mit
Judge + Nutzerfeedback. Beides speist das Golden Set (Block 7 Feedback-Loop).

## Walkthrough – Sprechtext-Stichpunkte
- A1: Baseline vs. hybrid+rerank auf 15 Fragen mit Judge (~2 Min). Tabelle: Korrektheit, Hit-Rate, Refusal, Kosten.
- A2: `RagMetrics` – während es läuft (~6 Judge-Calls/Frage, ~3 Min): `ragkurs/metrics.py` öffnen und die vier
  Funktionen zeigen. Dann pro-Frage-Tabelle: „Faithfulness 1.0, correct=False – hier: treu zum AX-300-Kontext.“
- A3: Judge kalibrieren – 8 Antworten gemeinsam anschauen, mündlich labeln, Labels in B2 eintragen.
- A4: Synthetische Fragen (~6 LLM-Calls). Zwei davon laut vorlesen: „Würdet ihr die ins Golden Set nehmen?“
- A5: `tests/test_rag_quality.py` zeigen (drei Ebenen), `pytest -k retrieval` live (3 Tests, < 5 s, kein Key nötig),
  dann die Workflow-YAML: „Ebene 1 bei jedem Push, Ebene 2/3 nur mit Secret.“

## Lab 5 (45 Min)
- B1: 5 eigene Fragen als JSONL, `best` darauf laufen lassen. Erwartung: 4/5 korrekt, die Multi-Hop-Rechenfrage
  (7.450 €) ist bewusst schwer.
- B2: Judge-Kalibrierung: Precision/Recall gegen die 8 Labels aus A3. Typisches Ergebnis: Judge zu streng bei
  „Zusatzinfo“ oder zu großzügig bei Zahlendrehern → Prompt-Anpassung, erneut messen.
- B3: `regression_gate` – 10 Zeilen. Erwartung: `df_base` gegen `df_best` → FAIL (Baseline ist schlechter).
- B4: CI-Strategie – 5 Min Markdown, jeder stellt kurz vor.
- Wo TN hängen: JSON-Format (ensure_ascii=False), `df.merge(...suffixes=...)`.

## Typische Fragen
- „Wie viele Fragen brauche ich?“ – Faustregel: 100 Fragen, um 5 Prozentpunkte Unterschied halbwegs sicher zu sehen;
  50 reichen für Cluster/Trends. Lieber 100 gute als 1.000 synthetische.
- „Judge = gleiches Modell wie Generator?“ – Vermeiden (Selbstbestätigung). Stärkeres oder anderes Modell; bei Kosten-
  druck: kleines Modell, aber kalibriert.
- „Ragas vs. DeepEval?“ – Ragas: Metriken-Bibliothek + Testset-Generator, Forschungsnähe. DeepEval: pytest-nativ,
  CI-Integration, mehr Metriken (G-Eval). Beide Apache-2.0. Wahl nach Workflow, nicht nach Metrikqualität.
- „Was, wenn sich die Dokumente ändern?“ – Golden Set versionieren mit dem Korpus; Referenzantworten mit
  `valid_from`; bei Reisekosten 2027 werden g18–g21 zu neuen Referenzen.

## Stolperfallen
- `RagMetrics.evaluate` auf 15 Fragen ≈ 90 Judge-Calls (~3 Min, ~0,20 USD). Nicht auf 64 Fragen im Lab.
- `pytest` im Notebook via subprocess: `cwd` ist das Repo-Root (Notebooks wechseln beim Start dorthin).
- DeepEval-Tests (Ebene 3) brauchen den Key und dauern ~30 s – im Lab nur zeigen, nicht laufen lassen, falls Zeit knapp.

## Überleitung zu Block 6
„Alles, was wir bisher gemessen haben, nimmt an, dass die Dokumente vertrauenswürdig sind und jeder alles sehen
darf. Beides stimmt in Unternehmen nicht.“
