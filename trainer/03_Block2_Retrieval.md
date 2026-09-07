# Block 2 – Fortgeschrittene Retrieval-Strategien (13:00–14:30, Lab 2: 35 Min)

## Kernbotschaft
Dense (Bedeutung) + Sparse (exakte Terme) fusioniert per RRF ist 2026 die Grundausstattung. Query-Transformationen
sind Spezialwerkzeuge für bestimmte Fragetypen – und werden gemessen, nicht geglaubt.

## Konzepte in Erklär-Sprache

**Dense vs. Sparse (Folie 2.1):** Dense = „versteht, dass ‚Kündigungsfrist‘ und ‚wie lange muss ich bleiben‘ dasselbe
meinen“. Sparse/BM25 = „findet AX-200, E-101, § 4 exakt“. Tabelle mit Stärken/Schwächen: Synonyme (Dense +),
Produktcodes (Sparse +), Tippfehler (beide −), neue Fachbegriffe, die das Embedding-Modell nie sah (Sparse +).

**BM25 in 3 Minuten (Folie 2.2):** „Zähle, wie oft ein Suchwort im Chunk vorkommt – aber mit abnehmendem Ertrag
(10× ‚Urlaub‘ ist nicht 10× so relevant wie 1×) und normiert auf die Chunk-Länge. Seltene Wörter zählen mehr (IDF).“
Im Notebook A1: `index.sparse.explain(text)` zeigt Tokens und Gewichte. `ragkurs/sparse.py` ist 60 Zeilen – „das ist
die ganze Magie“. Qdrant rechnet die IDF serverseitig (`Modifier.IDF`).

**Hybrid-Architektur (Folie 2.3):** zwei Suchen parallel (je `prefetch_k` Kandidaten) → Fusion → Top-k. In Qdrant ein
einziger `query_points`-Aufruf mit `prefetch`.

**RRF (Folie 2.4):** `score = Σ 1/(60 + Rang)`. Warum k=60? Empirisch robust (Cormack 2009) – dämpft den Einfluss von
Platz 1 vs. 2. Vorteil: Roh-Scores (Cosine 0,3 vs. BM25 12,5) müssen nicht vergleichbar sein. **DBSF** (Distribution-
Based Score Fusion) normalisiert Scores per Mittelwert/Std-Abw. – sinnvoll, wenn ein Retriever systematisch besser ist.

**Contextual Retrieval (Folie 2.6):** Anthropic 2024: LLM schreibt pro Chunk einen Kontextsatz („Dieser Abschnitt der
Reisekostenrichtlinie 2026 regelt Hotelkosten“), der mit eingebettet wird. Ergebnis im Paper: −49 % Retrieval-Fehler,
−67 % mit Reranking. Kosten: ein LLM-Call pro Chunk beim Ingest (mit Prompt Caching günstig). Im Lab B4 als Bonus.

**Query-Transformationen (Folie 2.7):** Tabelle im Notebook. Merksatz: „Rewriting räumt auf, Multi-Query streut,
Decomposition zerlegt, HyDE rät die Antwort und sucht nach der.“ ARAGOG-Studie (2024): HyDE + LLM-Rerank helfen,
Multi-Query brachte dort nichts – „auf eurem Korpus kann es anders sein, deshalb messen“.

**Metadaten-Filter (Folie 2.11):** Der billigste Qualitätsgewinn: `status == current` schlägt jede Embedding-
Optimierung bei der Stale-Data-Falle. Voraussetzung: gepflegte Metadaten – „das ist eine Organisationsfrage, keine
technische“.

## Walkthrough – Sprechtext-Stichpunkte
- A1: `tokenize` zeigt Stoppwort-Entfernung und Mini-Stemming („betriebsstund“). Gewichte erklären.
- A2: Spindelöl-Frage in drei Modi. Erwartung mit echten Embeddings: Dense bringt AX-300 nach vorn, Sparse AX-200
  (exakter Term), Hybrid beides – „genau das, was wir wollen: der Reranker in Block 3 entscheidet dann“.
- A3: RRF von Hand – Tabelle mit Dense-Rang / Sparse-Rang / RRF-Score. „Ein Chunk, der in beiden Listen vorne ist,
  gewinnt.“
- A4: `compare_retrieval` – drei Zeilen. Erwartung: Hybrid ≥ beide Einzelnen, besonders bei `near-miss` und `tabelle`.
  Danach die Aufschlüsselung nach Typ zeigen.
- A5: Query-Transformationen live: einmal `rewrite`, `multi_query`, `decompose` auf die 4-Monate-Frage. Kommentieren,
  was das LLM daraus macht.

## Lab 2 (35 Min)
- B1: RRF vs. DBSF, prefetch_k 10/20/50. Erwartung: kleine Unterschiede; prefetch_k > 20 bringt kaum etwas, kostet
  aber später Reranker-Zeit.
- B2: Query-Transformationen messen (5 Pipelines × 48 Fragen, ~3–4 Min LLM-Zeit). Erwartung: `decompose` hilft bei
  multi-hop, `rewrite` neutral, `multi` gemischt, `hyde` bei near-miss riskant (erfindet Zahlen des falschen Modells).
- B3: Status-Filter: ohne Filter taucht `hr-reisekosten-2024` auf. 30 Sekunden Code, große Diskussion:
  „Wer pflegt bei euch `status`?“
- B4 (Bonus): Contextual Retrieval auf HR-Chunks (~40 LLM-Calls). Erwartung: MRR steigt leicht.
- Wo TN hängen: `compare_retrieval` erwartet eine Liste von Pipelines; `PipelineConfig(name=...)` für lesbare Labels.

## Typische Fragen
- „Brauche ich BM25, wenn ich ein gutes Embedding-Modell habe?“ – Ja, für Bezeichner, Codes, Eigennamen, neue
  Begriffe. Alle großen Anbieter (Weaviate, Qdrant, Elastic, Azure AI Search) liefern Hybrid als Standard.
- „Was ist mit SPLADE / gelernten Sparse-Modellen?“ – Mittelweg: Sparse-Vektoren mit gelernter Termexpansion.
  Gut, aber Modellabhängigkeit; BM25 ist der robuste Start.
- „Wie wähle ich alpha (Gewichtung Dense/Sparse)?“ – Mit dem Golden Set sweepen; RRF braucht kein alpha, deshalb Default.
- „Kostet Multi-Query nicht zu viel?“ – 1 LLM-Call + 3 Suchen; per Routing nur für vage/komplexe Fragen einsetzen
  (Klassifikator oder einfache Heuristik: Fragelänge, Konjunktionen).

## Stolperfallen
- Offline-Modus (FAKE_EMBEDDINGS) liefert bei Dense Unsinn – nur mit echtem Key zeigen.
- `hyde` erzeugt einen längeren Text → Embedding-Kosten minimal höher, Latenz +1 LLM-Call.

## Überleitung zu Block 3
„Hybrid holt die richtigen Kandidaten in die Top-20. Aber in den Prompt passen fünf. Wer sortiert? Der Reranker.“
