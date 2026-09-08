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
- A2: Spindelöl-Frage in drei Modi. Messwert – **andersherum als intuitiv**: Dense AX-200 (0,75) vorn, Sparse setzt
  **AX-300** (12,3) vorn, weil im AX-300-Handbuch „gegenüber der AX-200“ und „Spindel“ mehrfach stehen; Hybrid AX-300
  vor AX-200. Sagen: „Exakte Terme helfen nur, wenn das falsche Dokument sie nicht auch enthält. Lab 3 (Reranker) und
  Lab 4 (Breadcrumb) lösen genau das.“
- A3: RRF von Hand – Tabelle mit Dense-Rang / Sparse-Rang / RRF-Score. „Ein Chunk, der in beiden Listen vorne ist,
  gewinnt.“
- A4: `compare_retrieval` – drei Zeilen. Messwert (n=60): Hit@5 alle 0,984; Recall 0,939 / 0,958 / 0,964;
  MRR 0,946 / 0,884 / 0,946; **p@1 0,917 / 0,800 / 0,917**. Lesart: BM25 allein rankt schlecht, Hybrid = Dense im
  Ranking, aber mehr Recall (Multi-Hop). Danach die Aufschlüsselung nach Typ zeigen (Hit@5 und p@1).
- A5: Query-Transformationen live: einmal `rewrite`, `multi_query`, `decompose` auf die 4-Monate-Frage. Kommentieren,
  was das LLM daraus macht.

## Lab 2 (35 Min)
- B1: RRF vs. DBSF, prefetch_k 10/20/50. Messwert: DBSF +1 Frage (Hit 1,0, MRR 0,952); prefetch_k 10 = 20, bei 50
  leicht schlechter (0,937). Kostet später Reranker-Zeit – linear.
- B2: Query-Transformationen messen (5 Pipelines × 60 Fragen, ~10 Min LLM-Zeit – das ist der längste Teil von Lab 2,
  parallel besprechen). Messwert: rewrite / multi / hyde je **+1 Frage** (Hit 1,0; p@1 0,917 → 0,933 bei multi/hyde)
  für 135 / 582 / 199 ms statt 8 ms; **decompose −1** (0,966; Teilfragen holen Nachbar-Dokumente). Fazit: Werkzeuge
  für Fragetypen, kein Standard.
- B3: Status-Filter: ohne Filter taucht `hr-reisekosten-2024` auf. 30 Sekunden Code, große Diskussion:
  „Wer pflegt bei euch `status`?“
- B4 (Bonus): Contextual Retrieval auf HR-Chunks (~40 LLM-Calls). Messwert: **kein Effekt** (0,963 → 0,963, n=27) –
  die Breadcrumbs tragen den Kontext schon. Sagen: Anthropics −49 % kamen von Chunks *ohne* Struktur.
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
