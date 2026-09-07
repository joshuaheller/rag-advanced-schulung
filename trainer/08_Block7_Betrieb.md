# Block 7 – Betrieb, Monitoring & Skalierung (14:45–16:00, Lab 7: 30 Min)

## Kernbotschaft
Ein RAG-System ohne Traces ist eine Blackbox mit API-Rechnung. Vier Dinge machen es betreibbar: Traces pro Stufe,
Kennzahlen mit Alerts, ein Feedback-Loop ins Golden Set und ein Plan für den Re-Index (Modell-/Chunking-Wechsel).

## Konzepte in Erklär-Sprache

**Observability-Stack (Folie 7.1):** Traces (ein Trace = eine Anfrage, Spans = Stufen: Transform, Retrieval, Rerank,
Generation), Metriken (p50/p95, Kosten, Raten), Logs. Standard: OpenTelemetry mit GenAI Semantic Conventions
(`gen_ai.usage.input_tokens`, `gen_ai.request.model`); Tools: Phoenix (lokal, Open Source), Langfuse (MIT, self-host),
LangSmith (SaaS). „Was für Web-Services APM ist, ist für RAG das Tracing – nur mit Tokens statt SQL-Zeiten.“

**Was loggen (Folie 7.2):** siehe `result.trace`: Query/Hash, Rewrites, Kandidaten mit Scores, Reranker-Scores,
Kontextlänge, Tokens, Zeiten je Stufe, Modell, Cache-Status, Rollen. Prompt-Hash statt Prompt (Datenschutz + Volumen).

**Dashboards & Alerts (Folie 7.3):** Team-Dashboard: p95 je Stufe, Kosten/Tag, Fehlerrate, Cache-Hit-Rate.
Fachbereichs-Dashboard: No-Answer-Rate, Nutzerfeedback, Top-Themen, Wissenslücken (Häufung niedriger Top-Scores).
Alerts: No-Answer-Rate-Sprung (Index kaputt? neue Themen?), p95-Anstieg (Reranker? API?), Kosten-Anomalie.

**Feedback-Loops (Folie 7.4/7.5):** Explizit (Daumen) und implizit (Nachfrage, Abbruch, Kopieren der Antwort).
Daumen runter → Kandidat fürs Golden Set (`candidates.jsonl`, `reviewed: false`) → Review → Regressionstest.
Online-Eval: Stichprobe der Produktionsantworten durch den Judge bewerten (Sampling 1–5 %), Trend beobachten.

**Kostenmodell (Folie 7.6):** Kosten/Anfrage = Input-Tokens (Kontext × k) + Output + Embedding + Reranker-CPU.
Hebel: k, Chunk-Größe, Modellwahl, Caching. Lab A4 rechnet drei Konfigurationen für 10k Anfragen/Tag.

**Caching (Folie 7.7):** Drei Ebenen: (1) Prompt Caching der Anbieter (gleiches Präfix → 50–90 % Rabatt auf Input;
System-Prompt + statischer Teil vorne!), (2) Semantic Cache (ähnliche Frage → gleiche Antwort; Schwelle 0,9+,
**Rollen im Key**, TTL, Invalidierung bei Re-Index), (3) Embedding-Cache (wie `.cache/` im Repo).

**Performance (Folie 7.8):** Qdrant: Quantisierung (int8/binary) für Speicher und Speed, HNSW-Parameter (ef, m),
Async/Batching, Reranker nur bei Bedarf (Score-Gap-Heuristik), kleine Modelle vorne in der Kaskade.

**Index-Lifecycle & Drift (Folie 7.9/7.10):** Inkrementeller Ingest (Change-Detection per Hash), Blue/Green-Reindex
(neue Collection parallel, Dual-Read-Vergleich auf Golden Set, Alias umschalten, alte löschen). Embedding-Modellwechsel
= kompletter Re-Index + Cache-Invalidierung + Query-Embedding mit neuem Modell (Vektoren verschiedener Modelle sind
nicht vergleichbar). Daten-Drift: neue Themen in Anfragen → Wissenslücken → Golden Set erweitern.

**Referenzarchitektur (Folie 7.11):** Ingest-Pipeline (Parser, Gate, Chunker, Embedder) → Index (Qdrant, versioniert)
→ Retrieval-Service (Hybrid, Filter, Rerank) → Gateway (Auth, Rollen, Rate-Limit, Cache) → LLM → Eval-Service
(Offline-Golden, Online-Sampling) → Observability (OTel → Phoenix/Langfuse).

## Walkthrough – Sprechtext-Stichpunkte
- A1: `start_phoenix()` → URL im Browser öffnen (Bildschirm teilen). 8 `traced_run`. In Phoenix einen Trace aufklappen:
  `rag.run` → `ChatOpenAI`-Span mit Tokens/Latenz. Attribute `rag.t_retrieval_ms`, `rag.retrieved_docs` zeigen.
- A2: Traffic simulieren (30 Anfragen inkl. Paraphrasen/Off-Topic, ~1–2 Min), `log_to_dataframe`, KPIs, drei Plots.
  Off-Topic-Fragen im Score-Histogramm zeigen: „Das linke Häufchen ist eure Wissenslücke oder Missbrauch.“
- A3: Semantic Cache: 6 Fragen, 2–3 Hits. `sim`-Werte kommentieren: „0,93 bei ‚Urlaubstage pro Jahr?‘ – Schwelle ist
  ein Tuning-Parameter, zu niedrig = falsche Antworten aus dem Cache.“
- A4: Kostentabelle k=3/5/10 – „k verdoppeln = Input-Kosten fast verdoppeln“.

## Lab 7 (30 Min)
- B1: `check_alerts` – 15 Zeilen. Auf `log` sollte mind. ein Alert feuern (No-Answer durch Off-Topic-Anfragen).
- B2: Blue/Green: zweite Collection (800 Zeichen), Dual-Read-Vergleich, `switch`. Diskussion Embedding-Modellwechsel.
- B3: Feedback-Kandidaten in `candidates.jsonl` + Prozessskizze.
- B4 (Bonus): Reranker-CPU-Kosten → Worker-Anzahl. Erwartung: `fast` p95 ~1–2 s auf CPU → bei 10k/Tag ~1–2 Worker.
- Wo TN hängen: Phoenix-Port; `pipe.index = ...` als Umschalten (in Produktion Qdrant-Alias).

## Typische Fragen
- „Langfuse oder Phoenix?“ – Beide Open Source; Phoenix läuft als `pip install` lokal (Kurs), Langfuse self-hosted
  braucht Postgres/ClickHouse, hat aber Prompt-Management und Nutzer-Feedback eingebaut. LangSmith = SaaS.
- „Semantic Cache bei zeitkritischen Inhalten (Preise)?“ – TTL kurz oder Cache pro Dokumentversion invalidieren;
  bei Re-Index Cache leeren.
- „Wie erkenne ich, dass ein neues Embedding-Modell besser ist?“ – Dual-Read auf dem Golden Set (Hit-Rate/MRR),
  dann Schatten-Betrieb (beide antworten, nur eine wird ausgespielt), dann Umschalten.
- „Kosten für 10k Anfragen/Tag realistisch?“ – Mit `gpt-5.6-luna`, k=5, ~1.500 Input-Tokens: ~3–5 USD/Tag reine LLM-Kosten.
  Der Reranker-Server kostet mehr als das LLM. Größere Modelle × 10.

## Stolperfallen
- Phoenix startet einen Server im Kernel; beim Kernel-Neustart Port 6006 kurz belegt → `start_phoenix(port=6007)`.
- `traced_run` braucht `openinference-semantic-conventions` (kommt mit den Instrumentation-Paketen).
- Matplotlib-Plots: `%matplotlib inline` ist in JupyterLab Standard.

## Überleitung zu Block 8
„Jetzt habt ihr alle Bausteine. Letzte Stunde: Wie sieht *eure* Architektur aus – und könnt ihr sie verteidigen?“
