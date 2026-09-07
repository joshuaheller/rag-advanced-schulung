# RAG in der Praxis – Advanced: Architektur, Evaluation & produktiver Betrieb

Begleit-Repository zur 2-tägigen Schulung (IT-Schulungen.com, Trainer: Joshua Heller / TAISC).
Alles hier ist Open Source, läuft lokal und braucht **keinen Account** – nur einen OpenAI-API-Key, den der Trainer ausgibt.

## Schnellstart

```bash
git clone https://github.com/joshuaheller/rag-advanced-schulung.git
cd rag-advanced-schulung
bash setup.sh                     # Windows: powershell -ExecutionPolicy Bypass -File setup.ps1
# .env öffnen und OPENAI_API_KEY eintragen
source .venv/bin/activate         # Windows: .\.venv\Scripts\Activate.ps1
python smoke_test.py              # muss mit "Setup OK" enden
jupyter lab                       # -> labs/lab0_setup_baseline.ipynb, Kernel "Python (rag-schulung)"
```

Systemvoraussetzungen: Python 3.11/3.12, 16 GB RAM (min. 8), ~10 GB Platte (Pakete + Modelle), Internet zu
`api.openai.com`, `pypi.org`, `huggingface.co`, `github.com`. Keine GPU, kein Docker.

## Was ist drin

| Ordner | Inhalt |
|---|---|
| `labs/` | 8 Notebooks für Teilnehmende (Walkthrough + Aufgaben mit TODOs) |
| `labs/solutions/` | dieselben Notebooks mit Musterlösungen (Quelle: `*.py` im jupytext-Format) |
| `ragkurs/` | Kurs-Bibliothek: Laden, Chunking, Qdrant-Hybrid-Index, Reranker, Pipeline, Evaluation, Security, Cache, Tracing |
| `data/corpus/` | Korpus „Aurelia Maschinenbau GmbH“: 25 Dokumente (Markdown + PDF) mit eingebauten Fallen |
| `data/golden/` | Golden Set: 50 Fragen mit Referenzantworten, Quellen und Typ |
| `data/poison/` | 2 vergiftete Dokumente für Lab 6 (RAG Poisoning, Prompt Injection) |
| `tests/` + `.github/workflows/` | Eval als CI-Quality-Gate (pytest + DeepEval + GitHub Actions), Lab 5 |
| `trainer/` | Briefings und Ablauf pro Block (für den Trainer) |
| `scripts/` | Korpus-PDFs bauen, Notebooks aus Lösungen generieren, Modelle vorab laden |

## Ablauf der Labs

| Lab | Thema | Dauer |
|---|---|---|
| 0 | Umgebungs-Check, Korpus, naive Baseline | 15 Min |
| 1 | Failure Analysis: Fehler klassifizieren, Hebel wählen | 30 Min |
| 2 | Hybrid Search (Dense + BM25), RRF, Query-Transformationen | 35 Min |
| 3 | Reranking: Cross-Encoder vs. ColBERT vs. LLM, Trade-offs | 30 Min |
| 4 | Context Engineering: Docling, struktur-erhaltendes Chunking, Parent-Child, Re-Ordering, Long-Context | 25 + 20 Min |
| 5 | Evaluation: Golden Set, Ragas-Metriken, Judge-Kalibrierung, CI/CD | 45 Min |
| 6 | Security: Poisoning, Injection, Ingest-Gate, Permission-aware Retrieval | 35 Min |
| 7 | Betrieb: Phoenix-Tracing, Dashboard, Semantic Cache, Kosten, Blue/Green-Reindex | 30 Min |

Jedes Notebook: **Teil A** Walkthrough (Trainer führt vor, alle führen mit aus) → **Teil B** Aufgaben (TODO-Zellen,
Lösungen in `labs/solutions/`) → **Teil C** Debrief-Fragen.

## Technischer Stack

Python · LangChain (OpenAI-Anbindung, Text-Splitter) · Qdrant im lokalen Modus (Dense + Sparse + Hybrid, Pre-Filter) ·
eigener BM25-Encoder · sentence-transformers (Cross-Encoder-Reranker) · rerankers (ColBERT) · Docling (PDF-Parsing) ·
DeepEval + pytest (CI) · Arize Phoenix (Tracing, lokal) · pandas/matplotlib.

Modelle (Standard, per `.env` änderbar): `gpt-5.6-luna` (Generator), `gpt-5.6-terra` (Judge), `text-embedding-3-small`.
Embeddings werden in `.cache/` zwischengespeichert – wiederholte Notebook-Läufe kosten nichts.

## Offline-Modus

`FAKE_EMBEDDINGS=1` ersetzt Embeddings und LLM durch lokale Platzhalter (Hashing-Embeddings, Dummy-Antworten).
Damit laufen Setup-Check, Retrieval-Tests und die Notebook-Mechanik ohne API-Key – für echte Ergebnisse braucht es den Key.

## Lizenz

Kursmaterial © 2026 Joshua Heller / TAISC – The AI Software Company GmbH, für Teilnehmende der Schulung zur
persönlichen Nutzung. Der Korpus „Aurelia Maschinenbau“ ist fiktiv; Gesetzesauszüge sind gemeinfrei (§ 5 UrhG).
Verwendete Bibliotheken stehen unter ihren jeweiligen Open-Source-Lizenzen (Apache-2.0 / MIT / BSD; Phoenix: Elastic-2.0).
