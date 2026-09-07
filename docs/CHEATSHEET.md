# ragkurs – API-Spickzettel

Alle Funktionen, die in den Labs gebraucht werden, mit Signatur und Beispiel. Im Notebook zeigt `funktion??` den Quelltext,
`funktion?` die Signatur. Die Objekte, die überall vorkommen:

| Objekt | Was es ist | Wichtige Attribute |
|---|---|---|
| `Document` | ein geladenes Dokument | `.doc_id`, `.title`, `.text`, `.metadata` (dict: department, access, status, version) |
| `Chunk` | ein Textstück im Index | `.chunk_id`, `.doc_id`, `.text`, `.metadata`, `.section` |
| `Hit` | ein Suchtreffer | `.chunk`, `.score`, `.rank`, `.doc_id`, `.source` (dense/sparse/hybrid/rerank), `.extra["pre_rank"]` |
| `RunResult` | Ergebnis eines Pipeline-Laufs | `.answer.text`, `.hits`, `.retrieved_doc_ids`, `.trace` (dict), `.show()` |
| `golden` | Liste von Dicts | `id`, `type`, `question`, `ground_truth`, `source_docs`, `answerable`, ggf. `allowed_roles` |

## Lab 0 – Setup & Baseline

```python
from ragkurs import load_corpus, chunk_fixed, HybridIndex, RAGPipeline, PipelineConfig
from ragkurs.eval import load_golden, evaluate_retrieval, retrieval_summary

docs   = load_corpus()                                   # 25 Documents; load_corpus(pdf_parser="docling") ab Lab 4
chunks = chunk_fixed(docs, chunk_size=800, chunk_overlap=100)
index  = HybridIndex(collection="baseline").build(chunks)
pipe   = RAGPipeline(index, PipelineConfig(retrieval="dense", k=5, name="baseline"))

result = pipe.run("Frage?", user_roles=["employee"])    # Retrieval + Antwort;  result.show()  zeigt alles
result.answer.text            # Antworttext          result.retrieved_doc_ids   # ['hr-urlaubsrichtlinie', ...]
result.trace["t_total_ms"]    # Zeiten, Tokens, Kosten, Kandidaten

golden = load_golden()                                   # 50 Fragen; load_golden(types=["tabelle"]) / ids=[...] / answerable=False
item = next(g for g in golden if g["id"] == "g33")       # eine bestimmte Frage
df  = evaluate_retrieval(pipe, golden, user_roles=["employee"])   # nur Retrieval, kein LLM -> DataFrame (hit, recall, first_rank, ...)
retrieval_summary(df)                                    # {'hit_rate': .., 'recall': .., 'mrr': .., 'avg_ms': ..}
df.groupby("type")["hit"].mean()                         # Hit-Rate je Fragetyp
```

## Lab 1 – Failure Analysis

```python
from ragkurs.eval import run_golden, e2e_summary, failure_breakdown, judge_correctness, retrieval_row
import pandas as pd

df = run_golden(pipe, golden[:20], user_roles=["employee"])   # Retrieval + Antwort + Judge + Fehlerklasse  (2 LLM-Calls/Frage)
# Spalten: id, type, question, answer, ground_truth, retrieved_docs, source_docs, hit, recall, first_rank,
#          correct (Judge), judge_reason, failure (Heuristik), is_no_answer, t_total_ms, cost_usd
e2e_summary(df)                                    # accuracy, retrieval_hit_rate, refusal_correct, Kosten
failure_breakdown(df)                              # Kreuztabelle failure x type
pd.crosstab(df["failure"], df["type"], margins=True)
df.loc[df.id == "g12", "failure"] = "generation:wrong_answer"     # Zelle korrigieren
df["failure"].str.split(":").str[0].value_counts()              # retrieval vs generation

judge_correctness(frage, antwort, referenz)        # -> {"correct": bool, "reason": str}
res = pipe.retrieve_only(frage, ["employee"])      # nur Retrieval (schnell, kostenlos)
retrieval_row(res, item, k=20)                     # -> dict mit hit, recall, first_rank fuer eine Golden-Frage
```

## Lab 2 – Hybrid Search & Query-Transformationen

```python
from ragkurs import chunk_by_headings
from ragkurs.eval import compare_retrieval
from ragkurs import query as qt
from ragkurs.chunking import add_contextual_prefix

PipelineConfig(retrieval="dense" | "sparse" | "hybrid", k=5, prefetch_k=20, fusion="rrf" | "dbsf",
               query_transform=None | "rewrite" | "multi" | "decompose" | "hyde" | "stepback",
               status_filter="current" | None, name="Label")
compare_retrieval([pipe_a, pipe_b, ...], golden, user_roles=["employee"])   # eine Zeile pro Pipeline: hit_rate, recall, mrr, avg_ms

index.search(q, mode="hybrid", k=5)           # -> list[Hit]      auch: search_dense / search_sparse / search_hybrid(q, k, prefetch_k, fusion)
index.sparse.explain("Text")                  # BM25-Tokens mit Gewichten
qt.rewrite(q); qt.multi_query(q, n=3); qt.decompose(q); qt.hyde(q); qt.step_back(q)   # je 1 LLM-Call

hr_chunks = [c for c in chunks if c.metadata.get("department") == "HR"]
add_contextual_prefix(hr_chunks, docs)        # LLM-Kontextsatz vor jeden Chunk (1 Call pro Chunk!)
HybridIndex(collection="anderer_name").build(neue_chunks)   # zweiter Index, eigener Name
```

## Lab 3 – Reranking

```python
from ragkurs.rerank import get_reranker, timed_rerank
import time

rr = get_reranker("fast" | "quality" | "colbert" | "llm")   # Instanz gecacht; erster Aufruf laedt das Modell
hits = index.search_hybrid(q, k=20, prefetch_k=30)
top5 = rr.rerank(q, hits, top_k=5)                # -> list[Hit] mit .score (Reranker) und .extra["pre_rank"]
top5, ms = timed_rerank(rr, q, hits, top_k=5)     # zusaetzlich Millisekunden

PipelineConfig(retrieval="hybrid", k=5, prefetch_k=20, rerank="fast")   # Reranker in der Pipeline
df = evaluate_retrieval(pipe, golden, ["employee"]); df["t_retrieval_ms"].quantile(0.95)   # Latenz-Quantile (inkl. Rerank)
t0 = time.perf_counter(); ...; ms = (time.perf_counter() - t0) * 1000   # Zeit messen
```

## Lab 4 – Context Engineering

```python
from ragkurs import chunk_by_headings, chunk_parent_child
from ragkurs.loading import load_document
from ragkurs.generate import build_context, lost_in_the_middle_reorder, answer as gen_answer

docs_docling = load_corpus(pdf_parser="docling")            # PDFs strukturiert (Tabellen!), einmal laden und wiederverwenden
chunk_by_headings(docs, max_chars=1500, breadcrumb=True)    # Abschnitte mit "[Dokument > Kapitel]"-Kopfzeile
children, parents = chunk_parent_child(docs, parent_chars=1500, child_chars=350)
RAGPipeline(HybridIndex(collection="pc").build(children), PipelineConfig(retrieval="hybrid", parent_expand=True), parent_store=parents)

PipelineConfig(..., reorder="none" | "lost_in_middle", max_context_chars=6000)
build_context(hits, max_chars=2500, reorder="lost_in_middle")   # -> (kontext_string, verwendete_hits)
gen_answer(q, hits, max_chars=10**7)                             # Antwort direkt aus einer Hit-Liste (ohne Pipeline)
sum(len(h.chunk.text) for h in res.hits)                         # Kontextlaenge in Zeichen
```

## Lab 5 – Evaluation

```python
from ragkurs.metrics import RagMetrics, metrics_summary
from ragkurs.eval_synth import generate_synthetic_testset
import json, subprocess

rm = RagMetrics(); m = rm.evaluate(df, pipe)     # df aus run_golden; Spalten context_precision, context_recall, faithfulness, answer_relevancy
metrics_summary(m)
generate_synthetic_testset(docs[:8], n=6)        # DataFrame: question, ground_truth, source_docs, reviewed=False

# eigenes Golden Set: eine JSON-Zeile pro Frage
with open("data/golden/golden_set_team.jsonl", "w", encoding="utf-8") as f:
    f.write(json.dumps({"id": "t01", "type": "faktisch", "question": "...", "ground_truth": "...", "source_docs": ["hr-arbeitszeit"], "answerable": True}, ensure_ascii=False) + "\n")
load_golden(path="data/golden/golden_set_team.jsonl")

# DataFrames ueber die id verbinden
m = df_ref[["id", "correct"]].merge(df_cand[["id", "correct"]], on="id", suffixes=("_ref", "_cand"))
m[(m["correct_ref"]) & (~m["correct_cand"])]["id"].tolist()      # Regressionen
```

## Lab 6 – Security

```python
from ragkurs.security import scan_text, scan_documents, ingest_gate, acl_leak_test
from ragkurs.loading import Document
from ragkurs.llm import get_llm
from ragkurs.generate import answer, SYSTEM_PROMPT

load_corpus(extra_dirs=[settings.poison_dir])                    # Korpus + vergiftete Dokumente
scan_text(doc_id, text)                                          # -> list[Finding] (leer = unauffaellig)
ingest_gate(docs, trusted_departments={"HR", "IT", ...}).report()
acl_leak_test(pipe, golden_acl, {"Mitarbeiter": ["employee"], "HR": ["hr"]})   # -> Zeilen mit leak True/False
PipelineConfig(..., enforce_acl=False)                           # Filter absichtlich aus (Demo)

Document(doc_id="x", title="...", text="...", metadata={"doc_id": "x", "department": "Service", "version": "2026-09", "access": ["all"], "status": "current"})
get_llm().invoke("Prompt-Text").content                          # ein LLM-Aufruf, Antwort als String
h.chunk.metadata.get("access", [])                               # Rollenliste eines Treffers, z. B. ["hr", "management"]
result.trace["candidates"]                                       # [(chunk_id, score), ...] – alles, was der Retriever sah
answer(q, hits, system_prompt=SYSTEM_PROMPT.replace("...", ""))  # Antwort mit veraendertem System-Prompt
```

## Lab 7 – Betrieb

```python
from ragkurs.tracing import start_phoenix, traced_run, log_to_dataframe
from ragkurs.cache import CachedPipeline, SemanticCache
from ragkurs.eval import compare_retrieval

start_phoenix()                                   # UI unter http://localhost:6006 (Port belegt? start_phoenix(port=6007))
traced_run(pipe, q, ["employee"])                 # wie pipe.run, aber mit Trace in Phoenix
log = log_to_dataframe(pipe)                      # alle bisherigen Laeufe: t_*_ms, no_answer, top_score, tokens, cost_usd, cache
log.tail(20)["no_answer"].mean(); log["t_total_ms"].quantile(0.95); log["top_score"].quantile(0.10)

cached = CachedPipeline(pipe, SemanticCache(threshold=0.92)); r = cached.run(q, ["employee"]); r.trace["cache"]   # "hit"/"miss"
index_v2 = HybridIndex(collection="ops_v2").build(chunk_by_headings(docs, max_chars=800))   # zweite Collection
pipe.index = index_v2                             # Umschalten (in Produktion: Qdrant-Alias)
r.trace.get("t_rerank_ms", 0)                     # Reranker-Zeit eines Laufs
```
