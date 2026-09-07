# %% [markdown]
# # Lab 7 – Betrieb, Monitoring & Skalierung
#
# **Ziel (30 Min):** Eine RAG-Pipeline beobachtbar machen (Traces in Phoenix), aus Logs ein Mini-Dashboard bauen,
# einen Semantic Cache messen und die Kosten pro 10.000 Anfragen für verschiedene Konfigurationen abschätzen.
#
# **Was pro Request geloggt werden sollte:** Query (oder Hash), Rewrites, Retrieval-IDs + Scores, Reranker-Scores,
# Prompt-Hash, Tokens, Latenz je Stufe, Modellversion, Cache-Status, Nutzerrolle. Genau das steckt in `result.trace`.

# %%
import sys, os, time
_root = os.getcwd()                      # Repo-Root finden (Notebook liegt in labs/ oder labs/solutions/)
while not os.path.isdir(os.path.join(_root, "ragkurs")):
    _root = os.path.dirname(_root)
sys.path.insert(0, _root); os.chdir(_root)
import pandas as pd
pd.set_option("display.max_colwidth", 80)

from ragkurs import load_corpus, chunk_by_headings, HybridIndex, RAGPipeline, PipelineConfig, settings
from ragkurs.eval import load_golden
from ragkurs.tracing import start_phoenix, traced_run, log_to_dataframe

docs = load_corpus()
index = HybridIndex(collection="ops").build(chunk_by_headings(docs))
pipe = RAGPipeline(index, PipelineConfig(retrieval="hybrid", k=5, prefetch_k=20, rerank="fast", name="prod"))
golden = load_golden()

# %% [markdown]
# ## Teil A – Walkthrough
#
# ### A1 Phoenix starten (lokal, Open Source, kein Account) und Traces erzeugen
#
# LangChain- und OpenAI-Aufrufe werden automatisch instrumentiert (OpenInference/OpenTelemetry); unser `traced_run`
# legt darüber einen `rag.run`-Span mit den Pipeline-Attributen (Retrieval-Zeit, Kosten, Dokumente, ...).

# %%
url = start_phoenix()
print("Phoenix UI:", url, "-> im Browser öffnen, Projekt 'rag-schulung'")

# %%
for g in golden[:8]:
    r = traced_run(pipe, g["question"], ["employee"])
print("8 Traces gesendet. In Phoenix: Trace öffnen -> Spans 'rag.run' > 'ChatOpenAI' > ... mit Latenz und Tokens.")

# %% [markdown]
# ### A2 „Produktionsverkehr“ simulieren und ein Mini-Dashboard bauen
#
# Ohne Phoenix geht es auch: `pipeline.log` sammelt jeden Lauf. Daraus bauen wir die Kennzahlen, die in Produktion
# auf ein Dashboard gehören.

# %%
import random
random.seed(7)
traffic = [g["question"] for g in golden] + [
    "Wie viele Urlaubstage hab ich?", "Urlaubstage pro Jahr?", "Wieviel Urlaub bekomme ich?",       # Paraphrasen
    "Was ist der Sinn des Lebens?", "Schreib mir ein Gedicht über Fräsmaschinen",                    # Off-Topic
    "Passwort mindestlänge", "vpn client 2026",                                                      # Stichwort-Queries
]
random.shuffle(traffic)
for q in traffic[:30]:
    pipe.run(q, ["employee"])

log = log_to_dataframe(pipe)
log.tail(5)

# %%
kpis = {
    "anfragen": len(log),
    "p50_total_ms": log["t_total_ms"].quantile(0.5),
    "p95_total_ms": log["t_total_ms"].quantile(0.95),
    "p95_retrieval_ms": (log["t_retrieval_ms"] + log["t_rerank_ms"]).quantile(0.95),
    "p95_generation_ms": log["t_generation_ms"].quantile(0.95),
    "no_answer_rate": log["no_answer"].mean(),
    "avg_input_tokens": log["input_tokens"].mean(),
    "kosten_usd_gesamt": log["cost_usd"].sum(),
    "kosten_usd_pro_anfrage": log["cost_usd"].mean(),
}
pd.Series(kpis).round(4)

# %%
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3, figsize=(14, 3.5))
log[["t_retrieval_ms", "t_rerank_ms", "t_generation_ms"]].plot.box(ax=ax[0], title="Latenz je Stufe (ms)")
log["top_score"].plot.hist(ax=ax[1], bins=15, title="Top-1 Retrieval-Score")
log["input_tokens"].plot.hist(ax=ax[2], bins=15, title="Input-Tokens pro Anfrage")
plt.tight_layout()

# %% [markdown]
# **Lesen:** Ein Häufungspunkt niedriger Top-Scores = Off-Topic-Anfragen oder Wissenslücke → Kandidaten fürs Golden Set.
# Die Generierung dominiert die Latenz → Caching und kleinere Modelle wirken dort am meisten.
#
# ### A3 Semantic Cache
#
# Cache-Key ist das Embedding der Frage; Treffer ab Cosine ≥ 0,92 **und gleicher Rolle** (sonst ACL-Leak über den Cache!).

# %%
from ragkurs.cache import CachedPipeline, SemanticCache

cached = CachedPipeline(pipe, SemanticCache(threshold=0.92))
for q in ("Wie viele Urlaubstage haben Vollzeitmitarbeitende pro Jahr?", "Wie viele Urlaubstage hab ich?", "Urlaubstage pro Jahr?",
          "Wieviel Urlaub bekomme ich?", "Wie lange ist die Probezeit?", "Wie lang ist die Probezeit bei Aurelia?"):
    r = cached.run(q, ["employee"])
    print(f"{r.trace['cache']:4s} sim={r.trace['cache_similarity']:.3f} {r.trace['t_total_ms']:6.0f} ms | {q}")
print("\nCache-Statistik:", cached.cache.stats)

# %% [markdown]
# ### A4 Kostenmodell: 10.000 Anfragen pro Tag
#
# Kostentreiber: Input-Tokens (Kontextlänge × k), Output-Tokens, Embedding der Query, Reranker-CPU. Aus dem Log
# kennen wir die Durchschnittswerte pro Konfiguration.

# %%
def daily_cost(avg_in, avg_out, requests=10_000, cache_hit=0.0):
    per_req = avg_in / 1e6 * settings.price_in + avg_out / 1e6 * settings.price_out + 20 / 1e6 * settings.price_embed
    return requests * per_req * (1 - cache_hit)

rows = []
for cfg in (PipelineConfig(retrieval="hybrid", k=3, name="k=3"), PipelineConfig(retrieval="hybrid", k=5, name="k=5"), PipelineConfig(retrieval="hybrid", k=10, prefetch_k=25, max_context_chars=12000, name="k=10")):
    p = RAGPipeline(index, cfg)
    for g in golden[:8]:
        p.run(g["question"], ["employee"])
    l = log_to_dataframe(p)
    rows.append({"config": cfg.name, "avg_input_tokens": l["input_tokens"].mean(), "avg_output_tokens": l["output_tokens"].mean(),
                 "usd_pro_tag": daily_cost(l["input_tokens"].mean(), l["output_tokens"].mean()),
                 "usd_pro_tag_mit_30%_cache": daily_cost(l["input_tokens"].mean(), l["output_tokens"].mean(), cache_hit=0.3)})
pd.DataFrame(rows).round(2)

# %% [markdown]
# ## Teil B – Aufgaben
#
# ### B1 Alert-Regeln
# Schreibt `check_alerts(log_df) -> list[str]`, das folgende Zustände meldet:
# (a) No-Answer-Rate > 20 % in den letzten 20 Anfragen, (b) p95-Gesamtlatenz > 4 s, (c) > 30 % der Anfragen mit
# Top-Score unter dem 10 %-Quantil der Baseline. Testet es auf `log`.

# %%
# === LOESUNG START ===
# HINWEIS: log.tail(20)["no_answer"].mean() ; log["t_total_ms"].quantile(0.95) ; log["top_score"].quantile(0.1)
def check_alerts(df, baseline_df=None, window=20):
    alerts = []
    recent = df.tail(window)
    if recent["no_answer"].mean() > 0.20:
        alerts.append(f"No-Answer-Rate {recent['no_answer'].mean():.0%} in den letzten {len(recent)} Anfragen")
    p95 = df["t_total_ms"].quantile(0.95)
    if p95 > 4000:
        alerts.append(f"p95-Latenz {p95:.0f} ms > 4000 ms")
    base = (baseline_df if baseline_df is not None else df)["top_score"].quantile(0.10)
    low = (recent["top_score"] < base).mean()
    if low > 0.30:
        alerts.append(f"{low:.0%} der Anfragen mit Top-Score < {base:.3f} (Wissenslücke oder Off-Topic?)")
    return alerts or ["keine Alerts"]

check_alerts(log)
# === LOESUNG ENDE ===

# %% [markdown]
# ### B2 Blue/Green-Reindex und Embedding-Drift
# Ein neues Embedding-Modell (oder neues Chunking) heißt: **kompletter Re-Index**, und zwar ohne Downtime.
# Baut eine zweite Collection `ops_v2` mit anderem Chunking (`max_chars=800`), vergleicht beide auf dem Golden Set
# (Dual-Read), und schreibt eine Funktion `switch(active)` , die die Pipeline atomar umschaltet. Was müsst ihr
# zusätzlich tun, wenn sich das **Embedding-Modell** ändert (Cache! Query-Embedding! Sparse bleibt gleich)?

# %%
# === LOESUNG START ===
# HINWEIS: compare_retrieval([pipe_v1, pipe_v2], golden, ["employee"]) ; Umschalten = pipe.index = neue_collection
from ragkurs.eval import compare_retrieval
index_v2 = HybridIndex(collection="ops_v2").build(chunk_by_headings(docs, max_chars=800))
v1 = RAGPipeline(index, PipelineConfig(retrieval="hybrid", k=5, prefetch_k=20, rerank="fast", name="v1 (1500)"))
v2 = RAGPipeline(index_v2, PipelineConfig(retrieval="hybrid", k=5, prefetch_k=20, rerank="fast", name="v2 (800)"))
display(compare_retrieval([v1, v2], golden, ["employee"]))

active = {"index": index}                     # in Produktion: Alias in Qdrant (update_collection_aliases) statt Python-Dict
def switch(new_index):
    old = active["index"]; active["index"] = new_index; pipe.index = new_index
    print(f"umgeschaltet: {old.collection} -> {new_index.collection} ({new_index.count()} Chunks)")
switch(index_v2)
# Beim Embedding-Modellwechsel zusätzlich: Embedding-Cache invalidieren (.cache/embeddings_*.jsonl), Query-Embedding mit
# dem NEUEN Modell, beide Collections parallel halten bis der Vergleich abgeschlossen ist, dann alte löschen.
# === LOESUNG ENDE ===

# %% [markdown]
# ### B3 Feedback-Loop schließen
# Simuliert drei „Daumen runter“ aus Produktion (Fragen + was die richtige Antwort gewesen wäre). Schreibt sie als
# **Kandidaten** in `data/golden/candidates.jsonl` (`reviewed: false`) und skizziert den Prozess: Wer reviewt,
# wann landen sie im Golden Set, wann läuft der Regressionstest?

# %%
# === LOESUNG START ===
# HINWEIS: json.dumps(..., ensure_ascii=False) zeilenweise anhängen
import json
feedback = [
    {"question": "vpn client 2026", "expected": "GlobalProtect; AnyConnect am 28.02.2026 abgeschaltet", "source_docs": ["it-vpn-zugang"]},
    {"question": "Wieviel Urlaub bekomme ich?", "expected": "30 Arbeitstage bei 5-Tage-Woche", "source_docs": ["hr-urlaubsrichtlinie"]},
    {"question": "Was kostet die Anfahrt vom Servicetechniker?", "expected": "Anfahrtspauschale Deutschland 280 €", "source_docs": ["sales-preisliste-2026"]},
]
with open("data/golden/candidates.jsonl", "a", encoding="utf-8") as f:
    for i, fb in enumerate(feedback, start=1):
        f.write(json.dumps({"id": f"c{int(time.time())}{i}", "type": "feedback", "question": fb["question"], "ground_truth": fb["expected"],
                            "source_docs": fb["source_docs"], "answerable": True, "reviewed": False}, ensure_ascii=False) + "\n")
print(open("data/golden/candidates.jsonl", encoding="utf-8").read()[-600:])
# Prozess: wöchentliches Review durch Fachbereich -> reviewed=true -> Merge ins Golden Set -> CI-Regressionstest bei jedem PR.
# === LOESUNG ENDE ===

# %% [markdown]
# ### B4 (Bonus) Was kostet der Reranker wirklich?
# Der Reranker kostet keine API-Tokens, aber CPU-Zeit. Messt p95 mit und ohne `rerank="fast"` über 20 Anfragen und
# rechnet: Bei 10.000 Anfragen/Tag – wie viele CPU-Sekunden, wie viele parallele Worker für < 2 s p95?

# %%
# === LOESUNG START ===
# HINWEIS: log_to_dataframe(p)["t_rerank_ms"] ; Worker ≈ (Anfragen/s im Peak) × (Latenz in s)
p_no = RAGPipeline(index, PipelineConfig(retrieval="hybrid", k=5, prefetch_k=20, name="ohne rerank"))
for g in golden[:20]:
    p_no.retrieve_only(g["question"], ["employee"]); pipe.retrieve_only(g["question"], ["employee"])
rr_ms = pd.Series([r.trace.get("t_rerank_ms", 0) for r in [pipe.retrieve_only(g["question"], ["employee"]) for g in golden[:20]]])
cpu_s_per_day = rr_ms.mean() / 1000 * 10_000
peak_rps = 10_000 / (8 * 3600) * 3        # Annahme: 3x Durchschnitt im Peak, 8-Stunden-Tag
print(f"Reranker p50={rr_ms.median():.0f} ms, p95={rr_ms.quantile(0.95):.0f} ms -> {cpu_s_per_day:.0f} CPU-Sekunden/Tag; "
      f"Peak {peak_rps:.2f} req/s -> ~{max(1, round(peak_rps * rr_ms.quantile(0.95) / 1000))} Worker für den Reranker allein")
# === LOESUNG ENDE ===

# %% [markdown]
# ## Teil C – Debrief
#
# 1. Welche drei Kennzahlen würdet ihr auf das Dashboard eures Teams setzen – und welche auf das der Fachabteilung?
# 2. Semantic Cache: Wo ist er gefährlich? (Rollen, zeitkritische Inhalte wie Preise, Cache-Invalidierung bei Re-Index)
# 3. Was muss passieren, damit ein „Daumen runter“ innerhalb einer Woche zu einem Regressionstest wird?
#
# **Merksatz:** Ein RAG-System ohne Traces ist eine Blackbox mit API-Rechnung. Ohne Feedback-Loop wird es nie besser
# als am Go-Live-Tag.
