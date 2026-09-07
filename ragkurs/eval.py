"""Evaluation: Golden Set, Retrieval-Metriken, LLM-as-Judge, Failure-Klassifikation, Ragas-Bruecke."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from .config import settings
from .llm import get_judge_llm
from .pipeline import RAGPipeline, RunResult

# ---------------------------------------------------------------------------
# Golden Set
# ---------------------------------------------------------------------------
def load_golden(path: Path | None = None, types: list[str] | None = None, answerable: bool | None = None, ids: list[str] | None = None) -> list[dict]:
    path = path or settings.golden_path
    rows = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    if types:
        rows = [r for r in rows if r["type"] in types]
    if answerable is not None:
        rows = [r for r in rows if r.get("answerable", True) == answerable]
    if ids:
        rows = [r for r in rows if r["id"] in ids]
    return rows


# ---------------------------------------------------------------------------
# Retrieval-Metriken (Dokument-Ebene: wurde das Quelldokument gefunden?)
# ---------------------------------------------------------------------------
def retrieval_row(result: RunResult, item: dict, k: int) -> dict:
    retrieved = result.retrieved_doc_ids[:k]
    sources = set(item.get("source_docs", []))
    ranks = [retrieved.index(s) + 1 for s in sources if s in retrieved]
    return {
        "id": item["id"],
        "type": item["type"],
        "question": item["question"],
        "source_docs": sorted(sources),
        "retrieved_docs": retrieved,
        "hit": bool(sources) and bool(ranks),                       # mind. eine Quelle in Top-k
        "recall": (len(ranks) / len(sources)) if sources else None,   # Anteil gefundener Quellen
        "first_rank": min(ranks) if ranks else None,
        "rr": (1.0 / min(ranks)) if ranks else 0.0,
        "t_retrieval_ms": result.trace.get("t_retrieval_ms", 0) + result.trace.get("t_rerank_ms", 0),
    }


def evaluate_retrieval(pipeline: RAGPipeline, golden: list[dict], user_roles: list[str] | None = None, k: int | None = None) -> pd.DataFrame:
    """Nur Retrieval (keine LLM-Kosten). Nutzt answerable=True-Fragen mit Quellen."""
    k = k or pipeline.config.k
    rows = []
    for item in golden:
        if not item.get("source_docs"):
            continue
        res = pipeline.retrieve_only(item["question"], user_roles=user_roles)
        rows.append(retrieval_row(res, item, k))
    df = pd.DataFrame(rows)
    df.attrs["config"] = pipeline.config.label()
    return df


def retrieval_summary(df: pd.DataFrame) -> dict:
    return {
        "config": df.attrs.get("config", ""),
        "n": len(df),
        "hit_rate": round(df["hit"].mean(), 3),
        "recall": round(df["recall"].mean(), 3),
        "mrr": round(df["rr"].mean(), 3),
        "p@1": round((df["first_rank"] == 1).mean(), 3),
        "avg_ms": round(df["t_retrieval_ms"].mean(), 0),
    }


def compare_retrieval(pipelines: list[RAGPipeline], golden: list[dict], user_roles: list[str] | None = None) -> pd.DataFrame:
    return pd.DataFrame([retrieval_summary(evaluate_retrieval(p, golden, user_roles)) for p in pipelines]).set_index("config")


# ---------------------------------------------------------------------------
# LLM-as-a-Judge (binaer, mit Begruendung - Hamel Husain: binaere Labels, kalibrierbar)
# ---------------------------------------------------------------------------
_JUDGE_PROMPT = """Du bewertest die Antwort eines Wissensassistenten. Vergleiche sie mit der Referenzantwort.

Frage: {question}
Referenzantwort (korrekt): {ground_truth}
Antwort des Assistenten: {answer}

Bewertungsregel: Die Antwort ist korrekt, wenn die KERNAUSSAGE der Referenz (die gefragte Zahl, Frist, Regel oder
Entscheidung) uebereinstimmt und die Antwort nichts enthaelt, was der Referenz widerspricht. Fehlende Zusatz- oder
Nebeninformationen aus der Referenz (Ausnahmen, Vergleichswerte, Begruendungen) machen die Antwort NICHT falsch.
Falsch ist die Antwort, wenn die Kernzahl/-frist abweicht, ein falsches Produkt/Dokument gemeint ist, oder wenn sie
"keine Information" sagt, obwohl die Referenz eine Antwort enthaelt (und umgekehrt).

Antworte NUR mit JSON: {{"correct": true/false, "reason": "<ein Satz>"}}"""


def judge_correctness(question: str, answer_text: str, ground_truth: str, llm=None) -> dict:
    llm = llm or get_judge_llm()
    raw = llm.invoke(_JUDGE_PROMPT.format(question=question, ground_truth=ground_truth, answer=answer_text)).content
    m = re.search(r"\{.*\}", raw, re.S)
    try:
        data = json.loads(m.group(0)) if m else {}
        return {"correct": bool(data.get("correct", False)), "reason": str(data.get("reason", ""))}
    except json.JSONDecodeError:
        return {"correct": False, "reason": f"Judge-Antwort nicht parsebar: {raw[:100]}"}


# ---------------------------------------------------------------------------
# End-to-End-Lauf ueber das Golden Set + Failure-Klassifikation
# ---------------------------------------------------------------------------
def _doc_family(doc_id: str) -> str:
    """'prod-handbuch-ax200' -> 'prod-handbuch' (fuer Near-Miss-Erkennung)."""
    return re.sub(r"-(ax\d+|20\d\d|\d+)$", "", doc_id)


def classify_failure(row: dict) -> str:
    """Heuristische Erstklassifikation - im Lab von den TN manuell zu pruefen/korrigieren."""
    if row["correct"]:
        return "ok"
    if not row["answerable"]:
        return "halluzination_statt_absage"            # Frage unbeantwortbar, System antwortet trotzdem
    if not row["hit"]:
        fams = {_doc_family(d) for d in row["retrieved_docs"]}
        if any(_doc_family(s) in fams for s in row["source_docs"]):
            return "retrieval:semantic_near_miss"       # aehnliches, aber falsches Dokument (AX-300 statt AX-200, 2024 statt 2026)
        return "retrieval:missing_evidence"
    if row["recall"] is not None and row["recall"] < 1.0:
        return "retrieval:partial_evidence"             # Multi-Hop: nur ein Teil der Quellen gefunden
    if row.get("is_no_answer"):
        return "generation:over_refusal"                # Evidenz da, Modell verweigert
    if row.get("first_rank", 1) and row["first_rank"] > 2:
        return "generation:context_noise"               # richtige Quelle weit hinten im Kontext
    return "generation:wrong_answer"


def run_golden(pipeline: RAGPipeline, golden: list[dict], user_roles: list[str] | None = None, judge: bool = True, verbose: bool = True) -> pd.DataFrame:
    """Kompletter Lauf (Retrieval + Generierung + Judge). Kosten: ~2 LLM-Calls pro Frage."""
    rows = []
    judge_llm = get_judge_llm() if judge else None
    for i, item in enumerate(golden, start=1):
        res = pipeline.run(item["question"], user_roles=user_roles)
        r = retrieval_row(res, item, pipeline.config.k)
        r.update(
            {
                "answerable": item.get("answerable", True),
                "answer": res.answer.text,
                "ground_truth": item["ground_truth"],
                "cited_docs": res.answer.cited_doc_ids(),
                "is_no_answer": res.answer.is_no_answer,
                "t_total_ms": res.trace.get("t_total_ms", 0),
                "cost_usd": res.trace.get("cost_usd", 0),
                "context_chars": res.trace.get("context_chars", 0),
            }
        )
        if judge:
            j = judge_correctness(item["question"], res.answer.text, item["ground_truth"], judge_llm)
            r["correct"], r["judge_reason"] = j["correct"], j["reason"]
        else:
            r["correct"], r["judge_reason"] = None, ""
        r["failure"] = classify_failure(r) if judge else ""
        rows.append(r)
        if verbose:
            mark = "OK " if r["correct"] else ("-- " if r["correct"] is None else "ERR")
            print(f"[{i:2d}/{len(golden)}] {mark} {item['id']} {r['failure']:<32} {item['question'][:60]}")
    df = pd.DataFrame(rows)
    df.attrs["config"] = pipeline.config.label()
    return df


def e2e_summary(df: pd.DataFrame) -> dict:
    ans = df[df["answerable"]]
    unans = df[~df["answerable"]]
    return {
        "config": df.attrs.get("config", ""),
        "n": len(df),
        "accuracy": round(df["correct"].mean(), 3) if df["correct"].notna().any() else None,
        "retrieval_hit_rate": round(ans["hit"].mean(), 3) if len(ans) else None,
        "refusal_correct": round(unans["is_no_answer"].mean(), 3) if len(unans) else None,
        "avg_total_ms": round(df["t_total_ms"].mean(), 0),
        "cost_usd_total": round(df["cost_usd"].sum(), 4),
    }


def failure_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(["failure", "type"]).size().unstack(fill_value=0)


# ---------------------------------------------------------------------------
# Ragas-Bruecke (Lab 5)
# ---------------------------------------------------------------------------
def to_ragas_samples(df: pd.DataFrame, pipeline: RAGPipeline | None = None) -> list:
    """Baut ragas SingleTurnSamples aus einem run_golden-DataFrame.

    Kontexte werden aus den Pipeline-Logs geholt (pipeline.log), damit keine zweite Generierung noetig ist.
    """
    from ragas import SingleTurnSample

    by_q = {r.query: r for r in (pipeline.log if pipeline else [])}
    samples = []
    for _, row in df.iterrows():
        res = by_q.get(row["question"])
        contexts = [h.chunk.text for h in res.answer.sources] if res else []
        samples.append(
            SingleTurnSample(user_input=row["question"], response=row["answer"], reference=row["ground_truth"], retrieved_contexts=contexts)
        )
    return samples


def evaluate_ragas(df: pd.DataFrame, pipeline: RAGPipeline, metrics: list | None = None):
    """Fuehrt die Original-Ragas-Metriken aus - nur, wenn `ragas` installiert ist (requirements-optional.txt).

    Im Kurs verwenden wir stattdessen ragkurs.metrics.RagMetrics (gleiche Definitionen, keine Versionskonflikte).
    """
    try:
        import ragas  # noqa: F401
    except ImportError as e:
        raise ImportError("ragas ist nicht installiert (siehe requirements-optional.txt). Nutze ragkurs.metrics.RagMetrics.") from e
    from ragas import EvaluationDataset, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness

    from .llm import get_embedder, get_judge_llm

    llm = LangchainLLMWrapper(get_judge_llm())
    emb = LangchainEmbeddingsWrapper(get_embedder())
    metrics = metrics or [ContextPrecision(llm=llm), ContextRecall(llm=llm), Faithfulness(llm=llm), AnswerRelevancy(llm=llm, embeddings=emb)]
    dataset = EvaluationDataset(samples=to_ragas_samples(df, pipeline))
    return evaluate(dataset=dataset, metrics=metrics, llm=llm, embeddings=emb)
