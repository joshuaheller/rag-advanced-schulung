"""RAG-Metriken nach den Ragas-Definitionen - transparent selbst implementiert.

Warum nicht direkt Ragas? Ragas pinnt LangChain-/OpenAI-Versionen, die mit dem restlichen Kurs-Stack
kollidieren (Stand 09/2026). Die Definitionen sind aber einfach genug, um sie in ~100 Zeilen nachzubauen -
und man sieht dabei genau, was ein LLM-Judge pro Metrik tut. (Ragas optional: requirements-optional.txt)

Metrik            | Kante             | Definition (vereinfacht)
------------------|-------------------|--------------------------------------------------------------
context_precision | Kontext <-> Frage | Anteil der Top-k-Chunks, die fuer die Referenzantwort relevant sind,
                  |                   | rang-gewichtet (Average Precision)
context_recall    | Kontext <-> Frage | Anteil der Aussagen der Referenzantwort, die durch den Kontext gedeckt sind
faithfulness      | Antwort <-> Kontext | Anteil der Aussagen der Antwort, die aus dem Kontext folgen
answer_relevancy  | Antwort <-> Frage | Cosine(Frage, aus der Antwort rueckgenerierte Fragen)
"""
from __future__ import annotations

import json
import math
import re

import pandas as pd

from .llm import get_embedder, get_judge_llm

_STATEMENTS = """Zerlege den folgenden Text in einzelne, eigenstaendige Aussagen (kurze Saetze, je eine Tatsache).
Text: {text}
Antworte NUR mit einer JSON-Liste von Strings."""

_SUPPORTED = """Pruefe fuer jede Aussage, ob sie aus dem Kontext folgt (durch ihn belegt ist).
Kontext:
{context}

Aussagen:
{statements}

Antworte NUR mit einer JSON-Liste von true/false in derselben Reihenfolge."""

_RELEVANT = """Ist der folgende Kontextabschnitt nuetzlich, um die Referenzantwort auf die Frage zu geben?
Frage: {question}
Referenzantwort: {reference}
Kontextabschnitt: {chunk}
Antworte NUR mit JSON: {{"relevant": true/false}}"""

_QUESTIONS = """Formuliere {n} Fragen, auf die der folgende Text eine direkte Antwort ist.
Text: {answer}
Antworte NUR mit einer JSON-Liste von Strings."""


def _json_list(raw: str) -> list:
    m = re.search(r"\[.*\]", raw, re.S)
    try:
        return json.loads(m.group(0)) if m else []
    except json.JSONDecodeError:
        return []


def _cos(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    return dot / ((math.sqrt(sum(x * x for x in a)) or 1) * (math.sqrt(sum(y * y for y in b)) or 1))


class RagMetrics:
    def __init__(self, llm=None, embedder=None):
        self.llm = llm or get_judge_llm()
        self.emb = embedder or get_embedder()

    def statements(self, text: str) -> list[str]:
        return [s for s in _json_list(self.llm.invoke(_STATEMENTS.format(text=text)).content) if isinstance(s, str)]

    def supported(self, statements: list[str], context: str) -> list[bool]:
        if not statements:
            return []
        raw = self.llm.invoke(_SUPPORTED.format(context=context, statements="\n".join(f"- {s}" for s in statements))).content
        flags = [bool(x) for x in _json_list(raw)]
        return (flags + [False] * len(statements))[: len(statements)]

    # ---- die vier Metriken ----------------------------------------------------
    def faithfulness(self, answer: str, contexts: list[str]) -> float:
        st = self.statements(answer)
        if not st:
            return 0.0
        flags = self.supported(st, "\n\n".join(contexts))
        return sum(flags) / len(flags)

    def context_recall(self, reference: str, contexts: list[str]) -> float:
        st = self.statements(reference)
        if not st:
            return 0.0
        flags = self.supported(st, "\n\n".join(contexts))
        return sum(flags) / len(flags)

    def context_precision(self, question: str, reference: str, contexts: list[str]) -> float:
        rel = []
        for c in contexts:
            raw = self.llm.invoke(_RELEVANT.format(question=question, reference=reference, chunk=c[:2000])).content
            m = re.search(r"\{.*\}", raw, re.S)
            try:
                rel.append(bool(json.loads(m.group(0)).get("relevant"))) if m else rel.append(False)
            except json.JSONDecodeError:
                rel.append(False)
        if not any(rel):
            return 0.0
        # Average Precision ueber die Raenge
        hits, ap = 0, 0.0
        for k, r in enumerate(rel, start=1):
            if r:
                hits += 1
                ap += hits / k
        return ap / sum(rel)

    def answer_relevancy(self, question: str, answer: str, n: int = 3) -> float:
        if "keine Information" in answer:
            return 0.0
        qs = [q for q in _json_list(self.llm.invoke(_QUESTIONS.format(n=n, answer=answer)).content) if isinstance(q, str)]
        if not qs:
            return 0.0
        qv = self.emb.embed_query(question)
        return sum(_cos(qv, v) for v in self.emb.embed_documents(qs)) / len(qs)

    # ---- Komfort: ganzen DataFrame bewerten ------------------------------------
    def evaluate(self, df: pd.DataFrame, pipeline, metrics: tuple[str, ...] = ("context_precision", "context_recall", "faithfulness", "answer_relevancy"), verbose: bool = True) -> pd.DataFrame:
        """df aus run_golden(); Kontexte kommen aus pipeline.log (keine zweite Generierung)."""
        by_q = {r.query: r for r in pipeline.log}
        out = df.copy()
        for i, row in out.iterrows():
            res = by_q.get(row["question"])
            ctx = [h.chunk.text for h in res.answer.sources] if res else []
            if "context_precision" in metrics:
                out.loc[i, "context_precision"] = self.context_precision(row["question"], row["ground_truth"], ctx)
            if "context_recall" in metrics:
                out.loc[i, "context_recall"] = self.context_recall(row["ground_truth"], ctx)
            if "faithfulness" in metrics:
                out.loc[i, "faithfulness"] = self.faithfulness(row["answer"], ctx)
            if "answer_relevancy" in metrics:
                out.loc[i, "answer_relevancy"] = self.answer_relevancy(row["question"], row["answer"])
            if verbose:
                print(f"  {row['id']}: " + " ".join(f"{m}={out.loc[i, m]:.2f}" for m in metrics))
        out.attrs["config"] = df.attrs.get("config", "")
        return out


def metrics_summary(df: pd.DataFrame) -> dict:
    cols = [c for c in ("context_precision", "context_recall", "faithfulness", "answer_relevancy") if c in df]
    return {"config": df.attrs.get("config", ""), **{c: round(df[c].mean(), 3) for c in cols}}
