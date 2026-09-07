"""Synthetische Testfragen erzeugen (Golden-Set-Bootstrapping).

Eigene, transparente Implementierung: Pro zufälligem Chunk erzeugt das LLM eine Frage samt Referenzantwort,
die NUR aus diesem Chunk beantwortbar ist. Für Multi-Hop werden zwei Chunks kombiniert.
Ragas' TestsetGenerator macht Ähnliches mit einem Knowledge Graph (siehe `ragas_testset`).

Synthetische Fragen sind ein Startpunkt - vor der Aufnahme ins Golden Set prüft ein Mensch jede Frage.
"""
from __future__ import annotations

import json
import random
import re

import pandas as pd

from .chunking import chunk_by_headings
from .llm import get_llm
from .loading import Document

_SINGLE = """Du erzeugst Testfragen fuer einen Unternehmens-Wissensassistenten.
Erzeuge aus dem folgenden Abschnitt EINE konkrete Frage, wie sie ein Mitarbeiter stellen wuerde, und die
korrekte Antwort. Die Frage muss allein aus dem Abschnitt beantwortbar sein und sollte eine konkrete Zahl,
Frist oder Regel abfragen. Nenne in der Frage nicht den Dokumentnamen.

Abschnitt:
{chunk}

Antworte NUR mit JSON: {{"question": "...", "ground_truth": "..."}}"""

_MULTI = """Du erzeugst Testfragen fuer einen Unternehmens-Wissensassistenten.
Erzeuge aus den beiden folgenden Abschnitten EINE Frage, die nur mit Informationen aus BEIDEN Abschnitten
vollstaendig beantwortet werden kann (Multi-Hop), und die korrekte Antwort.

Abschnitt A:
{a}

Abschnitt B:
{b}

Antworte NUR mit JSON: {{"question": "...", "ground_truth": "..."}}"""


def _parse(raw: str) -> dict | None:
    m = re.search(r"\{.*\}", raw, re.S)
    try:
        d = json.loads(m.group(0)) if m else None
        return d if d and d.get("question") and d.get("ground_truth") else None
    except json.JSONDecodeError:
        return None


def generate_synthetic_testset(docs: list[Document], n: int = 10, multi_hop_share: float = 0.3, seed: int = 42, llm=None, min_chars: int = 300) -> pd.DataFrame:
    """Erzeugt n Frage/Antwort-Paare (Anteil Multi-Hop einstellbar). Kosten: 1 LLM-Call pro Frage."""
    llm = llm or get_llm()
    rng = random.Random(seed)
    chunks = [c for c in chunk_by_headings(docs) if len(c.text) >= min_chars]
    rows = []
    n_multi = int(round(n * multi_hop_share))
    for i in range(n):
        if i < n_multi and len(chunks) >= 2:
            a, b = rng.sample(chunks, 2)
            d = _parse(llm.invoke(_MULTI.format(a=a.text, b=b.text)).content)
            src, typ = sorted({a.doc_id, b.doc_id}), "synthetic-multi-hop"
        else:
            c = rng.choice(chunks)
            d = _parse(llm.invoke(_SINGLE.format(chunk=c.text)).content)
            src, typ = [c.doc_id], "synthetic"
        if d:
            rows.append({"id": f"s{i + 1:02d}", "type": typ, "question": d["question"], "ground_truth": d["ground_truth"], "source_docs": src, "answerable": True, "reviewed": False})
    return pd.DataFrame(rows, columns=["id", "type", "question", "ground_truth", "source_docs", "answerable", "reviewed"])


def save_as_golden(df: pd.DataFrame, path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for _, r in df.iterrows():
            f.write(json.dumps({k: (v if not hasattr(v, "tolist") else v.tolist()) for k, v in r.items()}, ensure_ascii=False) + "\n")


def ragas_testset(docs: list[Document], n: int = 6):
    """Alternative mit Ragas' TestsetGenerator (Knowledge-Graph-basiert). API kann je nach Ragas-Version abweichen."""
    from langchain_core.documents import Document as LCDocument
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.testset import TestsetGenerator

    from .llm import get_embedder, get_judge_llm

    gen = TestsetGenerator(llm=LangchainLLMWrapper(get_judge_llm()), embedding_model=LangchainEmbeddingsWrapper(get_embedder()))
    lc_docs = [LCDocument(page_content=d.text, metadata={"doc_id": d.doc_id, "title": d.title}) for d in docs]
    return gen.generate_with_langchain_docs(lc_docs, testset_size=n).to_pandas()
