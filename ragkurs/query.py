"""Query-Transformationen: Rewriting, Multi-Query, Decomposition, HyDE, Step-back."""
from __future__ import annotations

import json
import re

from .llm import get_llm

_REWRITE = """Formuliere die folgende Nutzerfrage als praezise Suchanfrage fuer eine Unternehmens-Wissensdatenbank
(deutsch, Fachbegriffe beibehalten, Fuellwoerter entfernen, keine Antwort geben).
Frage: {q}
Suchanfrage:"""

_MULTI = """Erzeuge {n} unterschiedliche Formulierungen der folgenden Frage fuer eine Dokumentensuche.
Variiere Wortwahl und Perspektive (Synonyme, Fachbegriff vs. Alltagssprache, konkreter/abstrakter).
Frage: {q}
Antworte NUR mit einer JSON-Liste von Strings."""

_DECOMP = """Zerlege die Frage in die minimal noetigen Teilfragen, die jeweils mit EINEM Dokumentabschnitt
beantwortet werden koennen. Wenn die Frage bereits atomar ist, gib sie unveraendert als einzige Teilfrage zurueck.
Frage: {q}
Antworte NUR mit einer JSON-Liste von Strings (max. 4)."""

_HYDE = """Schreibe einen kurzen, plausiblen Absatz (3-4 Saetze) im Stil einer Unternehmensrichtlinie oder eines
Handbuchs, der die folgende Frage beantworten wuerde. Erfinde konkrete Details, es geht nur um die Sprache.
Frage: {q}"""

_STEPBACK = """Formuliere zu der konkreten Frage eine allgemeinere Hintergrundfrage, deren Antwort die konkrete
Frage einordnet (z. B. 'Welche Regel gilt fuer ...?' statt eines Sonderfalls).
Frage: {q}
Hintergrundfrage:"""


def _json_list(raw: str, fallback: list[str]) -> list[str]:
    m = re.search(r"\[.*\]", raw, re.S)
    try:
        val = json.loads(m.group(0)) if m else None
        if isinstance(val, list) and all(isinstance(x, str) for x in val) and val:
            return val
    except json.JSONDecodeError:
        pass
    return fallback


def rewrite(query: str, llm=None) -> str:
    llm = llm or get_llm()
    return llm.invoke(_REWRITE.format(q=query)).content.strip().strip('"')


def multi_query(query: str, n: int = 3, llm=None) -> list[str]:
    llm = llm or get_llm()
    variants = _json_list(llm.invoke(_MULTI.format(q=query, n=n)).content, [])
    return [query, *[v for v in variants if v.strip() and v.strip() != query][:n]]


def decompose(query: str, llm=None) -> list[str]:
    llm = llm or get_llm()
    return _json_list(llm.invoke(_DECOMP.format(q=query)).content, [query])[:4]


def hyde(query: str, llm=None) -> str:
    llm = llm or get_llm()
    return llm.invoke(_HYDE.format(q=query)).content.strip()


def step_back(query: str, llm=None) -> str:
    llm = llm or get_llm()
    return llm.invoke(_STEPBACK.format(q=query)).content.strip()
