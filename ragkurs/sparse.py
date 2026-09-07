"""Einfacher, transparenter BM25-Encoder fuer Qdrant-Sparse-Vektoren.

Warum selbst gebaut statt fastembed? Damit die TN jede Zeile sehen: Tokenisierung,
TF-Saettigung (k1), Laengennormalisierung (b). Die IDF-Gewichtung uebernimmt Qdrant
serverseitig (`Modifier.IDF`), genau wie bei fastembeds `Qdrant/bm25`.

score(q, d) = sum_t IDF(t) * tf(t,d) * (k1 + 1) / (tf(t,d) + k1 * (1 - b + b * |d| / avgdl))
"""
from __future__ import annotations

import hashlib
import math
import re
from collections import Counter

from qdrant_client import models

_TOKEN_RE = re.compile(r"[a-zA-ZäöüÄÖÜß0-9][a-zA-ZäöüÄÖÜß0-9\-]*")

# kleine deutsche Stoppwortliste (bewusst kurz - Fachbegriffe muessen ueberleben)
STOPWORDS = set(
    """der die das den dem des ein eine einer eines einem einen und oder aber in im am an auf aus bei mit
    von vom zu zum zur fuer für ist sind war waren wird werden wurde wurden hat haben hatte hatten sich nicht
    kein keine keinen auch nur noch als wie wenn dann dass da so es er sie ich wir ihr sein seine ihre ihren
    ihrem ihres dieser diese dieses diesem diesen welche welcher welches was wer wo wann bis über ueber unter
    nach vor durch ohne um bzw z b etc the of and to a is""".split()
)


def tokenize(text: str) -> list[str]:
    toks = [t.lower() for t in _TOKEN_RE.findall(text)]
    out = []
    for t in toks:
        if t in STOPWORDS or len(t) < 2:
            continue
        # sehr leichtes "Stemming": Plural-/Flexionsendungen kappen (deutsch)
        for suf in ("ungen", "en", "er", "es", "e", "n", "s"):
            if len(t) > 5 and t.endswith(suf):
                t = t[: -len(suf)]
                break
        out.append(t)
    return out


def token_id(token: str) -> int:
    return int(hashlib.md5(token.encode("utf-8")).hexdigest()[:8], 16) % (2**31 - 1)


class SimpleBM25Encoder:
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self.avgdl = 1.0
        self.vocab: dict[int, str] = {}  # id -> Token (nur zur Anzeige)

    def fit(self, texts: list[str]) -> "SimpleBM25Encoder":
        lengths = [len(tokenize(t)) for t in texts]
        self.avgdl = (sum(lengths) / len(lengths)) if lengths else 1.0
        return self

    def encode_document(self, text: str) -> models.SparseVector:
        toks = tokenize(text)
        tf = Counter(toks)
        dl = len(toks)
        indices, values = [], []
        for tok, f in tf.items():
            tid = token_id(tok)
            self.vocab[tid] = tok
            w = f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            indices.append(tid)
            values.append(float(w))
        return models.SparseVector(indices=indices, values=values)

    def encode_query(self, text: str) -> models.SparseVector:
        toks = set(tokenize(text))
        return models.SparseVector(indices=[token_id(t) for t in toks], values=[1.0] * len(toks))

    def explain(self, text: str) -> list[tuple[str, float]]:
        """Zeigt die Tokens und Gewichte eines Textes (fuer die Folie 'BM25 in 3 Minuten')."""
        v = self.encode_document(text)
        pairs = [(self.vocab.get(i, "?"), round(w, 3)) for i, w in zip(v.indices, v.values)]
        return sorted(pairs, key=lambda p: -p[1])


def idf(term_doc_freq: int, n_docs: int) -> float:
    """Qdrants IDF-Formel (zur Veranschaulichung)."""
    return math.log(1 + (n_docs - term_doc_freq + 0.5) / (term_doc_freq + 0.5))
