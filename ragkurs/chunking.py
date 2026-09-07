"""Chunking-Strategien.

- chunk_fixed        : Baseline, rekursiver Zeichen-Splitter (ignoriert Struktur)
- chunk_by_headings  : struktur-erhaltend, Breadcrumb "Dokument > Abschnitt" im Chunk-Text,
                       Tabellen bleiben moeglichst zusammen
- chunk_parent_child : kleine Kinder fuer das Retrieval, grosse Eltern fuer den Kontext
- add_contextual_prefix : "Contextual Retrieval" (Anthropic) - LLM schreibt pro Chunk einen
                       kurzen Kontextsatz, der vor den Chunk gestellt wird
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from .loading import Document


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    metadata: dict = field(default_factory=dict)

    @property
    def section(self) -> str:
        return self.metadata.get("section", "")

    def __repr__(self) -> str:
        return f"Chunk({self.chunk_id!r}, {len(self.text)} Zeichen, section={self.section!r})"


def _cid(doc_id: str, idx: int, strategy: str) -> str:
    return f"{doc_id}::{strategy}::{idx:03d}"


def _base_meta(doc: Document, strategy: str) -> dict:
    return {
        "doc_id": doc.doc_id,
        "title": doc.title,
        "department": doc.metadata.get("department"),
        "access": doc.metadata.get("access", ["all"]),
        "status": doc.metadata.get("status", "current"),
        "version": doc.metadata.get("version", ""),
        "valid_from": doc.metadata.get("valid_from"),
        "strategy": strategy,
    }


# ---------------------------------------------------------------------------
# 1) Baseline: fixe Groesse
# ---------------------------------------------------------------------------
def chunk_fixed(docs: list[Document], chunk_size: int = 800, chunk_overlap: int = 100) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks: list[Chunk] = []
    for doc in docs:
        for i, piece in enumerate(splitter.split_text(doc.text)):
            meta = _base_meta(doc, "fixed") | {"chunk_index": i, "section": ""}
            chunks.append(Chunk(_cid(doc.doc_id, i, "fixed"), doc.doc_id, piece, meta))
    return chunks


# ---------------------------------------------------------------------------
# 2) Struktur-erhaltend: nach Ueberschriften, mit Breadcrumb
# ---------------------------------------------------------------------------
_HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3")]


def _split_keep_tables(text: str, max_chars: int) -> list[str]:
    """Teilt zu lange Abschnitte, ohne Markdown-Tabellen auseinanderzureissen."""
    if len(text) <= max_chars:
        return [text]
    blocks = re.split(r"\n\s*\n", text)  # Absaetze / Tabellenbloecke
    out: list[str] = []
    buf = ""
    for block in blocks:
        if len(block) > max_chars and not block.lstrip().startswith("|"):
            # sehr langer Fliesstext: rekursiv teilen
            if buf:
                out.append(buf)
                buf = ""
            out += RecursiveCharacterTextSplitter(chunk_size=max_chars, chunk_overlap=80).split_text(block)
            continue
        if len(buf) + len(block) + 2 > max_chars and buf:
            out.append(buf)
            buf = block
        else:
            buf = f"{buf}\n\n{block}" if buf else block
    if buf:
        out.append(buf)
    return out


def chunk_by_headings(docs: list[Document], max_chars: int = 1500, breadcrumb: bool = True) -> list[Chunk]:
    header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=_HEADERS, strip_headers=True)
    chunks: list[Chunk] = []
    for doc in docs:
        idx = 0
        for sec in header_splitter.split_text(doc.text):
            path = [sec.metadata.get(k) for k in ("h1", "h2", "h3") if sec.metadata.get(k)]
            if not sec.metadata.get("h1"):  # ohne H1 (z. B. pypdf-Text): Dokumenttitel voranstellen
                path = [doc.title, *path]
            section = " > ".join(path)
            for piece in _split_keep_tables(sec.page_content.strip(), max_chars):
                if not piece.strip():
                    continue
                text = f"[{section}]\n{piece}" if breadcrumb else piece
                meta = _base_meta(doc, "headings") | {"chunk_index": idx, "section": section}
                chunks.append(Chunk(_cid(doc.doc_id, idx, "headings"), doc.doc_id, text, meta))
                idx += 1
    return chunks


# ---------------------------------------------------------------------------
# 3) Parent-Child: klein suchen, gross liefern
# ---------------------------------------------------------------------------
@dataclass
class ParentStore:
    parents: dict[str, Chunk]

    def get(self, parent_id: str) -> Chunk:
        return self.parents[parent_id]


def chunk_parent_child(
    docs: list[Document], parent_chars: int = 1500, child_chars: int = 350, child_overlap: int = 40
) -> tuple[list[Chunk], ParentStore]:
    """Eltern = Abschnitte (struktur-erhaltend), Kinder = kleine Fenster darin.

    Indexiert werden die Kinder; beim Antworten wird der Elternchunk geliefert (siehe
    RAGPipeline(parent_expand=True)).
    """
    parents = chunk_by_headings(docs, max_chars=parent_chars)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=child_chars, chunk_overlap=child_overlap)
    children: list[Chunk] = []
    store: dict[str, Chunk] = {}
    for p in parents:
        store[p.chunk_id] = p
        body = p.text.split("\n", 1)[1] if p.text.startswith("[") else p.text
        for j, piece in enumerate(child_splitter.split_text(body)):
            meta = dict(p.metadata) | {"strategy": "child", "parent_id": p.chunk_id, "child_index": j}
            text = f"[{p.section}]\n{piece}"
            children.append(Chunk(f"{p.chunk_id}::c{j:02d}", p.doc_id, text, meta))
    return children, ParentStore(store)


# ---------------------------------------------------------------------------
# 4) Contextual Retrieval (Anthropic, 2024): LLM-Kontextsatz vor jeden Chunk
# ---------------------------------------------------------------------------
_CTX_PROMPT = """Hier ist ein vollstaendiges Dokument:
<dokument>
{doc}
</dokument>

Hier ist ein Ausschnitt daraus:
<chunk>
{chunk}
</chunk>

Schreibe einen kurzen Kontext (1-2 Saetze, Deutsch), der diesen Ausschnitt im Gesamtdokument einordnet,
damit er bei einer Suche besser gefunden wird (z. B. worum es geht, welches Produkt/welche Regel,
welche Version). Antworte NUR mit dem Kontext."""


def add_contextual_prefix(chunks: list[Chunk], docs: list[Document], llm=None, max_doc_chars: int = 12000) -> list[Chunk]:
    """Erzeugt fuer jeden Chunk einen LLM-Kontextsatz (kostet einen LLM-Call pro Chunk!)."""
    from .llm import get_llm

    llm = llm or get_llm()
    by_id = {d.doc_id: d for d in docs}
    out: list[Chunk] = []
    for c in chunks:
        doc_text = by_id[c.doc_id].text[:max_doc_chars]
        ctx = llm.invoke(_CTX_PROMPT.format(doc=doc_text, chunk=c.text)).content.strip()
        meta = dict(c.metadata) | {"contextualized": True, "context_prefix": ctx}
        out.append(Chunk(c.chunk_id, c.doc_id, f"{ctx}\n\n{c.text}", meta))
    return out


def fingerprint(chunks: list[Chunk]) -> str:
    """Stabiler Hash ueber Chunk-Texte (fuer Caches)."""
    h = hashlib.sha1()
    for c in chunks:
        h.update(c.chunk_id.encode())
        h.update(c.text.encode())
    return h.hexdigest()[:12]
