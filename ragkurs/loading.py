"""Laden des Korpus: Markdown mit Frontmatter und PDFs mit Sidecar-Metadaten.

Zwei PDF-Parser stehen zur Wahl:
- "pypdf"   : schnell, verliert Tabellenstruktur (Baseline, Lab 0-3)
- "docling" : layoutbasiert, liefert Markdown mit Ueberschriften und Tabellen (Lab 4)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal

import yaml

from .config import settings

PdfParser = Literal["pypdf", "docling"]


@dataclass
class Document:
    doc_id: str
    title: str
    text: str
    metadata: dict = field(default_factory=dict)
    source_path: str = ""

    def __repr__(self) -> str:  # kompakt in Notebooks
        return f"Document({self.doc_id!r}, {len(self.text)} Zeichen, access={self.metadata.get('access')})"


def split_frontmatter(text: str) -> tuple[dict, str]:
    m = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    if not m:
        return {}, text
    return (yaml.safe_load(m.group(1)) or {}), text[m.end():]


def _normalize_meta(meta: dict, fallback_id: str) -> dict:
    meta = dict(meta)
    meta.setdefault("doc_id", fallback_id)
    meta.setdefault("title", fallback_id)
    meta.setdefault("access", ["all"])
    meta.setdefault("status", "current")
    meta.setdefault("department", "unbekannt")
    if isinstance(meta["access"], str):
        meta["access"] = [meta["access"]]
    # Datumsfelder als String halten (Qdrant-Payload, JSON)
    for k in ("valid_from", "valid_until"):
        if k in meta and meta[k] is not None:
            meta[k] = str(meta[k])
    meta["version"] = str(meta.get("version", ""))
    return meta


def _read_pdf_pypdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return "\n\n".join((page.extract_text() or "") for page in reader.pages)


_docling_converter = None


def _read_pdf_docling(path: Path) -> str:
    global _docling_converter
    if settings.fake_embeddings:  # Offline-Modus: Markdown-Original statt Docling (nur fuer Tests ohne Internet)
        src = path.parent.parent / "corpus_src" / (path.stem + ".md")
        if src.exists():
            return split_frontmatter(src.read_text(encoding="utf-8"))[1]
    if _docling_converter is None:
        from docling.document_converter import DocumentConverter

        _docling_converter = DocumentConverter()
    result = _docling_converter.convert(str(path))
    return result.document.export_to_markdown()


def load_document(path: Path, pdf_parser: PdfParser = "pypdf") -> Document:
    """Laedt eine einzelne Datei (.md oder .pdf) als Document."""
    if path.suffix.lower() == ".md":
        meta, body = split_frontmatter(path.read_text(encoding="utf-8"))
        meta = _normalize_meta(meta, path.stem)
        return Document(meta["doc_id"], meta["title"], body.strip(), meta, str(path))
    if path.suffix.lower() == ".pdf":
        sidecar = path.with_suffix(".meta.yaml")
        meta = yaml.safe_load(sidecar.read_text(encoding="utf-8")) if sidecar.exists() else {}
        meta = _normalize_meta(meta, path.stem)
        meta["parser"] = pdf_parser
        text = _read_pdf_docling(path) if pdf_parser == "docling" else _read_pdf_pypdf(path)
        return Document(meta["doc_id"], meta["title"], text.strip(), meta, str(path))
    raise ValueError(f"Nicht unterstuetztes Format: {path}")


def load_corpus(
    corpus_dir: Path | None = None,
    pdf_parser: PdfParser = "pypdf",
    include: Iterable[str] | None = None,
    extra_dirs: Iterable[Path] = (),
) -> list[Document]:
    """Laedt alle .md/.pdf-Dateien des Korpus (plus optionale Zusatzordner, z. B. data/poison)."""
    corpus_dir = corpus_dir or settings.corpus_dir
    paths: list[Path] = []
    for d in [corpus_dir, *extra_dirs]:
        paths += sorted(p for p in Path(d).iterdir() if p.suffix.lower() in {".md", ".pdf"})
    docs = [load_document(p, pdf_parser=pdf_parser) for p in paths]
    if include is not None:
        keep = set(include)
        docs = [d for d in docs if d.doc_id in keep]
    return docs


def corpus_overview(docs: list[Document]):
    """Kleine Tabelle fuer Notebooks (pandas optional)."""
    rows = [
        {
            "doc_id": d.doc_id,
            "department": d.metadata.get("department"),
            "access": ",".join(d.metadata.get("access", [])),
            "status": d.metadata.get("status"),
            "format": Path(d.source_path).suffix,
            "zeichen": len(d.text),
        }
        for d in docs
    ]
    try:
        import pandas as pd

        return pd.DataFrame(rows)
    except ImportError:  # pragma: no cover
        return rows
