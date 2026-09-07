"""Erzeugt aus Markdown-Quellen mit `render_pdf: true` echte PDF-Dokumente (mit Tabellen).

Zweck im Kurs: Ein Teil des Korpus liegt NUR als PDF vor. Der Baseline-Loader (pypdf)
verliert dabei die Tabellenstruktur - das ist der Ausgangspunkt fuer Lab 4
(struktur-erhaltendes Parsing mit Docling).

Aufruf (einmalig, ist bereits im Repo ausgefuehrt):
    python scripts/build_corpus_pdfs.py
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

import yaml
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "corpus"
SRC = ROOT / "data" / "corpus_src"  # Markdown-Originale der PDFs (nur fuer Referenz)

REPLACEMENTS = {"✓": "ja", "–": "-", "—": "-", "→": "->", "„": '"', "“": '"', "”": '"', "·": "-"}


def clean(text: str) -> str:
    for k, v in REPLACEMENTS.items():
        text = text.replace(k, v)
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`(.+?)`", r"<font face='Courier'>\1</font>", text)
    return text


def split_frontmatter(text: str) -> tuple[dict, str]:
    m = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    if not m:
        return {}, text
    return yaml.safe_load(m.group(1)) or {}, text[m.end():]


def md_to_flowables(body: str, styles) -> list:
    flow = []
    lines = body.splitlines()
    i = 0
    para_buf: list[str] = []

    def flush_para():
        if para_buf:
            flow.append(Paragraph(clean(" ".join(para_buf)), styles["Body"]))
            flow.append(Spacer(1, 3 * mm))
            para_buf.clear()

    while i < len(lines):
        line = lines[i]
        if line.startswith("|"):
            flush_para()
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
                    rows.append([Paragraph(clean(c), styles["Cell"]) for c in cells])
                i += 1
            ncols = max(len(r) for r in rows)
            width = (A4[0] - 40 * mm) / ncols
            t = Table(rows, colWidths=[width] * ncols, repeatRows=1)
            t.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DCE6F1")),
                        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#8FA3BF")),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                    ]
                )
            )
            flow.append(t)
            flow.append(Spacer(1, 4 * mm))
            continue
        if line.startswith("#"):
            flush_para()
            level = len(line) - len(line.lstrip("#"))
            text = line.lstrip("#").strip()
            style = {1: styles["H1"], 2: styles["H2"]}.get(level, styles["H3"])
            flow.append(Paragraph(clean(text), style))
            i += 1
            continue
        if line.startswith("- ") or re.match(r"^\d+\. ", line):
            flush_para()
            text = re.sub(r"^(- |\d+\. )", "", line)
            bullet = "-" if line.startswith("- ") else line.split(".")[0] + "."
            flow.append(Paragraph(f"{bullet} {clean(text)}", styles["KBullet"]))
            i += 1
            continue
        if line.startswith(">"):
            flush_para()
            flow.append(Paragraph(clean(line.lstrip("> ")), styles["KQuote"]))
            i += 1
            continue
        if not line.strip():
            flush_para()
            i += 1
            continue
        para_buf.append(line.strip())
        i += 1
    flush_para()
    return flow


def build_styles():
    ss = getSampleStyleSheet()
    ss.add(ParagraphStyle("Body", parent=ss["Normal"], fontSize=10, leading=14))
    ss.add(ParagraphStyle("Cell", parent=ss["Normal"], fontSize=8.5, leading=11))
    ss.add(ParagraphStyle("H1", parent=ss["Heading1"], fontSize=17, spaceAfter=6 * mm))
    ss.add(ParagraphStyle("H2", parent=ss["Heading2"], fontSize=13, spaceBefore=5 * mm))
    ss.add(ParagraphStyle("H3", parent=ss["Heading3"], fontSize=11))
    ss.add(ParagraphStyle("KBullet", parent=ss["Normal"], fontSize=10, leading=14, leftIndent=6 * mm))
    ss.add(ParagraphStyle("KQuote", parent=ss["Normal"], fontSize=9.5, leading=13, leftIndent=6 * mm, textColor=colors.HexColor("#444444")))
    return ss


def render(md_path: Path) -> None:
    meta, body = split_frontmatter(md_path.read_text(encoding="utf-8"))
    if not meta.get("render_pdf"):
        return
    styles = build_styles()
    pdf_path = CORPUS / (md_path.stem + ".pdf")
    doc = SimpleDocTemplate(
        str(pdf_path), pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm, topMargin=18 * mm, bottomMargin=18 * mm,
        title=meta.get("title", md_path.stem), author="Aurelia Maschinenbau GmbH",
    )
    header = Paragraph(
        clean(f"Aurelia Maschinenbau GmbH - {meta.get('department', '')} - Version {meta.get('version', '')}"),
        styles["KQuote"],
    )
    doc.build([header, Spacer(1, 3 * mm)] + md_to_flowables(body, styles))
    meta_out = {k: v for k, v in meta.items() if k != "render_pdf"}
    (CORPUS / (md_path.stem + ".meta.yaml")).write_text(yaml.safe_dump(meta_out, allow_unicode=True, sort_keys=False), encoding="utf-8")
    SRC.mkdir(exist_ok=True)
    shutil.move(str(md_path), SRC / md_path.name)
    print(f"PDF erzeugt: {pdf_path.name}")


if __name__ == "__main__":
    for md in sorted(CORPUS.glob("*.md")):
        render(md)
