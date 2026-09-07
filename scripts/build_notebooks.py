"""Baut aus den Musterloesungen (labs/solutions/*.py, jupytext percent-Format) die Notebooks:

    labs/solutions/labN_*.ipynb   - vollstaendige Loesung (Trainer)
    labs/labN_*.ipynb             - Teilnehmer-Version (Fill-in-the-blank)

Konventionen im Quelltext:
    # === LOESUNG START ===
    # HINWEIS: <bleibt in der TN-Version stehen>
    zeile = irgendwas(...)  # ? <Hinweis>      <- markierte Zeile: in der TN-Version wird der Code durch ... ersetzt,
    ...                                          der Hinweis bleibt als TODO-Kommentar
    # === LOESUNG ENDE ===

Enthaelt ein Block keine "# ?"-Markierung, wird der gesamte Block durch eine TODO-Zelle ersetzt.
Zusaetzlich bekommt jedes Notebook nach der ersten Code-Zelle einen API-Spickzettel (docs/CHEATSHEET.md, Abschnitt des Labs).

Aufruf:  python scripts/build_notebooks.py
"""
from __future__ import annotations

import re
from pathlib import Path

import jupytext
import nbformat

ROOT = Path(__file__).resolve().parents[1]
SOL = ROOT / "labs" / "solutions"
OUT = ROOT / "labs"
CHEAT = ROOT / "docs" / "CHEATSHEET.md"

START, END = "# === LOESUNG START ===", "# === LOESUNG ENDE ==="
MARK = re.compile(r"^(\s*)(.*?)\s*# \? (.*)$")


def blank_line(line: str) -> str:
    m = MARK.match(line)
    indent, code, hint = m.group(1), m.group(2).rstrip(), m.group(3)
    code_nc = code.split("  #")[0].rstrip()          # angehaengte Kommentare entfernen
    if code_nc.endswith(","):                       # Dict-/Listeneintrag: Schluessel behalten, Wert blank
        key = code_nc.split(":", 1)[0]
        return f'{indent}{key}: ...,  # TODO: {hint}' if ":" in code_nc else f"{indent}...,  # TODO: {hint}"
    if re.match(r"^(if|elif|while)\b", code_nc) and code_nc.endswith(":"):
        return f"{indent}{code_nc.split()[0]} ...:  # TODO: {hint}"
    if code_nc.startswith("return "):
        return f"{indent}return ...  # TODO: {hint}"
    if re.match(r"^[A-Za-z_][\w\.\[\]\"', ]*\s=\s", code_nc) and "==" not in code_nc.split("=", 1)[0]:
        lhs = code_nc.split(" = ", 1)[0]
        return f"{indent}{lhs} = ...  # TODO: {hint}"
    return f"{indent}...  # TODO: {hint}"


def strip_solutions(src: str) -> str:
    out, i, lines = [], 0, src.splitlines()
    while i < len(lines):
        line = lines[i]
        if line.strip() == START:
            indent = line[: len(line) - len(line.lstrip())]
            block = []
            i += 1
            while i < len(lines) and lines[i].strip() != END:
                block.append(lines[i])
                i += 1
            i += 1  # END-Zeile
            if any(MARK.match(b) for b in block):
                for b in block:
                    out.append(blank_line(b) if MARK.match(b) else b)
            else:
                out += [b for b in block if b.strip().startswith("# HINWEIS")]
                out.append(f"{indent}# TODO: hier implementieren (Loesung: labs/solutions/)")
                out.append(f"{indent}...")
            continue
        out.append(line)
        i += 1
    return "\n".join(out) + "\n"


def cheat_section(lab_stem: str) -> str | None:
    """Abschnitt '## Lab N' aus docs/CHEATSHEET.md."""
    if not CHEAT.exists():
        return None
    n = lab_stem[3]
    text = CHEAT.read_text(encoding="utf-8")
    m = re.search(rf"^## Lab {n}\b.*?(?=^## Lab |\Z)", text, re.S | re.M)
    if not m:
        return None
    body = m.group(0).split("\n", 1)[1].strip()
    return f"### API-Spickzettel für dieses Lab\n\nAlle Funktionen mit Signatur und Beispiel: `docs/CHEATSHEET.md`. Im Notebook: `funktion??` zeigt den Quelltext.\n\n{body}"


def insert_cheatsheet(nb, lab_stem: str) -> None:
    section = cheat_section(lab_stem)
    if not section:
        return
    for idx, cell in enumerate(nb.cells):
        if cell.cell_type == "code":
            nb.cells.insert(idx + 1, nbformat.v4.new_markdown_cell(section))
            return


def build(py: Path) -> None:
    src = py.read_text(encoding="utf-8")
    nb_sol = jupytext.reads(src.replace("  # ? ", "  # "), fmt="py:percent")   # Loesung: Hinweis als normaler Kommentar
    nb_tn = jupytext.reads(strip_solutions(src), fmt="py:percent")
    for nb in (nb_sol, nb_tn):
        nb.metadata["kernelspec"] = {"name": "rag-schulung", "display_name": "Python (rag-schulung)", "language": "python"}
        insert_cheatsheet(nb, py.stem)
    jupytext.write(nb_sol, SOL / (py.stem + ".ipynb"))
    jupytext.write(nb_tn, OUT / (py.stem + ".ipynb"))
    n_blocks = src.count(START)
    n_marks = len([l for l in src.splitlines() if MARK.match(l)])
    print(f"{py.name}: {len(nb_sol.cells)} Zellen, {n_blocks} Aufgabenbloecke, {n_marks} Luecken")


if __name__ == "__main__":
    for py in sorted(SOL.glob("lab*.py")):
        build(py)
