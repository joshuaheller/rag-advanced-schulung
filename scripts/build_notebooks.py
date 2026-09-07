"""Baut aus den Musterloesungen (labs/solutions/*.py, jupytext percent-Format) die Notebooks:

    labs/solutions/labN_*.ipynb   - vollstaendige Loesung (Trainer)
    labs/labN_*.ipynb             - Teilnehmer-Version: Loesungsbloecke durch TODO ersetzt

Konvention im Quelltext:
    # === LOESUNG START ===
    # HINWEIS: <Text bleibt in der TN-Version stehen>
    ...Loesungscode...
    # === LOESUNG ENDE ===

Aufruf:  python scripts/build_notebooks.py
"""
from __future__ import annotations

import re
from pathlib import Path

import jupytext

ROOT = Path(__file__).resolve().parents[1]
SOL = ROOT / "labs" / "solutions"
OUT = ROOT / "labs"

START, END = "# === LOESUNG START ===", "# === LOESUNG ENDE ==="


def strip_solutions(src: str) -> str:
    out, i, lines = [], 0, src.splitlines()
    while i < len(lines):
        line = lines[i]
        if line.strip() == START:
            indent = line[: len(line) - len(line.lstrip())]
            i += 1
            hints = []
            while i < len(lines) and lines[i].strip() != END:
                if lines[i].strip().startswith("# HINWEIS"):
                    hints.append(lines[i])
                i += 1
            out += hints
            out.append(f"{indent}# TODO: hier implementieren (Loesung: labs/solutions/)")
            out.append(f"{indent}...")
            i += 1  # END-Zeile ueberspringen
            continue
        out.append(line)
        i += 1
    return "\n".join(out) + "\n"


def build(py: Path) -> None:
    src = py.read_text(encoding="utf-8")
    nb_sol = jupytext.reads(src, fmt="py:percent")
    jupytext.write(nb_sol, SOL / (py.stem + ".ipynb"))
    nb_tn = jupytext.reads(strip_solutions(src), fmt="py:percent")
    for nb in (nb_sol, nb_tn):
        nb.metadata["kernelspec"] = {"name": "rag-schulung", "display_name": "Python (rag-schulung)", "language": "python"}
    jupytext.write(nb_tn, OUT / (py.stem + ".ipynb"))
    n_todo = src.count(START)
    print(f"{py.name}: {len(nb_sol.cells)} Zellen, {n_todo} Aufgabenbloecke")


if __name__ == "__main__":
    for py in sorted(SOL.glob("lab*.py")):
        build(py)
