"""Fuehrt alle Loesungs-Notebooks nacheinander mit echten Modellen aus und schreibt einen Bericht.

Aufruf im Repo-Root (venv aktiv, .env mit OPENAI_API_KEY):
    python scripts/run_all_labs.py              # alle Labs
    python scripts/run_all_labs.py 0 1 2        # nur bestimmte Labs
    python scripts/run_all_labs.py --offline    # Testmodus ohne API (FAKE_EMBEDDINGS=1)

Ergebnis:
    labs/solutions/executed/labN_*.ipynb   ausgefuehrte Kopien mit allen Outputs (gitignored)
    run_report.md                          Status je Lab, Fehler mit Zellennummer, Laufzeit, wichtige Ergebnistabellen
Dauer: ca. 45-90 Min (Modelle laden, ~600 LLM-Calls), Kosten ca. 2-4 USD.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOL = ROOT / "labs" / "solutions"
OUT = SOL / "executed"
REPORT = ROOT / "run_report.md"

args = [a for a in sys.argv[1:] if not a.startswith("--")]
offline = "--offline" in sys.argv
if offline:
    os.environ["FAKE_EMBEDDINGS"] = "1"

sys.path.insert(0, str(ROOT))
from ragkurs.config import settings  # noqa: E402

if not offline and not settings.openai_api_key:
    sys.exit("OPENAI_API_KEY fehlt in .env - oder --offline verwenden.")

OUT.mkdir(exist_ok=True)
notebooks = sorted(SOL.glob("lab*.ipynb"))
if args:
    notebooks = [nb for nb in notebooks if nb.name[3] in args]

lines = [f"# Testlauf {time.strftime('%Y-%m-%d %H:%M')}", "",
         f"Modus: {'OFFLINE (Fake-Embeddings, Dummy-LLM)' if offline else 'echte Modelle'} | chat={settings.chat_model} judge={settings.judge_model} embed={settings.embed_model}", ""]
summary = []

for nb in notebooks:
    t0 = time.time()
    print(f"==> {nb.name} ...", flush=True)
    target = OUT / nb.name
    cmd = [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--allow-errors",
           "--ExecutePreprocessor.timeout=1800", "--output", str(target), str(nb)]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(SOL))
    dt = time.time() - t0
    if not target.exists():
        summary.append((nb.name, "ABBRUCH", dt, 0))
        lines += [f"## {nb.name} - ABBRUCH nach {dt:.0f}s", "", "```", proc.stderr[-2000:], "```", ""]
        continue
    data = json.loads(target.read_text(encoding="utf-8"))
    errors, tables = [], []
    for i, cell in enumerate(data["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        for out in cell.get("outputs", []):
            if out.get("output_type") == "error":
                errors.append((i, out.get("ename"), out.get("evalue", "")[:300], src[:200]))
            txt = "".join(out.get("data", {}).get("text/plain", [])) if "data" in out else "".join(out.get("text", []))
            if any(k in txt for k in ("hit_rate", "accuracy", "p95", "kosten", "usd_pro_tag")) and len(txt) < 3000:
                tables.append((i, txt))
    status = "OK" if not errors else f"{len(errors)} FEHLER"
    summary.append((nb.name, status, dt, len(errors)))
    lines += [f"## {nb.name} - {status} ({dt / 60:.1f} Min)", ""]
    for i, ename, evalue, src in errors:
        lines += [f"### Fehler in Zelle {i}: {ename}", "", "```", evalue, "```", "Zellenanfang:", "```python", src, "```", ""]
    for i, txt in tables[:12]:
        lines += [f"<details><summary>Ergebnis Zelle {i}</summary>", "", "```", txt.strip(), "```", "</details>", ""]
    print(f"    {status} in {dt / 60:.1f} Min")

lines.insert(3, "| Notebook | Status | Dauer |\n|---|---|---|\n" + "\n".join(f"| {n} | {s} | {d / 60:.1f} Min |" for n, s, d, _ in summary) + "\n")
REPORT.write_text("\n".join(lines), encoding="utf-8")
print(f"\nBericht: {REPORT}")
print("\n".join(f"{n:32s} {s:12s} {d / 60:5.1f} Min" for n, s, d, _ in summary))
sys.exit(1 if any(e for *_, e in summary) else 0)
