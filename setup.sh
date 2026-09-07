#!/usr/bin/env bash
# Setup fuer Ubuntu / macOS. Aufruf im Repo-Root:  bash setup.sh
set -euo pipefail
cd "$(dirname "$0")"

PY=${PYTHON:-python3}
echo "== Python: $($PY --version)"
$PY -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip wheel

echo "== PyTorch (CPU-Variante, spart ~2 GB gegenueber der CUDA-Version)"
pip install torch --index-url https://download.pytorch.org/whl/cpu

echo "== Python-Pakete"
pip install -r requirements.txt

echo "== Jupyter-Kernel registrieren"
python -m ipykernel install --user --name rag-schulung --display-name "Python (rag-schulung)"

if [ ! -f .env ]; then cp .env.example .env; echo "== .env angelegt - bitte OPENAI_API_KEY eintragen"; fi

echo "== Modelle vorab herunterladen (ca. 3 GB, einmalig)"
python scripts/download_models.py || echo "WARNUNG: Modell-Download unvollstaendig - wird beim ersten Lab nachgeholt"

echo "== Smoke-Test (offline)"
FAKE_EMBEDDINGS=1 python smoke_test.py

echo
echo "Setup OK. Starten mit:  source .venv/bin/activate && jupyter lab"
