# Setup fuer Windows (PowerShell). Aufruf im Repo-Root:  powershell -ExecutionPolicy Bypass -File setup.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "== Python:" (python --version)
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip wheel

Write-Host "== PyTorch (CPU)"
pip install "torch==2.14.0" --index-url https://download.pytorch.org/whl/cpu

Write-Host "== Python-Pakete"
pip install -r requirements.txt -c constraints.txt

Write-Host "== Jupyter-Kernel registrieren"
python -m ipykernel install --user --name rag-schulung --display-name "Python (rag-schulung)"

if (-not (Test-Path .env)) { Copy-Item .env.example .env; Write-Host "== .env angelegt - bitte OPENAI_API_KEY eintragen" }

Write-Host "== Modelle vorab herunterladen (ca. 3 GB, einmalig)"
try { python scripts/download_models.py } catch { Write-Host "WARNUNG: Modell-Download unvollstaendig - wird beim ersten Lab nachgeholt" }

Write-Host "== Smoke-Test (offline)"
$env:FAKE_EMBEDDINGS = "1"
python smoke_test.py
Remove-Item Env:FAKE_EMBEDDINGS

Write-Host ""
Write-Host "Setup OK. Starten mit:  .\.venv\Scripts\Activate.ps1 ; jupyter lab"
