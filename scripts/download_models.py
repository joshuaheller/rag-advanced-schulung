"""Laedt alle Open-Source-Modelle vorab (damit im Kurs keine Wartezeiten entstehen).

- cross-encoder/mmarco-mMiniLMv2-L12-H384-v1  (~470 MB)  schneller multilingualer Reranker
- BAAI/bge-reranker-v2-m3                      (~2.2 GB)  Qualitaets-Reranker
- answerdotai/answerai-colbert-small-v1        (~130 MB)  Late-Interaction-Reranker
- Docling Layout-/Tabellenmodelle              (~500 MB)  PDF-Parsing
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ragkurs.config import settings  # noqa: E402  (setzt Telemetrie-Opt-outs)

ok = True

def step(name, fn):
    global ok
    print(f"-> {name} ...", flush=True)
    try:
        fn()
        print("   ok")
    except Exception as e:  # noqa: BLE001
        ok = False
        print(f"   FEHLER: {e}")


def cross_encoders():
    from sentence_transformers import CrossEncoder
    CrossEncoder(settings.reranker_fast, max_length=512)
    CrossEncoder(settings.reranker_quality, max_length=512)


def colbert():
    from rerankers import Reranker
    Reranker(settings.colbert_model, model_type="colbert", verbose=0)


def docling():
    from docling.document_converter import DocumentConverter
    pdf = next(settings.corpus_dir.glob("*.pdf"))
    DocumentConverter().convert(str(pdf))


step("Cross-Encoder-Reranker", cross_encoders)
step("ColBERT-Reranker", colbert)
step("Docling (PDF-Parsing, Testkonvertierung)", docling)
print("\nAlle Modelle geladen." if ok else "\nEinzelne Downloads fehlgeschlagen - Skript spaeter erneut ausfuehren.")
sys.exit(0 if ok else 1)
