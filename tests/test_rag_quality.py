"""RAG-Qualitaetstests fuer CI/CD (Lab 5).

Drei Ebenen:
  1. test_retrieval_*   deterministisch, ohne API-Key (BM25 / Sparse) -> laeuft bei jedem Push
  2. test_e2e_*         End-to-End mit LLM + Judge auf einem festen Subset -> nur mit OPENAI_API_KEY
  3. test_deepeval_*    DeepEval-Metriken (Faithfulness, Answer Relevancy) per assert_test -> nur mit Key

Aufruf:
  pytest tests -q                      # alles, LLM-Tests werden ohne Key uebersprungen
  pytest tests -q -k retrieval         # nur Ebene 1
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ragkurs import HybridIndex, PipelineConfig, RAGPipeline, chunk_by_headings, load_corpus  # noqa: E402
from ragkurs.config import settings  # noqa: E402
from ragkurs.eval import evaluate_retrieval, load_golden, retrieval_summary, run_golden  # noqa: E402

HAS_KEY = bool(settings.openai_api_key) and not settings.fake_embeddings
needs_llm = pytest.mark.skipif(not HAS_KEY, reason="OPENAI_API_KEY nicht gesetzt - LLM-Tests uebersprungen")

# Fester Subset fuer stabile, guenstige Regressionstests (IDs bewusst gemischt: faktisch, tabelle, near-miss, negativ)
REGRESSION_IDS = ["g01", "g06", "g13", "g18", "g22", "g26", "g33", "g36", "g40", "g47"]


@pytest.fixture(scope="session")
def docs():
    return load_corpus()


@pytest.fixture(scope="session")
def sparse_pipeline(docs):
    # Sparse/BM25 braucht keine Embeddings -> laeuft ohne API-Key und ohne Fake-Modus
    os.environ.setdefault("FAKE_EMBEDDINGS", "1")
    index = HybridIndex(collection="ci_sparse").build(chunk_by_headings(docs))
    return RAGPipeline(index, PipelineConfig(retrieval="sparse", k=5, name="ci-sparse"))


# ---------------------------------------------------------------- Ebene 1
def test_retrieval_hit_rate_sparse(sparse_pipeline):
    golden = load_golden(answerable=True)
    summary = retrieval_summary(evaluate_retrieval(sparse_pipeline, golden, ["employee"]))
    assert summary["hit_rate"] >= 0.80, f"Hit-Rate@5 eingebrochen: {summary}"


def test_retrieval_acl_no_leak(sparse_pipeline):
    """Ein Mitarbeiter ohne HR-Rolle darf die Gehaltsbaender nie im Kontext haben."""
    for g in load_golden(types=["acl"]):
        res = sparse_pipeline.retrieve_only(g["question"], ["employee"])
        protected = set(res.retrieved_doc_ids) & set(g["source_docs"])
        assert not protected, f"ACL-Leak bei {g['id']}: {protected}"


def test_retrieval_superseded_excluded(sparse_pipeline):
    res = sparse_pipeline.retrieve_only("Hotel Hoechstbetrag Nacht Deutschland Reisekosten", ["employee"])
    assert "hr-reisekosten-2024" not in res.retrieved_doc_ids


# ---------------------------------------------------------------- Ebene 2
@pytest.fixture(scope="session")
def prod_pipeline(docs):
    index = HybridIndex(collection="ci_prod").build(chunk_by_headings(docs))
    return RAGPipeline(index, PipelineConfig(retrieval="hybrid", k=5, prefetch_k=20, name="ci-prod"))


@needs_llm
def test_e2e_regression_accuracy(prod_pipeline):
    golden = load_golden(ids=REGRESSION_IDS)
    df = run_golden(prod_pipeline, golden, ["employee"], verbose=False)
    acc = df["correct"].mean()
    failed = df[~df["correct"]][["id", "failure"]].to_dict("records")
    assert acc >= 0.8, f"Korrektheit {acc:.2f} < 0.80 - Regressionen: {failed}"


@needs_llm
def test_e2e_refuses_unanswerable(prod_pipeline):
    for g in load_golden(answerable=False):
        res = prod_pipeline.run(g["question"], ["employee"])
        assert res.answer.is_no_answer, f"{g['id']}: haette absagen muessen, antwortete: {res.answer.text[:120]}"


# ---------------------------------------------------------------- Ebene 3
@needs_llm
@pytest.mark.parametrize("gid", ["g01", "g22", "g36"])
def test_deepeval_faithfulness_and_relevancy(prod_pipeline, gid):
    from deepeval import assert_test
    from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
    from deepeval.models import GPTModel
    from deepeval.test_case import LLMTestCase

    g = load_golden(ids=[gid])[0]
    res = prod_pipeline.run(g["question"], ["employee"])
    judge = GPTModel(model=settings.judge_model, api_key=settings.openai_api_key)
    case = LLMTestCase(
        input=g["question"],
        actual_output=res.answer.text,
        expected_output=g["ground_truth"],
        retrieval_context=[h.chunk.text for h in res.answer.sources],
    )
    assert_test(case, [FaithfulnessMetric(threshold=0.7, model=judge), AnswerRelevancyMetric(threshold=0.6, model=judge)])
