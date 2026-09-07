"""Observability mit Arize Phoenix (lokal, Open Source, kein Account).

    from ragkurs.tracing import start_phoenix, traced_run
    url = start_phoenix()            # startet die UI unter http://localhost:6006
    result = traced_run(pipe, "Frage?", ["employee"])

LangChain- und OpenAI-Aufrufe werden automatisch instrumentiert (OpenInference);
zusaetzlich legt `traced_run` einen Eltern-Span "rag.run" mit unseren Pipeline-Attributen an.
"""
from __future__ import annotations

import json

_tracer_provider = None
_session = None


def start_phoenix(project_name: str = "rag-schulung", port: int = 6006) -> str:
    global _tracer_provider, _session
    import phoenix as px
    from phoenix.otel import register

    if _session is None:
        _session = px.launch_app(port=port)
    if _tracer_provider is None:
        _tracer_provider = register(project_name=project_name, endpoint=f"http://localhost:{port}/v1/traces", auto_instrument=True)
    return str(_session.url)


def get_tracer(name: str = "ragkurs"):
    if _tracer_provider is None:
        raise RuntimeError("Erst start_phoenix() aufrufen.")
    return _tracer_provider.get_tracer(name)


def traced_run(pipeline, query: str, user_roles: list[str] | None = None):
    """Fuehrt pipeline.run in einem Eltern-Span aus und haengt Trace-Infos als Attribute an."""
    from openinference.semconv.trace import SpanAttributes

    tracer = get_tracer()
    with tracer.start_as_current_span("rag.run") as span:
        span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, "CHAIN")
        span.set_attribute(SpanAttributes.INPUT_VALUE, query)
        span.set_attribute("rag.config", pipeline.config.label())
        span.set_attribute("rag.user_roles", json.dumps(user_roles or []))
        result = pipeline.run(query, user_roles)
        t = result.trace
        span.set_attribute(SpanAttributes.OUTPUT_VALUE, result.answer.text)
        span.set_attribute("rag.retrieved_docs", json.dumps(result.retrieved_doc_ids))
        span.set_attribute("rag.top_score", float(result.hits[0].score) if result.hits else 0.0)
        for key in ("t_retrieval_ms", "t_rerank_ms", "t_generation_ms", "t_total_ms", "cost_usd", "context_chars"):
            if key in t:
                span.set_attribute(f"rag.{key}", float(t[key]))
        span.set_attribute("rag.no_answer", bool(result.answer.is_no_answer))
        return result


def log_to_dataframe(pipeline):
    """Einfache 'Produktions-Auswertung' aus dem Pipeline-Log ohne Phoenix (Lab 7)."""
    import pandas as pd

    rows = []
    for r in pipeline.log:
        t = r.trace
        rows.append(
            {
                "query": r.query,
                "no_answer": r.answer.is_no_answer,
                "top_score": r.hits[0].score if r.hits else None,
                "n_docs": len(r.retrieved_doc_ids),
                "t_retrieval_ms": round(t.get("t_retrieval_ms", 0)),
                "t_rerank_ms": round(t.get("t_rerank_ms", 0)),
                "t_generation_ms": round(t.get("t_generation_ms", 0)),
                "t_total_ms": round(t.get("t_total_ms", 0)),
                "input_tokens": t.get("usage", {}).get("input_tokens", 0),
                "output_tokens": t.get("usage", {}).get("output_tokens", 0),
                "cost_usd": t.get("cost_usd", 0),
                "cache": t.get("cache", "-"),
            }
        )
    return pd.DataFrame(rows)
