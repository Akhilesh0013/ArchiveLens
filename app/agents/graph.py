"""LangGraph wiring for marriage-record extraction agents."""

from __future__ import annotations

from functools import lru_cache

from app.agents.nodes import (
    MAX_RETRIES,
    extract_node,
    finalize_node,
    judge_node,
    retry_node,
    validate_node,
)
from app.agents.state import GraphState
from langgraph.graph import END, StateGraph


def _after_judge(state: GraphState) -> str:
    if state.get("judge_verdict") == "pass":
        return "finalize"
    if state.get("retries", 0) >= MAX_RETRIES:
        return "finalize"
    return "retry"


def build_graph(*, use_judge: bool = True):
    """Build and compile the extraction agent graph.

    When ``use_judge`` is False (bulk ingest / upload), the pipeline is
    extract → validate → finalize with no judge LLM call or retries.
    Gold eval keeps ``use_judge=True``.
    """
    graph = StateGraph(GraphState)

    graph.add_node("extract", extract_node)
    graph.add_node("validate", validate_node)
    graph.add_node("finalize", finalize_node)

    graph.set_entry_point("extract")
    graph.add_edge("extract", "validate")

    if use_judge:
        graph.add_node("judge", judge_node)
        graph.add_node("retry", retry_node)
        graph.add_edge("validate", "judge")
        graph.add_conditional_edges(
            "judge",
            _after_judge,
            {
                "finalize": "finalize",
                "retry": "retry",
            },
        )
        graph.add_edge("retry", "extract")
    else:
        graph.add_edge("validate", "finalize")

    graph.add_edge("finalize", END)
    return graph.compile()


@lru_cache(maxsize=2)
def _get_app(use_judge: bool):
    return build_graph(use_judge=use_judge)


def run_extraction_agent(
    ocr_text: str,
    *,
    source_pdf: str | None = None,
    source_page: int | None = None,
    ocr_engine: str | None = None,
    use_judge: bool = True,
) -> GraphState:
    """Run the extraction pipeline (with optional judge for gold eval)."""
    app = _get_app(use_judge)
    return app.invoke(
        {
            "ocr_text": ocr_text,
            "source_pdf": source_pdf,
            "source_page": source_page,
            "ocr_engine": ocr_engine,
            "retries": 0,
            "validation_errors": [],
        }
    )
