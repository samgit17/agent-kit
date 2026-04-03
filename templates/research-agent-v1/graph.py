"""
graph.py — LangGraph state graph for the Research Agent.

Flow:
  planner → searcher → synthesiser → verifier
                                         ├── (confidence ok) → formatter → END
                                         └── (low confidence, retry < 1) → synthesiser
"""

from langgraph.graph import StateGraph, END
from models import ResearchState
from agents import (
    planner_node,
    searcher_node,
    synthesiser_node,
    verifier_node,
    should_rewrite,
    formatter_node,
)


def build_graph() -> StateGraph:
    graph = StateGraph(ResearchState)

    # ── Nodes ────────────────────────────────────────────────
    graph.add_node("planner",     planner_node)
    graph.add_node("searcher",    searcher_node)
    graph.add_node("synthesiser", synthesiser_node)
    graph.add_node("verifier",    verifier_node)
    graph.add_node("formatter",   formatter_node)

    # ── Edges ─────────────────────────────────────────────────
    graph.set_entry_point("planner")
    graph.add_edge("planner",     "searcher")
    graph.add_edge("searcher",    "synthesiser")
    graph.add_edge("synthesiser", "verifier")

    # Conditional: retry once if confidence is too low
    graph.add_conditional_edges(
        "verifier",
        should_rewrite,
        {
            "retry":  "synthesiser",   # resynthesize with same results
            "format": "formatter",
        },
    )

    graph.add_edge("formatter", END)

    return graph.compile()


# Compiled graph — import and call directly in run.py
research_graph = build_graph()
