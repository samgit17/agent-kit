from langgraph.graph import StateGraph, END
from .models import ResearchState
from .nodes import (
    planner_node,
    searcher_node,
    synthesiser_node,
    verifier_node,
    should_retry,
    formatter_node,
)
from program_parser import ProgramConfig


def build_graph(cfg: ProgramConfig):
    g = StateGraph(ResearchState)

    g.add_node("planner", planner_node)
    g.add_node("searcher", searcher_node)
    g.add_node("synthesiser", synthesiser_node)
    g.add_node("verifier", verifier_node)
    g.add_node("formatter", formatter_node)

    g.set_entry_point("planner")
    g.add_edge("planner", "searcher")
    g.add_edge("searcher", "synthesiser")
    g.add_edge("synthesiser", "verifier")
    g.add_conditional_edges("verifier", should_retry, {"retry": "synthesiser", "format": "formatter"})
    g.add_edge("formatter", END)

    return g.compile()


def build_initial_state(cfg: ProgramConfig) -> dict:
    return {
        "goal": cfg.goal,
        "success_criteria": cfg.success_criteria,
        "max_iterations": cfg.web.max_iterations,
        "confidence_threshold": cfg.web.confidence_threshold,
    }
