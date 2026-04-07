import os

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

from agent.nodes import build_ingest_node, build_query_node, build_lint_node
from agent.state import WikiState
from agent.tools_lc import WIKI_TOOLS


def build_model() -> ChatOpenAI:
    return ChatOpenAI(
        model=os.environ["LLM_MODEL"],
        api_key=os.environ["LLM_API_KEY"],
        base_url=os.getenv("LLM_BASE_URL"),
    ).bind_tools(WIKI_TOOLS)


def _router(state: WikiState) -> str:
    return state["operation"]


def build_graph(model=None):
    if model is None:
        model = build_model()

    tool_map = {t.name: t for t in WIKI_TOOLS}

    g = StateGraph(WikiState)
    g.add_node("ingest_node", build_ingest_node(model, tool_map))
    g.add_node("query_node", build_query_node(model, tool_map))
    g.add_node("lint_node", build_lint_node(model, tool_map))

    g.add_conditional_edges(START, _router, {
        "ingest": "ingest_node",
        "query":  "query_node",
        "lint":   "lint_node",
    })
    g.add_edge("ingest_node", END)
    g.add_edge("query_node", END)
    g.add_edge("lint_node", END)

    return g.compile()
