import pytest
from langchain_core.messages import AIMessage

from agent.graph import _router


# ---------------------------------------------------------------------------
# Fake model
# ---------------------------------------------------------------------------

class FakeModel:
    def __init__(self, response: str = "Done."):
        self._response = response

    def invoke(self, messages):
        msg = AIMessage(content=self._response)
        msg.tool_calls = []
        return msg

    def bind_tools(self, tools):
        return self


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

def test_router_ingest(base_state):
    assert _router(base_state("ingest")) == "ingest"


def test_router_query(base_state):
    assert _router(base_state("query")) == "query"


def test_router_lint(base_state):
    assert _router(base_state("lint")) == "lint"


# ---------------------------------------------------------------------------
# Graph structure
# ---------------------------------------------------------------------------

def test_graph_compiles():
    from agent.graph import build_graph
    assert build_graph(model=FakeModel()) is not None


def test_graph_nodes_present():
    from agent.graph import build_graph
    graph = build_graph(model=FakeModel())
    node_names = set(graph.get_graph().nodes.keys())
    assert {"ingest_node", "query_node", "lint_node"}.issubset(node_names)


# ---------------------------------------------------------------------------
# End-to-end routing with fake model
# ---------------------------------------------------------------------------

def test_graph_routes_ingest(wiki_env, base_state):
    from agent.graph import build_graph
    result = build_graph(model=FakeModel("Ingest done.")).invoke(base_state("ingest"))
    assert result["output"] == "Ingest done."


def test_graph_routes_query(wiki_env, base_state):
    from agent.graph import build_graph
    result = build_graph(model=FakeModel("Query answer.")).invoke(base_state("query"))
    assert result["output"] == "Query answer."


def test_graph_routes_lint(wiki_env, base_state):
    from agent.graph import build_graph
    result = build_graph(model=FakeModel("2 issues found.")).invoke(base_state("lint"))
    assert result["output"] == "2 issues found." 
