import importlib
import pytest


@pytest.fixture()
def wiki_env(tmp_path, monkeypatch):
    """Set WIKI_DIR to a temp path and reload all dependent modules.
    Returns the Path to the wiki directory."""
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    monkeypatch.setenv("WIKI_DIR", str(wiki))

    import agent.config as cfg;      importlib.reload(cfg)
    import agent.tools as tools;     importlib.reload(tools)
    import agent.tools_lc as tlc;    importlib.reload(tlc)
    import agent.nodes as nodes;     importlib.reload(nodes)
    import agent.graph as graph_mod; importlib.reload(graph_mod)

    return wiki


@pytest.fixture()
def base_state():
    """Return a factory for minimal WikiState dicts."""
    def _make(operation: str, input_: str = "test", wiki_index: str = "") -> dict:
        return {
            "operation": operation,
            "input": input_,
            "messages": [],
            "wiki_index": wiki_index,
            "output": "",
        }
    return _make
