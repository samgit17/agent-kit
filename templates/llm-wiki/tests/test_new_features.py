"""
Tests for new features: URL ingestion and --save compound loop.
No live network calls, no live LLM calls.
"""
import json
import pytest
from unittest.mock import patch, MagicMock
from langchain_core.messages import AIMessage

from agent.tools import fetch_url, _is_likely_dynamic, _extract_text
from agent.tools_lc import WIKI_TOOLS
from agent.nodes import build_ingest_node, build_query_node
from agent.state import WikiState
from agent.exceptions import MaxIterationsError


# ---------------------------------------------------------------------------
# Fake model
# ---------------------------------------------------------------------------

class FakeModel:
    def __init__(self, responses):
        self._responses = iter(responses)

    def invoke(self, messages):
        return next(self._responses)

    def bind_tools(self, tools):
        return self


def _tool_map():
    return {t.name: t for t in WIKI_TOOLS}


def _base_state(**overrides) -> WikiState:
    state = {
        "operation": "ingest",
        "input": "",
        "messages": [],
        "wiki_index": "",
        "pages_read": [],
        "output": "",
        "fetched_content": "",
        "save_output": False,
    }
    state.update(overrides)
    return state


# ---------------------------------------------------------------------------
# fetch_url — unit tests (no network)
# ---------------------------------------------------------------------------

def test_is_likely_dynamic_empty_body():
    html = "<html><body></body></html>"
    assert _is_likely_dynamic(html) is True


def test_is_likely_dynamic_rich_body():
    html = "<html><body>" + ("word " * 100) + "</body></html>"
    assert _is_likely_dynamic(html) is False


def test_extract_text_prefers_article():
    html = """
    <html><body>
      <nav>Nav junk</nav>
      <article>Main content here</article>
      <footer>Footer junk</footer>
    </body></html>
    """
    text = _extract_text(html)
    assert "Main content here" in text
    assert "Nav junk" not in text


def test_fetch_url_httpx_success():
    rich_html = "<html><body><article>" + ("word " * 100) + "</article></body></html>"
    with patch("agent.tools._fetch_with_httpx", return_value=(200, rich_html)):
        result = fetch_url("https://example.com/article")
    assert "word" in result
    assert "[fetch_url error]" not in result


def test_fetch_url_httpx_dynamic_falls_back_to_playwright():
    sparse_html = "<html><body><div>hi</div></body></html>"
    playwright_html = "<html><body><article>" + ("word " * 100) + "</article></body></html>"
    with patch("agent.tools._fetch_with_httpx", return_value=(200, sparse_html)), \
         patch("agent.tools._fetch_with_playwright", return_value=playwright_html):
        result = fetch_url("https://example.com/spa")
    assert "word" in result


def test_fetch_url_httpx_non200_falls_back_to_playwright():
    playwright_html = "<html><body><article>" + ("word " * 100) + "</article></body></html>"
    with patch("agent.tools._fetch_with_httpx", return_value=(403, "")), \
         patch("agent.tools._fetch_with_playwright", return_value=playwright_html):
        result = fetch_url("https://example.com/gated")
    assert "word" in result


def test_fetch_url_both_fail_returns_error_string():
    with patch("agent.tools._fetch_with_httpx", side_effect=Exception("timeout")), \
         patch("agent.tools._fetch_with_playwright", side_effect=Exception("launch failed")):
        result = fetch_url("https://example.com/broken")
    assert result.startswith("[fetch_url error]")


# ---------------------------------------------------------------------------
# ingest_node — URL path (fetched_content populated)
# ---------------------------------------------------------------------------

def test_ingest_node_uses_fetched_content_inline():
    """When fetched_content is set, HumanMessage contains content inline."""
    captured = []

    class CapturingModel:
        def invoke(self, messages):
            captured.extend(messages)
            return AIMessage(content="Ingested.")

    node = build_ingest_node(CapturingModel(), _tool_map())
    state = _base_state(
        operation="ingest",
        input="https://example.com/article",
        fetched_content="This is the fetched article content.",
    )
    result = node(state)
    assert result["output"] == "Ingested."
    from langchain_core.messages import HumanMessage
    human_msgs = [m for m in captured if isinstance(m, HumanMessage)]
    assert any("This is the fetched article content." in m.content for m in human_msgs)


def test_ingest_node_file_path_references_path():
    """When fetched_content is empty, HumanMessage references the file path."""
    captured = []

    class CapturingModel:
        def invoke(self, messages):
            captured.extend(messages)
            return AIMessage(content="Ingested.")

    node = build_ingest_node(CapturingModel(), _tool_map())
    state = _base_state(
        operation="ingest",
        input="/some/local/file.md",
        fetched_content="",
    )
    result = node(state)
    from langchain_core.messages import HumanMessage
    human_msgs = [m for m in captured if isinstance(m, HumanMessage)]
    assert any("/some/local/file.md" in m.content for m in human_msgs)
    assert not any("Content:\n" in m.content for m in human_msgs)


# ---------------------------------------------------------------------------
# query_node — --save compound loop
# ---------------------------------------------------------------------------

def test_query_node_no_save_flag_skips_decision():
    model = FakeModel([AIMessage(content="Here is the answer.")])
    node = build_query_node(model, _tool_map())
    state = _base_state(operation="query", input="What is X?", save_output=False)
    result = node(state)
    assert result["output"] == "Here is the answer."


def test_query_node_save_flag_new_page(tmp_path, monkeypatch):
    monkeypatch.setenv("WIKI_DIR", str(tmp_path / "wiki"))
    (tmp_path / "wiki" / "answers").mkdir(parents=True)

    decision = json.dumps({"action": "new", "target": None, "title": "Test Answer"})

    model = FakeModel([
        AIMessage(content="The answer is 42."),
        AIMessage(content=decision),
        AIMessage(content="Filed."),
    ])
    node = build_query_node(model, _tool_map())
    state = _base_state(operation="query", input="What is X?", save_output=True)
    result = node(state)
    assert result["output"] == "The answer is 42."


def test_query_node_save_flag_bad_json_skips_gracefully():
    model = FakeModel([
        AIMessage(content="The answer is 42."),
        AIMessage(content="not valid json"),
    ])
    node = build_query_node(model, _tool_map())
    state = _base_state(operation="query", input="What is X?", save_output=True)
    result = node(state)
    assert result["output"] == "The answer is 42."


# ---------------------------------------------------------------------------
# run.py — _is_url helper
# ---------------------------------------------------------------------------

def test_is_url_detection():
    from run import _is_url
    assert _is_url("https://example.com") is True
    assert _is_url("http://example.com") is True
    assert _is_url("sources/my-file.md") is False
    assert _is_url("/absolute/path.md") is False
