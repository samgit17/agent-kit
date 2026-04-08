"""
tests/test_diagrams.py — Tests for the diagram generation skill.

All tests are pure-logic or use monkeypatched LLM — no live model calls.
Run: python -m pytest tests/test_diagrams.py -v
"""

import pytest
from backends.web.models import ResearchState


# ---------------------------------------------------------------------------
# is_enabled
# ---------------------------------------------------------------------------

def test_is_enabled_false_by_default(monkeypatch):
    monkeypatch.delenv("DIAGRAMS_ENABLED", raising=False)
    from backends.web.diagrams import is_enabled
    assert is_enabled() is False


def test_is_enabled_true_when_set(monkeypatch):
    monkeypatch.setenv("DIAGRAMS_ENABLED", "true")
    from backends.web.diagrams import is_enabled
    assert is_enabled() is True


def test_is_enabled_accepts_1(monkeypatch):
    monkeypatch.setenv("DIAGRAMS_ENABLED", "1")
    from backends.web.diagrams import is_enabled
    assert is_enabled() is True


def test_is_enabled_rejects_yes(monkeypatch):
    monkeypatch.setenv("DIAGRAMS_ENABLED", "yes")
    from backends.web.diagrams import is_enabled
    assert is_enabled() is False


# ---------------------------------------------------------------------------
# _xml_to_drawio_url
# ---------------------------------------------------------------------------

def test_make_url_returns_drawio_url():
    from backends.web.diagrams import _xml_to_drawio_url
    xml = "<mxGraphModel><root><mxCell id='0'/></root></mxGraphModel>"
    url = _xml_to_drawio_url(xml)
    assert url.startswith("https://app.diagrams.net/#R")
    assert len(url) > len("https://app.diagrams.net/#R")


def test_make_url_is_deterministic():
    from backends.web.diagrams import _xml_to_drawio_url
    xml = "<mxGraphModel><root></root></mxGraphModel>"
    assert _xml_to_drawio_url(xml) == _xml_to_drawio_url(xml)


def test_make_url_differs_for_different_xml():
    from backends.web.diagrams import _xml_to_drawio_url
    url_a = _xml_to_drawio_url("<mxGraphModel><root><mxCell id='0'/></root></mxGraphModel>")
    url_b = _xml_to_drawio_url("<mxGraphModel><root><mxCell id='1'/></root></mxGraphModel>")
    assert url_a != url_b


# ---------------------------------------------------------------------------
# _strip_xml_fence
# ---------------------------------------------------------------------------

def test_strip_xml_fence_passthrough():
    from backends.web.diagrams import _strip_xml_fence
    xml = "<mxGraphModel/>"
    assert _strip_xml_fence(xml) == "<mxGraphModel/>"


def test_strip_xml_fence_removes_xml_fence():
    from backends.web.diagrams import _strip_xml_fence
    assert _strip_xml_fence("```xml\n<mxGraphModel/>\n```") == "<mxGraphModel/>"


def test_strip_xml_fence_removes_plain_fence():
    from backends.web.diagrams import _strip_xml_fence
    assert _strip_xml_fence("```\n<mxGraphModel/>\n```") == "<mxGraphModel/>"


# ---------------------------------------------------------------------------
# diagram_node
# ---------------------------------------------------------------------------

def _make_state(**kwargs) -> ResearchState:
    defaults = dict(goal="test goal", success_criteria=[], max_iterations=3, synthesis="Some findings.")
    return ResearchState(**{**defaults, **kwargs})


def _fake_llm(xml: str):
    """Returns a get_llm() replacement whose invoke() yields xml as content."""
    class FakeResponse:
        content = xml

    class FakeLLM:
        def invoke(self, messages):
            return FakeResponse()

    return lambda: FakeLLM()


def test_diagram_node_returns_diagram_url(monkeypatch):
    from backends.web import diagrams
    xml = "<mxGraphModel><root><mxCell id='0'/></root></mxGraphModel>"
    monkeypatch.setattr(diagrams, "get_llm", _fake_llm(xml))
    result = diagrams.diagram_node(_make_state())
    assert "diagram_url" in result
    assert result["diagram_url"].startswith("https://app.diagrams.net/#R")


def test_diagram_node_strips_fence_from_llm_response(monkeypatch):
    from backends.web import diagrams
    xml = "<mxGraphModel><root></root></mxGraphModel>"
    monkeypatch.setattr(diagrams, "get_llm", _fake_llm(f"```xml\n{xml}\n```"))
    result = diagrams.diagram_node(_make_state())
    assert result["diagram_url"].startswith("https://app.diagrams.net/#R")


def test_diagram_node_uses_goal_and_synthesis(monkeypatch):
    from backends.web import diagrams

    captured = {}

    class FakeResponse:
        content = "<mxGraphModel/>"

    class CapturingLLM:
        def invoke(self, messages):
            captured["prompt"] = messages[0].content
            return FakeResponse()

    monkeypatch.setattr(diagrams, "get_llm", lambda: CapturingLLM())
    diagrams.diagram_node(_make_state(goal="AI security", synthesis="Key finding: zero trust matters."))
    assert "AI security" in captured["prompt"]
    assert "Key finding: zero trust matters." in captured["prompt"]
