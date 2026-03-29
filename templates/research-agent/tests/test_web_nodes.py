"""
tests/test_web_nodes.py — Tests for web backend node behaviour.

Run: python -m pytest tests/test_web_nodes.py -v
"""

import pytest
from pathlib import Path


def test_confidence_threshold_parsed_from_program_md(tmp_path):
    from program_parser import parse_program
    p = tmp_path / "program.md"
    p.write_text("## backend\nweb\n\n## goal\nTest\n\n## constraints\nconfidence_threshold: 0.5\n")
    cfg = parse_program(p)
    assert cfg.web.confidence_threshold == 0.5


def test_default_confidence_threshold():
    from program_parser import ProgramConfig
    cfg = ProgramConfig()
    assert cfg.web.confidence_threshold == 0.6


def test_should_retry_uses_threshold():
    from backends.web.models import ResearchState
    from backends.web.nodes import should_retry

    state = ResearchState(
        goal="test", success_criteria=[], max_iterations=3,
        confidence_threshold=0.6, confidence=0.4,
        search_results=[{"title": "x", "content": "y"}],
        iterations=1,
    )
    assert should_retry(state) == "retry"


def test_should_not_retry_when_above_threshold():
    from backends.web.models import ResearchState
    from backends.web.nodes import should_retry

    state = ResearchState(
        goal="test", success_criteria=[], max_iterations=3,
        confidence_threshold=0.6, confidence=0.7,
        search_results=[{"title": "x", "content": "y"}],
        iterations=1,
    )
    assert should_retry(state) == "format"


def test_should_not_retry_when_no_results():
    from backends.web.models import ResearchState
    from backends.web.nodes import should_retry

    state = ResearchState(
        goal="test", success_criteria=[], max_iterations=3,
        confidence_threshold=0.6, confidence=0.1,
        search_results=[],
        iterations=1,
    )
    assert should_retry(state) == "format"


def test_should_not_retry_when_max_iterations_reached():
    from backends.web.models import ResearchState
    from backends.web.nodes import should_retry

    state = ResearchState(
        goal="test", success_criteria=[], max_iterations=3,
        confidence_threshold=0.6, confidence=0.1,
        search_results=[{"title": "x", "content": "y"}],
        iterations=3,
    )
    assert should_retry(state) == "format"


def test_confidence_threshold_passed_to_initial_state():
    from program_parser import ProgramConfig, WebConfig
    from backends.web.graph import build_initial_state

    cfg = ProgramConfig(goal="test", backend="web")
    cfg.web = WebConfig(max_iterations=2, confidence_threshold=0.5)
    state = build_initial_state(cfg)
    assert state["confidence_threshold"] == 0.5
