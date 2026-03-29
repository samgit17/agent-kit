"""
tests/test_log.py — Tests for backends/log.py

Run: python -m pytest tests/test_log.py -v
"""

import io
import sys
import pytest


def test_strip_json_fence_removes_think_blocks():
    from backends.llm import strip_json_fence
    raw = '<think>\nLet me think about this.\n</think>\n```json\n{"confidence": 0.9}\n```'
    result = strip_json_fence(raw)
    assert "<think>" not in result
    assert result == '{"confidence": 0.9}'


def test_strip_json_fence_no_think_block():
    from backends.llm import strip_json_fence
    raw = '```json\n["query one", "query two"]\n```'
    result = strip_json_fence(raw)
    assert result == '["query one", "query two"]'
    from backends.log import log
    log("[planner]", "Generated 3 queries")


def test_log_with_style_does_not_raise():
    from backends.log import log
    log("[searcher]", "query failed: 'foo'", style="red")


def test_console_is_singleton():
    from backends.log import console as c1
    from backends.log import console as c2
    assert c1 is c2


def test_log_tag_and_msg_appear_in_output():
    import io
    from rich.console import Console
    import backends.log as log_mod

    buf = io.StringIO()
    fake_console = Console(file=buf, highlight=False)
    original = log_mod.console
    log_mod.console = fake_console
    try:
        log_mod.log("[planner]", "Generated 3 queries")
    finally:
        log_mod.console = original

    output = buf.getvalue()
    assert "[planner]" in output
    assert "Generated 3 queries" in output
