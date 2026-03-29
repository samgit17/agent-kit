"""
tests/test_run_config.py — Tests for run.py config output and report file save.

Run: python -m pytest tests/test_run_config.py -v
"""

import os
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


def _make_web_cfg():
    from program_parser import ProgramConfig, WebConfig
    cfg = ProgramConfig(backend="web", goal="Test goal")
    cfg.web = WebConfig(max_iterations=1)
    return cfg


def test_config_echo_shows_backend_and_providers(tmp_path):
    """run.py must print LLM provider and search provider before invoking graph."""
    from rich.console import Console
    import io

    buf = io.StringIO()
    fake_console = Console(file=buf, highlight=False)

    cfg = _make_web_cfg()
    fake_graph = MagicMock()
    fake_graph.invoke.return_value = {"report": "# Done"}

    with patch("backends.log.console", fake_console), \
         patch("run.console", fake_console), \
         patch("run.parse_program", return_value=cfg), \
         patch("run._load_backend", return_value=(lambda c: fake_graph, lambda c: {})), \
         patch.dict(os.environ, {"LLM_PROVIDER": "ollama", "SEARCH_PROVIDER": "duckduckgo"}):
        import run
        run.main()

    output = buf.getvalue()
    assert "ollama" in output.lower()
    assert "duckduckgo" in output.lower()


def test_report_saved_to_file(tmp_path):
    """Web backend report must be written to output/report.md."""
    cfg = _make_web_cfg()
    fake_graph = MagicMock()
    fake_graph.invoke.return_value = {"report": "# My Report"}

    output_file = tmp_path / "report.md"

    with patch("run.parse_program", return_value=cfg), \
         patch("run._load_backend", return_value=(lambda c: fake_graph, lambda c: {})), \
         patch("run.REPORT_PATH", output_file), \
         patch("run.console", MagicMock()):
        import run
        run.main()

    assert output_file.exists()
    assert "# My Report" in output_file.read_text()
