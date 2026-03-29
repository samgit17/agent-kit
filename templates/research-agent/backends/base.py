"""
backends/base.py — Protocol that every backend must satisfy.

run.py calls build_graph() and build_initial_state() without knowing
which backend it's talking to.
"""

from __future__ import annotations
from typing import Protocol, Any
from program_parser import ProgramConfig


class BackendProtocol(Protocol):
    def build_graph(self, cfg: ProgramConfig) -> Any:
        """Return a compiled LangGraph graph."""
        ...

    def build_initial_state(self, cfg: ProgramConfig) -> dict:
        """Return the initial state dict to invoke the graph with."""
        ...
