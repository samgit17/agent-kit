"""
models.py — State model for voice research agent.
"""

from __future__ import annotations
from pydantic import BaseModel, Field


class ResearchState(BaseModel):
    query: str
    max_iterations: int = 3
    confidence_threshold: float = 0.6

    queries: list[str] = Field(default_factory=list)
    search_results: list[dict] = Field(default_factory=list)
    synthesis: str = ""
    confidence: float = 0.0
    verifier_gaps: list[str] = Field(default_factory=list)
    iterations: int = 0
    report: str = ""
