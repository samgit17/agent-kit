from __future__ import annotations
from pydantic import BaseModel, Field


class ResearchState(BaseModel):
    goal: str
    success_criteria: list[str]
    max_iterations: int
    confidence_threshold: float = 0.6

    queries: list[str] = Field(default_factory=list)
    search_results: list[dict] = Field(default_factory=list)
    synthesis: str = ""
    confidence: float = 0.0
    verifier_gaps: list[str] = Field(default_factory=list)
    iterations: int = 0
    report: str = ""
    diagram_url: str | None = None  # populated by diagram_node when DIAGRAMS_ENABLED=true
