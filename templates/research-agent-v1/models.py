"""
models.py — Pydantic models for Research Agent state and data types.
"""

from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    query: str
    url: str
    title: str
    content: str
    score: float = 0.0


class Citation(BaseModel):
    url: str
    title: str
    section: str = ""


class ResearchState(BaseModel):
    """LangGraph state — passed between every agent node."""

    # Input
    query: str = ""

    # Planner output
    search_queries: list[str] = Field(default_factory=list)

    # Searcher output
    search_results: list[SearchResult] = Field(default_factory=list)

    # Synthesiser output
    draft_answer: str = ""
    citations: list[Citation] = Field(default_factory=list)
    retries: int = 0                 # incremented by synthesiser_node; gates retry in should_rewrite

    # Verifier output
    confidence_score: float = 0.0
    uncertainty_flagged: bool = False
    verifier_feedback: str = ""

    # Formatter output
    final_report: str = ""

    # Control flow
    error: str = ""
