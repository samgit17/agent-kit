"""
agents/synthesiser.py — Reasons over search results and produces a draft answer with citations.
Fault-tolerant: if the LLM call fails, returns a plain concatenation of search snippets.
"""

import logging

from langchain_core.prompts import ChatPromptTemplate
from json_parser import ThinkingJsonOutputParser
from llm_client import get_llm
from models import ResearchState, Citation, SearchResult

logger = logging.getLogger("research_agent")

_SYSTEM = """You are a research synthesis agent. You are given a question and a set of
search results. Your job is to reason over the results and produce a comprehensive,
grounded answer with citations.

Return ONLY a JSON object in this exact format:
{{
  "answer": "your detailed answer here",
  "citations": [
    {{"url": "https://...", "title": "Source title", "section": "relevant quote or section"}}
  ]
}}

Rules:
- Only use information present in the search results — do not hallucinate
- Every factual claim must be supported by at least one citation
- If the search results don't contain enough information, say so explicitly
- Be comprehensive but concise"""

_HUMAN = """Question: {query}

Search Results:
{results}"""

_prompt = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM),
    ("human", _HUMAN),
])


def _format_results(results: list[SearchResult]) -> str:
    return "\n".join(
        f"[{i}] {r.title}\nURL: {r.url}\n{r.content}"
        for i, r in enumerate(results, 1)
    )


def _fallback_answer(state: ResearchState) -> dict:
    """Build a plain answer from raw snippets when LLM synthesis fails."""
    lines = ["Research results for: " + state.query, ""]
    citations = []
    for r in state.search_results[:5]:
        lines.append(f"• {r.title}: {r.content[:200]}")
        citations.append(Citation(url=r.url, title=r.title))
    return {
        "draft_answer": "\n".join(lines),
        "citations": citations,
        "retries": state.retries + 1,
    }


def synthesiser_node(state: ResearchState) -> dict:
    try:
        chain = _prompt | get_llm() | ThinkingJsonOutputParser()
        result = chain.invoke({
            "query": state.query,
            "results": _format_results(state.search_results),
        })
        answer = result.get("answer", "")
        if not answer:
            raise ValueError("Model returned empty answer")
        citations = [Citation(**c) for c in result.get("citations", [])]
        return {
            "draft_answer": answer,
            "citations": citations,
            "retries": state.retries + 1,
        }
    except Exception as e:
        logger.warning("Synthesiser LLM failed (%s) — using fallback answer", e)
        return _fallback_answer(state)
