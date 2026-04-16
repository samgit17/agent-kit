"""
nodes.py — Node functions for the research graph.

Planner   → generates search queries from query
Searcher  → executes queries via Tavily or DuckDuckGo
Synthesiser → synthesises results into a draft
Verifier  → scores confidence; flags retry if too low
Formatter → produces final markdown report
"""

from __future__ import annotations
import os
import json
from models import ResearchState
from langchain_core.messages import HumanMessage
from llm import get_llm, strip_json_fence
from log import log


def _search(query: str) -> list[dict]:
    provider = os.getenv("SEARCH_PROVIDER", "duckduckgo").lower()
    if provider == "tavily":
        from langchain_tavily import TavilySearch
        tool = TavilySearch(max_results=5)
        results = tool.invoke({"query": query})
        return results if isinstance(results, list) else []
    else:
        from langchain_community.tools import DuckDuckGoSearchResults
        tool = DuckDuckGoSearchResults(output_format="list", max_results=5)
        return tool.invoke(query)


def planner_node(state: ResearchState) -> dict:
    prompt = f"""You are a research planner. Generate 3 precise search queries to answer:

QUESTION: {state.query}

Return ONLY a JSON array of 3 query strings. No explanation."""

    response = get_llm().invoke([HumanMessage(content=prompt)])
    raw = strip_json_fence(response.content)
    queries = json.loads(raw)
    log("[planner]", f"Generated {len(queries)} queries")
    return {"queries": queries}


def searcher_node(state: ResearchState) -> dict:
    all_results = []
    for query in state.queries:
        try:
            results = _search(query)
            all_results.extend(results[:3])
        except Exception as e:
            log("[searcher]", f"query failed: {query!r} — {type(e).__name__}: {e}", style="red")
    log("[searcher]", f"Retrieved {len(all_results)} results across {len(state.queries)} queries")
    if not all_results:
        log("[searcher]", "No results — check TAVILY_API_KEY or switch SEARCH_PROVIDER=duckduckgo", style="red")
    return {"search_results": all_results}


def synthesiser_node(state: ResearchState) -> dict:
    results_text = "\n\n".join(
        f"[{i+1}] {r.get('title','')}\n{r.get('content','')[:800]}"
        for i, r in enumerate(state.search_results)
    )
    gaps_text = ""
    if state.verifier_gaps:
        gaps_text = "\n\nPREVIOUS REVIEW FLAGGED THESE GAPS — address them:\n" + \
                    "\n".join(f"- {g}" for g in state.verifier_gaps)

    prompt = f"""Synthesise the search results into a coherent answer.

QUESTION: {state.query}
{gaps_text}

RESULTS:
{results_text}

Write a clear, factual answer. Cite sources by number."""

    response = get_llm().invoke([HumanMessage(content=prompt)])
    log("[synthesiser]", f"Draft complete (iteration {state.iterations + 1}/{state.max_iterations})")
    return {"synthesis": response.content, "iterations": state.iterations + 1}


def verifier_node(state: ResearchState) -> dict:
    prompt = f"""Rate how well this answer addresses the question.

QUESTION: {state.query}

ANSWER:
{state.synthesis}

Return ONLY a JSON object: {{"confidence": 0.0-1.0, "gaps": ["..."]}}"""

    response = get_llm().invoke([HumanMessage(content=prompt)])
    raw = strip_json_fence(response.content)
    result = json.loads(raw)
    confidence = result.get("confidence", 0.0)
    gaps = result.get("gaps", [])
    log("[verifier]", f"Confidence: {confidence:.0%}" + ("  ✅" if confidence >= state.confidence_threshold else "  ↩ retrying"))
    return {"confidence": confidence, "verifier_gaps": gaps}


def should_retry(state: ResearchState) -> str:
    if not state.search_results:
        return "format"
    if state.confidence < state.confidence_threshold and state.iterations < state.max_iterations:
        return "retry"
    return "format"


def formatter_node(state: ResearchState) -> dict:
    report = f"""{state.synthesis}

---
*Confidence: {state.confidence:.0%} | Iterations: {state.iterations}*
"""
    log("[formatter]", "Answer ready")
    return {"report": report}
