"""
tools/search.py — Web search wrapper.
Supports Tavily (structured, recommended) and DuckDuckGo (free, no key).
Provider selected via SEARCH_PROVIDER env var.
"""

import os
from models import SearchResult


def search(query: str, max_results: int | None = None) -> list[SearchResult]:
    """Execute a single search query and return structured results."""
    provider = os.getenv("SEARCH_PROVIDER", "tavily").lower()
    n = max_results or int(os.getenv("MAX_SEARCH_RESULTS", "3"))

    if provider == "tavily":
        return _tavily_search(query, n)
    elif provider == "duckduckgo":
        return _ddg_search(query, n)
    else:
        raise ValueError(
            f"Unknown SEARCH_PROVIDER='{provider}'. Set to 'tavily' or 'duckduckgo' in .env"
        )


def _tavily_search(query: str, max_results: int) -> list[SearchResult]:
    from tavily import TavilyClient
    client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    response = client.search(query=query, max_results=max_results, include_answer=False)
    return [
        SearchResult(
            query=query,
            url=r.get("url", ""),
            title=r.get("title", ""),
            content=r.get("content", ""),
            score=r.get("score", 0.0),
        )
        for r in response.get("results", [])
    ]


def _ddg_search(query: str, max_results: int) -> list[SearchResult]:
    from duckduckgo_search import DDGS
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            results.append(
                SearchResult(
                    query=query,
                    url=r.get("href", ""),
                    title=r.get("title", ""),
                    content=r.get("body", ""),
                    score=0.0,  # DuckDuckGo doesn't return relevance scores
                )
            )
    return results
