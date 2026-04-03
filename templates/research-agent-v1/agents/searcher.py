"""
agents/searcher.py — Executes all planned search queries in parallel.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from models import ResearchState, SearchResult
from tools.search import search

logger = logging.getLogger(__name__)


def searcher_node(state: ResearchState) -> dict:
    all_results: list[SearchResult] = []

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(search, q): q for q in state.search_queries}
        for future in as_completed(futures):
            try:
                results = future.result()
                all_results.extend(results)
            except Exception as e:
                logger.warning("search failed for %r: %s", futures[future], e)

    return {"search_results": all_results}
