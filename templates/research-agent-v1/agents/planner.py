"""
agents/planner.py — Breaks the user's question into focused search queries.
"""

import logging
import os

from langchain_core.prompts import ChatPromptTemplate
from json_parser import ThinkingJsonOutputParser
from llm_client import get_llm
from models import ResearchState

logger = logging.getLogger("research_agent")

_SYSTEM = """You are a research planning agent. Given a research question, generate a list of
focused web search queries that together will provide comprehensive coverage of the topic.

Return ONLY a JSON object in this exact format — no preamble, no explanation:
{{"queries": ["query 1", "query 2", "query 3"]}}

Rules:
- Generate between 3 and {max_queries} queries
- Each query should target a distinct aspect of the question
- Queries should be specific enough to return useful results
- Do not repeat the same query with minor wording changes"""

_HUMAN = "Research question: {query}"

_prompt = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM),
    ("human", _HUMAN),
])


def planner_node(state: ResearchState) -> dict:
    max_q = int(os.getenv("MAX_SEARCH_QUERIES", "5"))
    try:
        chain = _prompt | get_llm() | ThinkingJsonOutputParser()
        result = chain.invoke({"query": state.query, "max_queries": max_q})
        queries = result.get("queries", [])
        if not queries:
            raise ValueError("Model returned empty queries list")
    except Exception as e:
        # Fallback: split the query itself into 3 basic search terms
        logger.warning("Planner LLM failed (%s) — using fallback queries", e)
        queries = [
            state.query,
            f"{state.query} best practices",
            f"{state.query} examples tutorial",
        ]
    return {"search_queries": queries}
