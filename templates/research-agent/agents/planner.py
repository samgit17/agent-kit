"""
agents/planner.py — Breaks the user's question into focused search queries.
"""

import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from llm_client import get_llm
from models import ResearchState

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
    chain = _prompt | get_llm() | JsonOutputParser()
    result = chain.invoke({"query": state.query, "max_queries": max_q})
    return {"search_queries": result.get("queries", [])}
