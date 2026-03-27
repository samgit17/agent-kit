"""
agents/synthesiser.py — Reasons over search results and produces a draft answer with citations.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from llm_client import get_llm
from models import ResearchState, Citation, SearchResult

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
- If the search results don't contain enough information, say so explicitly in the answer
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


def synthesiser_node(state: ResearchState) -> dict:
    chain = _prompt | get_llm() | JsonOutputParser()
    result = chain.invoke({
        "query": state.query,
        "results": _format_results(state.search_results),
    })
    citations = [Citation(**c) for c in result.get("citations", [])]
    return {
        "draft_answer": result.get("answer", ""),
        "citations": citations,
        "retries": state.retries + 1,   # increment here — gates the retry in should_rewrite
    }
