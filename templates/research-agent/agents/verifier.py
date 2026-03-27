"""
agents/verifier.py — Scores faithfulness and confidence of the draft answer.
Routes to formatter if confident enough, or flags uncertainty.
"""

import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from llm_client import get_llm
from models import ResearchState

_SYSTEM = """You are a research verification agent. You are given a question, a set of
source documents, and a draft answer. Your job is to assess how well the answer is
grounded in the sources.

Return ONLY a JSON object:
{{
  "confidence_score": 0.85,
  "feedback": "The answer accurately reflects the sources on points X and Y. Point Z could not be verified."
}}

Scoring guide:
- 0.9-1.0: Answer is fully grounded, all claims traceable to sources
- 0.7-0.9: Mostly grounded, minor gaps
- 0.5-0.7: Partially grounded, some unverifiable claims
- 0.0-0.5: Significant hallucination risk"""

_HUMAN = """Question: {query}

Draft Answer: {answer}

Sources used:
{sources}"""

_prompt = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM),
    ("human", _HUMAN),
])


def verifier_node(state: ResearchState) -> dict:
    threshold = float(os.getenv("VERIFIER_THRESHOLD", "0.70"))
    chain = _prompt | get_llm() | JsonOutputParser()

    sources = "\n".join(f"- [{c.title}]({c.url})" for c in state.citations)

    result = chain.invoke({
        "query": state.query,
        "answer": state.draft_answer,
        "sources": sources,
    })

    score = float(result.get("confidence_score", 0.0))
    return {
        "confidence_score": score,
        "uncertainty_flagged": score < threshold,
        "verifier_feedback": result.get("feedback", ""),
    }


def should_rewrite(state: ResearchState) -> str:
    """Conditional edge — retry once if confidence is too low.
    retries is incremented by synthesiser_node, so after one retry it equals 2.
    """
    if state.uncertainty_flagged and state.retries < 2:
        return "retry"
    return "format"
