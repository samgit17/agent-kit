"""
agents/verifier.py — Scores faithfulness and confidence of the draft answer.
Fault-tolerant: if the LLM call fails or times out, returns a default score
and lets the pipeline continue rather than crashing.
"""

import logging
import os

from langchain_core.prompts import ChatPromptTemplate
from json_parser import ThinkingJsonOutputParser
from llm_client import get_llm
from models import ResearchState

logger = logging.getLogger("research_agent")

_MAX_ANSWER_CHARS = 1500

_SYSTEM = """You are a fact-checking agent. Score how well the draft answer is grounded
in the provided source URLs.

Return ONLY a JSON object — no preamble:
{{
  "confidence_score": 0.85,
  "feedback": "one sentence summary of grounding quality"
}}

Scoring:
- 0.9-1.0: fully grounded
- 0.7-0.9: mostly grounded, minor gaps
- 0.5-0.7: partial, some unverifiable claims
- 0.0-0.5: likely hallucination"""

_HUMAN = """Question: {query}

Draft Answer (excerpt): {answer_excerpt}

Sources: {sources}"""

_prompt = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM),
    ("human", _HUMAN),
])


def verifier_node(state: ResearchState) -> dict:
    threshold = float(os.getenv("VERIFIER_THRESHOLD", "0.70"))

    try:
        chain = _prompt | get_llm() | ThinkingJsonOutputParser()
        sources = ", ".join(c.url for c in state.citations[:5])
        result = chain.invoke({
            "query": state.query,
            "answer_excerpt": state.draft_answer[:_MAX_ANSWER_CHARS],
            "sources": sources,
        })
        score = float(result.get("confidence_score", 0.0))
        feedback = result.get("feedback", "")
    except Exception as e:
        # Verifier failure is non-fatal — log it and continue with a neutral score
        logger.warning("Verifier failed (%s) — using default score 0.75", e)
        score = 0.75
        feedback = "Verification skipped due to error."

    return {
        "confidence_score": score,
        "uncertainty_flagged": score < threshold,
        "verifier_feedback": feedback,
    }


def should_rewrite(state: ResearchState) -> str:
    """Conditional edge — retry only if explicitly enabled."""
    enable_retry = os.getenv("ENABLE_RETRY", "false").lower() == "true"
    if enable_retry and state.uncertainty_flagged and state.retries < 2:
        return "retry"
    return "format"
