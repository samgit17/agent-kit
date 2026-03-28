"""
json_parser.py — JSON output parser with think-tag stripping and truncation recovery.
"""

import json
import logging
import re

from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import JsonOutputParser

logger = logging.getLogger("research_agent.parser")

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_JSON_RE   = re.compile(r"\{.*\}", re.DOTALL)


def _get_raw(result) -> str:
    for r in result:
        if hasattr(r, "message") and hasattr(r.message, "content"):
            c = r.message.content
            if isinstance(c, str):
                return c
        if hasattr(r, "text") and isinstance(r.text, str):
            return r.text
    return ""


def _recover_truncated(text: str) -> dict | None:
    """
    Try to salvage a truncated JSON response.
    If the answer field is present but the JSON is unclosed, extract what we have.
    """
    answer_match = re.search(r'"answer"\s*:\s*"(.*?)(?:"|$)', text, re.DOTALL)
    if answer_match:
        answer = answer_match.group(1).rstrip('\\').strip()
        # Extract any complete citation objects
        citations = re.findall(
            r'\{"url"\s*:\s*"([^"]+)".*?"title"\s*:\s*"([^"]+)"',
            text,
            re.DOTALL,
        )
        return {
            "answer": answer,
            "citations": [{"url": u, "title": t, "section": ""} for u, t in citations],
        }
    return None


class ThinkingJsonOutputParser(JsonOutputParser):

    def parse_result(self, result, *, partial: bool = False) -> dict:
        raw = _get_raw(result)
        logger.debug("Raw LLM output (%d chars): %r", len(raw), raw[:400])

        cleaned = _THINK_RE.sub("", raw).strip()

        if not cleaned:
            logger.error("Empty model output. Raw was: %r", raw[:500])
            raise OutputParserException(
                "Model returned empty output. Try increasing OLLAMA_MAX_TOKENS."
            )

        # Standard parse
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Extract first complete {...} block
        m = _JSON_RE.search(cleaned)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass

        # Truncation recovery — extract partial answer field
        recovered = _recover_truncated(cleaned)
        if recovered and recovered.get("answer"):
            logger.warning("JSON was truncated — recovered partial answer (%d chars). "
                           "Increase OLLAMA_MAX_TOKENS to avoid this.", len(recovered["answer"]))
            return recovered

        logger.error("JSON parse failed. Cleaned output: %r", cleaned[:400])
        raise OutputParserException(
            f"Could not parse JSON. Output started with: {cleaned[:100]}"
        )
