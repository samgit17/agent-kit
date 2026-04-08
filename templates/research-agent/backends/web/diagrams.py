"""
diagrams.py — Optional diagram generation skill for research-agent.

Set DIAGRAMS_ENABLED=true to activate. Uses the same LLM as the rest of
the graph (get_llm()) — no additional model config required.

Output: a clickable draw.io URL written to state.diagram_url.
The URL opens an editable concept map in the browser — no install needed
for the end user.

Encoding: draw.io #R fragment = urlencoded(base64(raw_deflate(xml)))
Matches pako.deflateRaw used by the draw.io client.
"""

from __future__ import annotations
import base64
import os
import urllib.parse
import zlib

from langchain_core.messages import HumanMessage

from backends.llm import get_llm
from backends.log import log
from .models import ResearchState

_DIAGRAM_PROMPT = """\
You are a diagram generator. Given this research report, produce draw.io mxGraph XML \
representing the key concepts, entities, and relationships as a concept map.

GOAL: {goal}

REPORT SUMMARY:
{synthesis}

Rules:
- Return ONLY valid mxGraph XML starting with <mxGraphModel>
- Rectangular nodes for concepts, labeled edges for relationships
- 5-10 nodes maximum — keep it readable
- No markdown, no code fences, no explanation"""


def is_enabled() -> bool:
    return os.getenv("DIAGRAMS_ENABLED", "").lower() in ("1", "true")


def _xml_to_drawio_url(xml: str) -> str:
    """Compress XML and return a draw.io #R URL (no install required for end user)."""
    raw_deflate = zlib.compress(xml.encode("utf-8"))[2:-4]  # strip zlib header + adler32
    encoded = urllib.parse.quote(base64.b64encode(raw_deflate).decode("utf-8"), safe="")
    return f"https://app.diagrams.net/#R{encoded}"


def _strip_xml_fence(text: str) -> str:
    """Best-effort removal of markdown code fences around XML."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0]
    return text.strip()


def diagram_node(state: ResearchState) -> dict:
    prompt = _DIAGRAM_PROMPT.format(
        goal=state.goal,
        synthesis=state.synthesis[:2000],
    )
    response = get_llm().invoke([HumanMessage(content=prompt)])
    xml = _strip_xml_fence(response.content)
    url = _xml_to_drawio_url(xml)
    log("[diagrams]", "Diagram URL ready — open in browser to view/edit")
    return {"diagram_url": url}
