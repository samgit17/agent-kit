"""
agents/formatter.py — Assembles the final markdown research report.
"""

from datetime import datetime
from models import ResearchState


def formatter_node(state: ResearchState) -> dict:
    lines = [
        "# Research Report",
        f"**Query:** {state.query}",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Confidence:** {state.confidence_score:.0%}",
        "",
    ]

    if state.uncertainty_flagged:
        lines += [
            "> ⚠️ **Low confidence** — some claims could not be fully verified against sources.",
            "",
        ]

    lines += [
        "## Summary",
        "",
        state.draft_answer,
        "",
        "## Sources",
        "",
    ]

    seen = set()
    counter = 1
    for c in state.citations:
        if c.url not in seen:
            seen.add(c.url)
            lines.append(f"{counter}. [{c.title}]({c.url})")
            if c.section:
                lines.append(f"   > {c.section}")
            counter += 1

    lines += [
        "",
        "---",
        f"*Queries used: {len(state.search_queries)} · "
        f"Sources gathered: {len(state.search_results)} · "
        f"Citations: {len(seen)}*",
    ]

    return {"final_report": "\n".join(lines)}
