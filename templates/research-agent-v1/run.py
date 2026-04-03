"""
run.py — Entry point for the Research Agent.

Usage:
  python run.py --query "What are the latest developments in agentic AI security?"
  python run.py          # interactive mode — prompts for query

Observability:
  - Per-node timing printed to terminal in real time via LangGraph stream()
  - Full structured log written to output/run.log after each run
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# Allow `python run.py` to work directly from the template directory.
_here = Path(__file__).resolve().parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

from dotenv import load_dotenv
from rich.console import Console
from rich.logging import RichHandler
from rich.markdown import Markdown
from rich.panel import Panel

load_dotenv()

console = Console()

# Node display names — shown in terminal during streaming
_NODE_LABELS = {
    "planner":     "Planning search queries",
    "searcher":    "Searching the web",
    "synthesiser": "Synthesising answer",
    "verifier":    "Verifying answer",
    "formatter":   "Formatting report",
}


def _setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("research_agent")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    # File handler — full debug log
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(fh)

    # Console handler via Rich — INFO and above only
    ch = RichHandler(console=console, show_path=False, markup=True)
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)

    return logger


def run_research(query: str, logger: logging.Logger) -> str:
    """
    Run the research pipeline, streaming node-by-node.
    Returns the final markdown report.
    Can be imported and called from any web framework.
    """
    from graph import research_graph
    from models import ResearchState

    initial_state = ResearchState(query=query)
    final_state: dict = {}
    node_start: float = 0.0

    logger.info("Pipeline started — query: %r", query)

    for event in research_graph.stream(initial_state, stream_mode="updates"):
        for node_name, node_output in event.items():
            elapsed = time.perf_counter() - node_start if node_start else 0.0
            if node_start:
                logger.info("  ✓ %-14s %.1fs", prev_node, elapsed)  # noqa: F821

            label = _NODE_LABELS.get(node_name, node_name)
            console.print(f"[cyan]→[/cyan] {label}…")
            logger.debug("Node started: %s | state keys: %s", node_name, list(node_output.keys()))

            node_start = time.perf_counter()
            prev_node = node_name  # noqa: F841
            final_state.update(node_output)

    # Log final timing for last node
    if node_start:
        logger.info("  ✓ %-14s %.1fs", prev_node, time.perf_counter() - node_start)  # noqa: F821

    report = final_state.get("final_report", "")
    confidence = final_state.get("confidence_score", 0.0)
    retries = final_state.get("retries", 0)
    logger.info(
        "Pipeline complete — confidence: %.0f%% | retries: %d | report: %d chars",
        confidence * 100, retries, len(report),
    )
    return report


def main():
    parser = argparse.ArgumentParser(description="Research Agent — multi-step LangGraph researcher")
    parser.add_argument("--query", "-q", type=str, help="Research question")
    parser.add_argument("--output", "-o", type=str, default="output/report.md",
                        help="Output file path (default: output/report.md)")
    args = parser.parse_args()

    output_path = Path(args.output)
    log_path = output_path.parent / "run.log"
    logger = _setup_logging(log_path)

    # Get query
    if args.query:
        query = args.query
    else:
        console.print(Panel("🔍 [bold cyan]Research Agent[/bold cyan] — powered by LangGraph", expand=False))
        query = console.input("\n[bold]Research question:[/bold] ").strip()
        if not query:
            console.print("[red]No query provided. Exiting.[/red]")
            sys.exit(1)

    provider = os.getenv("LLM_PROVIDER", "openai").upper()
    search_provider = os.getenv("SEARCH_PROVIDER", "tavily").upper()
    console.print(f"\n[dim]LLM: {provider} · Search: {search_provider} · Log: {log_path}[/dim]\n")

    start = time.perf_counter()
    try:
        report = run_research(query, logger)
    except Exception as e:
        logger.exception("Pipeline failed: %s", e)
        console.print(f"\n[red]Error:[/red] {e}")
        sys.exit(1)

    total = time.perf_counter() - start
    console.print(f"\n[dim]Total time: {total:.1f}s[/dim]\n")
    logger.info("Total wall time: %.1fs", total)

    console.print(Markdown(report))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    console.print(f"\n[dim]Report → {output_path}[/dim]")
    console.print(f"[dim]Log    → {log_path}[/dim]")


if __name__ == "__main__":
    main()
