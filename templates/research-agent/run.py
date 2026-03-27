"""
run.py — Entry point for the Research Agent.

Usage:
  python run.py --query "What are the latest developments in agentic AI security?"
  python run.py          # interactive mode — prompts for query
"""

import argparse
import os
import sys
from pathlib import Path

# Allow `python run.py` to work directly from the template directory.
# Adds research-agent/ to sys.path so sibling modules (graph, models) are importable.
_here = Path(__file__).resolve().parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

load_dotenv()

console = Console()


def run_research(query: str) -> str:
    """
    Run the research pipeline for a given query.
    Returns the final markdown report as a string.
    Can be imported and called from any web framework.
    """
    from graph import research_graph
    from models import ResearchState

    initial_state = ResearchState(query=query)
    final_state = research_graph.invoke(initial_state)
    return final_state["final_report"]


def main():
    parser = argparse.ArgumentParser(description="Research Agent — multi-step LangGraph researcher")
    parser.add_argument("--query", "-q", type=str, help="Research question")
    parser.add_argument("--output", "-o", type=str, default="output/report.md",
                        help="Output file path (default: output/report.md)")
    args = parser.parse_args()

    # Get query
    if args.query:
        query = args.query
    else:
        console.print(Panel("🔍 [bold cyan]Research Agent[/bold cyan] — powered by LangGraph", expand=False))
        query = console.input("\n[bold]Research question:[/bold] ").strip()
        if not query:
            console.print("[red]No query provided. Exiting.[/red]")
            sys.exit(1)

    # Show config
    provider = os.getenv("LLM_PROVIDER", "openai").upper()
    search = os.getenv("SEARCH_PROVIDER", "tavily").upper()
    console.print(f"\n[dim]LLM: {provider} · Search: {search}[/dim]\n")

    # Run with progress indicator
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Running research pipeline...", total=None)

        try:
            report = run_research(query)
            progress.update(task, description="✓ Done")
        except Exception as e:
            progress.stop()
            console.print(f"\n[red]Error:[/red] {e}")
            sys.exit(1)

    # Print report to terminal
    console.print("\n")
    console.print(Markdown(report))

    # Save to file
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report)
    console.print(f"\n[dim]Report saved to {output_path}[/dim]")


if __name__ == "__main__":
    main()
