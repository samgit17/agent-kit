"""
run.py — Entry point for Research Agent v2.

Edit program.md to change what the agent does.
Do not pass CLI args — program.md is the only control surface.
"""

import os
from dotenv import load_dotenv
load_dotenv()

    
if os.getenv("TRACING_ENABLED"):
    try:
        from tracing import init_tracing
        init_tracing("research-agent")
    except ImportError:
        pass  # tracing deps not installed
    
import subprocess
import sys
from pathlib import Path
from rich.markdown import Markdown
from rich.panel import Panel
from backends.log import console, log
from program_parser import parse_program, ProgramConfig



REPORT_PATH = Path("output/report.md")


def _load_backend(cfg: ProgramConfig):
    """Return (build_graph, build_initial_state) for the configured backend."""
    if cfg.backend == "web":
        from backends.web.graph import build_graph, build_initial_state
        return build_graph, build_initial_state
    elif cfg.backend == "ml_experiment":
        from backends.ml_experiment.graph import build_graph, build_initial_state
        return build_graph, build_initial_state
    else:
        console.print(f"[red]Unknown backend: {cfg.backend!r}[/red]")
        sys.exit(1)


def _check_git():
    """Abort early if train.py is not inside a git repo — required for revert."""
    train_py = Path(__file__).parent / "backends" / "ml_experiment" / "train.py"
    result = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"],
        cwd=str(train_py.parent),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        console.print(
            "[red]ML experiment backend requires train.py to be inside a git repo.\n"
            "Run: git init backends/ml_experiment && git add train.py && git commit -m 'baseline'[/red]"
        )
        sys.exit(1)


def main():
    from datetime import datetime
    start = datetime.now()
    log("[run]", f"Started at {start.strftime('%Y-%m-%d %H:%M:%S')}")
    program_path = Path("program.md")
    if not program_path.exists():
        console.print("[red]program.md not found.[/red]")
        sys.exit(1)

    cfg = parse_program(program_path)

    llm_provider = os.getenv("LLM_PROVIDER", "ollama").upper()
    search_provider = os.getenv("SEARCH_PROVIDER", "duckduckgo").upper()

    console.print(Panel(
        f"[bold]Backend:[/bold]  {cfg.backend}\n"
        f"[bold]Goal:[/bold]     {cfg.goal}\n"
        f"[dim]LLM: {llm_provider}"
        + (f"  ·  Search: {search_provider}" if cfg.backend == "web" else "") + "[/dim]",
        title="Research Agent v2",
        border_style="blue",
    ))

    if cfg.backend == "ml_experiment":
        _check_git()

    build_graph, build_initial_state = _load_backend(cfg)
    graph = build_graph(cfg)
    initial_state = build_initial_state(cfg)

    if cfg.backend == "ml_experiment":
        best = initial_state.get("best_val_bpb", float("inf"))
        if best != float("inf"):
            log("[run]", f"Resuming from git history — best val_bpb so far: {best:.4f}")

    result = graph.invoke(initial_state)

    report = result.get("report", "")
    if not report:
        return

    console.print(Markdown(report))

    if cfg.backend == "web":
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(report, encoding="utf-8")
        log("[run]", f"Report saved to {REPORT_PATH}")

        diagram_url = result.get("diagram_url")
        if diagram_url:
            log("[run]", f"Diagram: {diagram_url}")

    elapsed = datetime.now() - start
    log("[run]", f"Finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} — elapsed {str(elapsed).split('.')[0]}")


if __name__ == "__main__":
    main()
