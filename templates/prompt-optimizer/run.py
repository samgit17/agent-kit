"""
run.py — Entry point for Prompt Optimizer.

Edit program.md to change goal, eval criteria, and constraints.
python run.py
"""

import os
from dotenv import load_dotenv
load_dotenv()

    
if os.getenv("TRACING_ENABLED"):
    try:
        from tracing import init_tracing
        init_tracing("prompt-optomizer")
    except ImportError:
        pass  # tracing deps not installed

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


from rich.markdown import Markdown
from rich.panel import Panel

from backends.log import console, log
from program_parser import parse_program, ProgramConfig




def _load_backend(cfg: ProgramConfig):
    if cfg.backend == "prompt_optimizer":
        from backends.prompt_optimizer.graph import build_graph, build_initial_state
        return build_graph, build_initial_state
    else:
        console.print(f"[red]Unknown backend: {cfg.backend!r}[/red]")
        sys.exit(1)


def _check_git():
    result = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"],
        cwd=str(Path(__file__).parent),
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        console.print(
            "[red]prompt-optimizer requires a git repo.\n"
            "Run: git init && git add skill.md && git commit -m 'baseline'[/red]"
        )
        sys.exit(1)


def main():
    start = datetime.now()

    program_path = Path("program.md")
    if not program_path.exists():
        console.print("[red]program.md not found.[/red]")
        sys.exit(1)

    cfg = parse_program(program_path)
    llm_provider = os.getenv("LLM_PROVIDER", "ollama").upper()

    console.print(Panel(
        f"[bold]Backend:[/bold]  {cfg.backend}\n"
        f"[bold]Goal:[/bold]     {cfg.goal}\n"
        f"[bold]Target:[/bold]   {cfg.po.target_file}\n"
        f"[dim]LLM: {llm_provider}  --  Started: {start.strftime('%Y-%m-%d %H:%M:%S')}[/dim]",
        title="Prompt Optimizer",
        border_style="blue",
    ))

    _check_git()

    build_graph, build_initial_state = _load_backend(cfg)
    graph = build_graph(cfg)
    initial_state = build_initial_state(cfg)

    if initial_state.get("best_score", 0.0) > 0.0:
        log("[run]", f"Resuming from git history -- best score so far: {initial_state['best_score']:.2%}")

    result = graph.invoke(initial_state)

    report = result.get("report", "")
    if report:
        console.print(Markdown(report))

    elapsed = datetime.now() - start
    log("[run]", f"Finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} -- elapsed {str(elapsed).split('.')[0]}")


if __name__ == "__main__":
    main()
