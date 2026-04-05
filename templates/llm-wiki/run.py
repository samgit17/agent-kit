#!/usr/bin/env python
"""
LLM Wiki — AgentKit template

Usage:
    python run.py ingest <source_path>
    python run.py query "<question>"
    python run.py lint
"""
import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from agent.config import validate_env, WIKI_DIR, SOURCES_DIR
from agent.exceptions import MaxIterationsError
from agent.graph import build_graph
from agent.state import WikiState


def _ensure_dirs() -> None:
    """Create wiki/ and sources/ on first run if they don't exist."""
    Path(WIKI_DIR).mkdir(parents=True, exist_ok=True)
    Path(SOURCES_DIR).mkdir(parents=True, exist_ok=True)


def _load_index() -> str:
    index_path = Path(WIKI_DIR) / "index.md"
    if index_path.exists():
        return index_path.read_text(encoding="utf-8")
    return ""


def _build_state(operation: str, input_: str) -> WikiState:
    return {
        "operation": operation,
        "input": input_,
        "messages": [],
        "wiki_index": _load_index(),
        "output": "",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM Wiki agent")
    subparsers = parser.add_subparsers(dest="operation", required=True)

    ingest = subparsers.add_parser("ingest", help="Ingest a source document into the wiki")
    ingest.add_argument("source", help="Path to the source file")

    query = subparsers.add_parser("query", help="Ask a question against the wiki")
    query.add_argument("question", help="Question to answer")

    subparsers.add_parser("lint", help="Health-check the wiki")

    args = parser.parse_args()

    _ensure_dirs()

    try:
        validate_env()
    except EnvironmentError as e:
        print(f"[error] {e}", file=sys.stderr)
        sys.exit(1)

    if args.operation == "ingest":
        source = Path(args.source)
        if not source.exists():
            print(f"[error] Source file not found: {args.source}", file=sys.stderr)
            sys.exit(1)
        from agent.tools import SUPPORTED_EXTENSIONS
        if source.suffix.lower() not in SUPPORTED_EXTENSIONS:
            print(
                f"[error] Unsupported file type: '{source.suffix}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
                file=sys.stderr,
            )
            sys.exit(1)
        state = _build_state("ingest", str(source))

    elif args.operation == "query":
        state = _build_state("query", args.question)

    else:  # lint
        state = _build_state("lint", "")

    try:
        graph = build_graph()
        result = graph.invoke(state)
        print(result["output"])
    except MaxIterationsError as e:
        print(f"[error] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
