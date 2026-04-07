#!/usr/bin/env python
"""
LLM Wiki — AgentKit template

Usage:
    python run.py ingest <source_path_or_url>
    python run.py query "<question>" [--save]
    python run.py lint
"""
import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from agent.config import validate_env, WIKI_DIR
from agent.exceptions import MaxIterationsError
from agent.graph import build_graph
from agent.state import WikiState


def _is_url(source: str) -> bool:
    return source.startswith("http://") or source.startswith("https://")


def _load_index() -> str:
    index_path = Path(WIKI_DIR) / "index.md"
    if index_path.exists():
        return index_path.read_text(encoding="utf-8")
    return ""


def _build_state(
    operation: str,
    input_: str,
    fetched_content: str = "",
    save_output: bool = False,
) -> WikiState:
    return {
        "operation": operation,
        "input": input_,
        "messages": [],
        "wiki_index": _load_index(),
        "pages_read": [],
        "output": "",
        "fetched_content": fetched_content,
        "save_output": save_output,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM Wiki agent")
    subparsers = parser.add_subparsers(dest="operation", required=True)

    ingest = subparsers.add_parser("ingest", help="Ingest a source file or URL")
    ingest.add_argument("source", help="Path to a source file, or a URL")

    query = subparsers.add_parser("query", help="Ask a question against the wiki")
    query.add_argument("question", help="Question to answer")
    query.add_argument(
        "--save",
        action="store_true",
        help="File the answer back into the wiki (LLM decides new page vs update)",
    )

    subparsers.add_parser("lint", help="Health-check the wiki")

    args = parser.parse_args()

    try:
        validate_env()
    except EnvironmentError as e:
        print(f"[error] {e}", file=sys.stderr)
        sys.exit(1)

    if args.operation == "ingest":
        source = args.source
        fetched_content = ""
        if _is_url(source):
            from agent.tools import fetch_url
            fetched_content = fetch_url(source)
            if fetched_content.startswith("[fetch_url error]"):
                print(fetched_content, file=sys.stderr)
                sys.exit(1)
        else:
            if not Path(source).exists():
                print(f"[error] Source file not found: {source}", file=sys.stderr)
                sys.exit(1)
        state = _build_state("ingest", source, fetched_content=fetched_content)

    elif args.operation == "query":
        state = _build_state("query", args.question, save_output=args.save)

    else:
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
