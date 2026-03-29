"""
backends/log.py — Shared Rich console and log helper for all backends.

Single import: `from backends.log import log`
"""

from rich.console import Console

console = Console()


def log(tag: str, msg: str, style: str = "dim") -> None:
    safe_tag = tag.replace("[", "\\[")
    console.print(f"[{style}]{safe_tag}[/{style}] {msg}")
