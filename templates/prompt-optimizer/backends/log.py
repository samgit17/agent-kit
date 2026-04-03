from rich.console import Console

console = Console()


def log(tag: str, msg: str, style: str = "dim") -> None:
    safe_tag = tag.replace("[", "\\[")
    console.print(f"[{style}]{safe_tag}[/{style}] {msg}")
