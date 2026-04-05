from pathlib import Path

from agent.config import WIKI_DIR

SUPPORTED_EXTENSIONS = {".md", ".txt", ".rst", ".html", ".pdf", ".docx"}
TEXT_EXTENSIONS = {".md", ".txt", ".rst", ".html"}
DOCLING_EXTENSIONS = {".pdf", ".docx"}


def _converter_factory():
    """Returns a DocumentConverter instance. Extracted for testability."""
    try:
        from docling.document_converter import DocumentConverter
    except ImportError:
        raise ImportError(
            "Docling is required for PDF and DOCX files. "
            "Install it with: pip install docling"
        )
    return DocumentConverter()


def read_file(path: str) -> str:
    """Read a source file, dispatching to docling for PDF/DOCX."""
    p = Path(path)
    ext = p.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: '{ext}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    if ext in DOCLING_EXTENSIONS:
        return _read_via_docling(path)
    return p.read_text(encoding="utf-8")


def _read_via_docling(path: str) -> str:
    converter = _converter_factory()
    result = converter.convert(path)
    return result.document.export_to_markdown()


def write_file(path: str, content: str) -> None:
    """Write content to path, confined to WIKI_DIR to prevent path traversal."""
    p = Path(path).resolve()
    allowed = Path(WIKI_DIR).resolve()
    if not str(p).startswith(str(allowed)):
        raise ValueError(
            f"write_file is restricted to '{WIKI_DIR}/'. "
            f"Attempted path: {path}"
        )
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")


def list_wiki(wiki_dir: str) -> list[str]:
    """Return relative paths of all .md files in wiki_dir."""
    return sorted(str(p) for p in Path(wiki_dir).rglob("*.md"))
