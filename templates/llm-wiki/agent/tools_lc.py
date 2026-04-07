from langchain_core.tools import tool

from agent.config import WIKI_DIR
from agent.tools import read_file as _read_file
from agent.tools import write_file as _write_file
from agent.tools import list_wiki as _list_wiki


@tool
def read_file(path: str) -> str:
    """Read the full contents of a file at the given path.
    Supports .md, .txt, .rst, .html, .pdf, .docx.
    For PDF and DOCX, content is extracted via Docling.
    """
    return _read_file(path)


@tool
def write_file(path: str, content: str) -> str:
    """Write content to a file inside the wiki directory.
    Creates parent directories as needed.
    Never call this for index.md or log.md until all other pages are written.
    """
    return _write_file(path, content)


@tool
def list_wiki() -> str:
    """List all markdown files currently in the wiki directory.
    Use this at the start of query and lint operations to discover
    what pages exist before deciding which ones to read.
    Returns full paths including the wiki directory prefix
    (e.g. wiki/entities/python.md).
    Never use this during ingest — read index.md directly instead.
    """
    result = _list_wiki(WIKI_DIR)
    return "\n".join(result) if isinstance(result, list) else result


WIKI_TOOLS = [read_file, write_file, list_wiki]
