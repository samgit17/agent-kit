from langchain_core.tools import tool

from agent.tools import read_file as _read_file
from agent.tools import write_file as _write_file
from agent.tools import list_wiki as _list_wiki
from agent.config import WIKI_DIR


@tool
def read_file(path: str) -> str:
    """Read the full text content of a file at the given path.
    Use this to read a source document before ingesting it, or to read
    a specific wiki page after identifying it from the index.
    Always use the full path including the wiki directory prefix
    (e.g. wiki/entities/python.md, not entities/python.md).
    Input: full file path (str).
    Returns: file content as a string.
    Raises FileNotFoundError if path does not exist.
    """
    return _read_file(path)


@tool
def write_file(path: str, content: str) -> str:
    """Write content to a markdown file inside the wiki directory.
    Use this to create a new wiki page or overwrite an existing one.
    Always use the full path including the wiki directory prefix
    (e.g. wiki/entities/python.md, not entities/python.md).
    Never write index.md or log.md until all other pages for this operation are written.
    Input: path must be inside the wiki directory (e.g. wiki/entities/python.md).
           content must be valid markdown.
    Raises ValueError if path is outside the wiki directory.
    Returns: confirmation message with the path written.
    """
    _write_file(path, content)
    return f"Written: {path}"


@tool
def list_wiki() -> list[str]:
    """List all markdown files currently in the wiki directory.
    Use this at the start of query and lint operations to discover
    what pages exist before deciding which ones to read.
    Returns full paths including the wiki directory prefix
    (e.g. wiki/entities/python.md).
    Never use this during ingest — read index.md directly instead.
    Returns: sorted list of full file paths as strings.
    """
    return _list_wiki(WIKI_DIR)


WIKI_TOOLS = [read_file, write_file, list_wiki]
