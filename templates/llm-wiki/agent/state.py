from typing import Literal
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage


class WikiState(TypedDict):
    operation: Literal["ingest", "query", "lint"]
    input: str                    # source file path or query string
    messages: list[BaseMessage]
    wiki_index: str               # contents of wiki/index.md, injected into prompts
    output: str                   # final human-readable response
