from typing import TypedDict


class WikiState(TypedDict):
    operation: str
    input: str          # file path, question, or URL
    messages: list
    wiki_index: str
    pages_read: list
    output: str
    fetched_content: str  # populated when input is a URL; empty otherwise
    save_output: bool     # --save flag for query operation
