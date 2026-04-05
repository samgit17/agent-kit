from datetime import datetime
from typing import Callable

from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool

from agent.config import MAX_ITERATIONS, WIKI_DIR
from agent.exceptions import MaxIterationsError
from agent.state import WikiState


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

def _ingest_prompt(wiki_index: str) -> str:
    return f"""\
You are a wiki maintainer. Process a source document and integrate
its knowledge into the wiki directory: {WIKI_DIR}/

Current wiki index:
{wiki_index or "(empty — this is the first ingest)"}

Steps:
1. Read the source document using read_file.
2. Extract key entities, concepts, facts, and relationships.
3. Create new pages for significant entities/concepts not yet in the wiki.
   Always use full paths including the wiki prefix (e.g. {WIKI_DIR}/entities/python.md).
4. For existing pages that need updating, read first, then write.
   Always use full paths when calling read_file and write_file.
5. Never write index.md or log.md until all other pages are written.
6. Update {WIKI_DIR}/index.md — index links must use full paths (e.g. {WIKI_DIR}/entities/python.md).
7. Append to {WIKI_DIR}/log.md: ## [DATE] ingest | <source title>
"""


def _query_prompt(wiki_index: str) -> str:
    return f"""\
You are a wiki researcher. Answer questions using the wiki directory: {WIKI_DIR}/

Current wiki index:
{wiki_index or "(empty — no pages ingested yet)"}

Steps:
1. Identify relevant pages from the index above.
2. Read relevant pages using read_file.
3. Synthesize a clear answer with citations to wiki pages.
4. If the answer reveals a useful analysis, indicate it could be filed
   as a new wiki page — but wait for user confirmation before writing.
5. Never write index.md or log.md until all other pages are written.
"""


LINT_PROMPT = f"""\
You are a wiki health inspector. Find and report issues in: {WIKI_DIR}/

Steps:
1. Use list_wiki to discover all pages.
2. Read each page using read_file.
3. Flag: contradictions, orphan pages, stale claims,
   missing cross-references, concepts lacking their own page.
4. Write your full findings to the lint report path provided in the prompt.
5. Never write index.md or log.md until all other pages are written.
6. Append to {WIKI_DIR}/log.md: ## [DATE] lint | <issue count> issues found
"""


# ---------------------------------------------------------------------------
# ReAct loop
# ---------------------------------------------------------------------------

def _react_loop(
    model,
    messages: list,
    tool_map: dict[str, BaseTool],
    max_iterations: int = MAX_ITERATIONS,
) -> list:
    messages = list(messages)  # never mutate caller's list
    for _ in range(max_iterations):
        response = model.invoke(messages)
        messages.append(response)
        if not response.tool_calls:
            break
        for tc in response.tool_calls:
            result = tool_map[tc["name"]].invoke(tc["args"])
            messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
    else:
        raise MaxIterationsError(
            f"Exceeded {max_iterations} iterations without completing. "
            f"Increase WIKI_MAX_ITERATIONS or simplify the operation."
        )
    return messages


# ---------------------------------------------------------------------------
# Node factories
# ---------------------------------------------------------------------------

def build_ingest_node(model, tool_map: dict[str, BaseTool]):
    def ingest_node(state: WikiState) -> WikiState:
        messages = _react_loop(model, [
            SystemMessage(content=_ingest_prompt(state["wiki_index"])),
            HumanMessage(content=f"Ingest this source: {state['input']}"),
        ], tool_map)
        return {**state, "messages": messages, "output": messages[-1].content}
    return ingest_node


def build_query_node(model, tool_map: dict[str, BaseTool], prompter: Callable[[str], str] = input):
    def query_node(state: WikiState) -> WikiState:
        messages = _react_loop(model, [
            SystemMessage(content=_query_prompt(state["wiki_index"])),
            HumanMessage(content=state["input"]),
        ], tool_map)
        answer = messages[-1].content

        if "could be filed" in answer.lower():
            reply = prompter("File this analysis as a wiki page? [y/N] ").strip().lower()
            if reply == "y":
                messages = _react_loop(model, [
                    *messages,
                    HumanMessage(content="Yes, please file this as a new wiki page."),
                ], tool_map)

        return {**state, "messages": messages, "output": messages[-1].content}
    return query_node


def build_lint_node(model, tool_map: dict[str, BaseTool]):
    def lint_node(state: WikiState) -> WikiState:
        report_path = (
            f"{WIKI_DIR}/lint-{datetime.now().strftime('%Y-%m-%dT%H-%M')}.md"
        )
        messages = _react_loop(model, [
            SystemMessage(content=LINT_PROMPT),
            HumanMessage(content=f"Lint the wiki. Write your report to: {report_path}"),
        ], tool_map)
        output = messages[-1].content
        return {**state, "messages": messages, "output": output}
    return lint_node
