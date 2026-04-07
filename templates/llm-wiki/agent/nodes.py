"""
ReAct nodes for llm-wiki.
- build_ingest_node: handles file paths and pre-fetched URL content (fetched_content in state)
- build_query_node:  save_output flag triggers structured LLM decision → new page or update existing
- build_lint_node:   health-check, writes versioned lint report
"""
import json
from datetime import datetime
from typing import Callable

from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from rich.console import Console

from agent.config import MAX_ITERATIONS, WIKI_DIR
from agent.exceptions import MaxIterationsError
from agent.state import WikiState

console = Console()


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

def _ingest_prompt(wiki_index: str) -> str:
    return f"""\
You are a wiki maintainer. Process a source and integrate its knowledge into: {WIKI_DIR}/

Current wiki index:
{wiki_index or "(empty — this is the first ingest)"}

Steps:
1. Read or receive the source content.
2. Extract key entities, concepts, facts, and relationships.
3. Create new pages for significant entities/concepts not yet in the wiki.
4. For existing pages that need updating, read first, then write.
5. Never write index.md or log.md until all other pages are written.
6. Update {WIKI_DIR}/index.md to include any new or changed pages.
7. Append to {WIKI_DIR}/log.md: ## [DATE] ingest | <source title>
"""


def _query_prompt(wiki_index: str) -> str:
    return f"""\
You are a wiki researcher. Answer questions using: {WIKI_DIR}/

Current wiki index:
{wiki_index or "(empty — no pages ingested yet)"}

Steps:
1. Identify relevant pages from the index above.
2. Read relevant pages using read_file.
3. Synthesize a clear answer with citations to wiki pages.
4. If the answer reveals useful synthesis not yet in the wiki,
   indicate it could be filed as a new wiki page.
5. Never write index.md or log.md unless explicitly asked to file an answer.
"""


LINT_PROMPT = f"""\
You are a wiki health inspector. Find and report issues in: {WIKI_DIR}/

Steps:
1. Use list_wiki to discover all pages.
2. Read each page using read_file.
3. Flag: contradictions, orphan pages, stale claims, missing cross-references,
   concepts lacking their own page.
4. Write your full findings to the lint report path provided in the prompt.
5. Never write index.md or log.md until all other pages are written.
6. Append to {WIKI_DIR}/log.md: ## [DATE] lint | <issue count> issues found
7. Output a one-line summary: "<N> issues found."
"""

_SAVE_DECISION_PROMPT = """\
You just answered a query. Decide how to file this answer in the wiki.

Respond with ONLY valid JSON, no markdown, no explanation:
{
  "action": "new" | "update",
  "target": "<wiki/path/to/file.md or null if new>",
  "title": "<title for the new or updated page>"
}

Rules:
- "new" if this answer introduces synthesis not already in any wiki page.
- "update" if it clearly extends an existing page (provide its path in target).
- target must be null when action is "new".
"""


# ---------------------------------------------------------------------------
# ReAct loop — original signature preserved
# ---------------------------------------------------------------------------

def _react_loop(model, messages: list, tool_map: dict, max_iterations: int = MAX_ITERATIONS) -> list:
    for _ in range(max_iterations):
        response = model.invoke(messages)
        messages = [*messages, response]

        if not response.tool_calls:
            return messages

        for tc in response.tool_calls:
            tool_fn = tool_map.get(tc["name"])
            if tool_fn is None:
                result = f"[error] Unknown tool: {tc['name']}"
            else:
                try:
                    result = tool_fn.invoke(tc["args"])
                except Exception as e:
                    result = f"[error] {tc['name']} failed: {e}"
            messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

    raise MaxIterationsError(
        f"ReAct loop exceeded {max_iterations} iterations. "
        "Increase WIKI_MAX_ITERATIONS or simplify the operation."
    )


# ---------------------------------------------------------------------------
# Node factories — original signatures preserved
# ---------------------------------------------------------------------------

def build_ingest_node(model, tool_map: dict):
    def ingest_node(state: WikiState) -> WikiState:
        console.rule("[bold blue]ingest")

        fetched = state.get("fetched_content", "")
        if fetched:
            human_msg = HumanMessage(
                content=(
                    f"Ingest the following content into the wiki.\n\n"
                    f"Source URL: {state['input']}\n\n"
                    f"Content:\n{fetched}"
                )
            )
        else:
            human_msg = HumanMessage(content=f"Ingest this source file: {state['input']}")

        messages = _react_loop(model, [
            SystemMessage(content=_ingest_prompt(state["wiki_index"])),
            human_msg,
        ], tool_map)
        output = messages[-1].content
        console.print(f"[green]ingest complete[/green]")
        return {**state, "messages": messages, "output": output}

    return ingest_node


def build_query_node(model, tool_map: dict, prompter: Callable[[str], str] = input):
    def query_node(state: WikiState) -> WikiState:
        console.rule("[bold blue]query")

        messages = _react_loop(model, [
            SystemMessage(content=_query_prompt(state["wiki_index"])),
            HumanMessage(content=state["input"]),
        ], tool_map)
        answer = messages[-1].content
        console.print(answer)


        # Interactive prompter path (original): LLM signals filing opportunity
        if not state.get("save_output") and "could be filed" in answer.lower():
            reply = prompter("File this analysis as a wiki page? [y/N] ").strip().lower()
            if reply == "y":
                messages = _react_loop(model, [
                    *messages,
                    HumanMessage(content="Yes, please file this as a new wiki page."),
                ], tool_map)
                return {**state, "messages": messages, "output": messages[-1].content}

        # --save flag: structured LLM decision (non-interactive)
        if state.get("save_output"):
            console.rule("[bold yellow]save decision")
            decision_response = model.invoke([
                *messages,
                HumanMessage(content=f"{_SAVE_DECISION_PROMPT}\n\nAnswer to file:\n{answer}"),
            ])
            raw = decision_response.content.strip()
            try:
                decision = json.loads(raw)
            except json.JSONDecodeError:
                console.print(f"[red]save decision parse failed — skipping[/red]")
                return {**state, "messages": messages, "output": answer}

            action = decision.get("action")
            target = decision.get("target")
            title = decision.get("title", "untitled")

            if action == "new":
                path = f"{WIKI_DIR}/answers/{title.lower().replace(' ', '-')}.md"
                content = f"# {title}\n\n{answer}\n"
            elif action == "update" and target:
                path = target
                try:
                    existing = _read_existing(path)
                    content = existing + f"\n\n## Update — {datetime.now().strftime('%Y-%m-%d')}\n\n{answer}\n"
                except FileNotFoundError:
                    content = f"# {title}\n\n{answer}\n"
            else:
                console.print("[red]invalid save decision — skipping[/red]")
                return {**state, "messages": messages, "output": answer}

            save_messages = _react_loop(model, [
                SystemMessage(content=_query_prompt(state["wiki_index"])),
                HumanMessage(
                    content=(
                        f"File this answer as a wiki page.\n"
                        f"Write to: {path}\nContent:\n{content}\n\n"
                        f"Then update {WIKI_DIR}/index.md and append to {WIKI_DIR}/log.md."
                    )
                ),
            ], tool_map)
            console.print(f"[green]saved → {path}[/green]")
            return {**state, "messages": save_messages, "output": answer}

        return {**state, "messages": messages, "output": answer}

    return query_node


def _read_existing(path: str) -> str:
    from pathlib import Path
    return Path(path).read_text(encoding="utf-8")


def build_lint_node(model, tool_map: dict):
    def lint_node(state: WikiState) -> WikiState:
        console.rule("[bold blue]lint")
        report_path = f"{WIKI_DIR}/lint-{datetime.now().strftime('%Y-%m-%dT%H-%M')}.md"
        messages = _react_loop(model, [
            SystemMessage(content=LINT_PROMPT),
            HumanMessage(content=f"Lint the wiki. Write your report to: {report_path}"),
        ], tool_map)
        output = messages[-1].content
        console.print(output)
        return {**state, "messages": messages, "output": output}

    return lint_node
