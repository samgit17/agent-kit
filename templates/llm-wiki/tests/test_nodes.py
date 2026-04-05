import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent.exceptions import MaxIterationsError


# ---------------------------------------------------------------------------
# Fake model helpers
# ---------------------------------------------------------------------------

class FakeModel:
    """Returns a sequence of AIMessages on successive invoke() calls."""

    def __init__(self, responses: list[AIMessage]):
        self._responses = iter(responses)

    def invoke(self, messages):
        return next(self._responses)


def ai(content: str, tool_calls: list = None) -> AIMessage:
    msg = AIMessage(content=content)
    msg.tool_calls = tool_calls or []
    return msg


def tc(name: str, args: dict, id: str = "tc1") -> dict:
    return {"name": name, "args": args, "id": id}


# ---------------------------------------------------------------------------
# _react_loop
# ---------------------------------------------------------------------------

def test_react_loop_stops_on_no_tool_calls(wiki_env):
    from agent.nodes import _react_loop
    from agent.tools_lc import WIKI_TOOLS
    tool_map = {t.name: t for t in WIKI_TOOLS}
    model = FakeModel([ai("Done.")])
    original = [HumanMessage(content="go")]
    result = _react_loop(model, original, tool_map, max_iterations=5)
    assert result[-1].content == "Done."
    assert len(original) == 1  # input list not mutated


def test_react_loop_does_not_mutate_input(wiki_env):
    from agent.nodes import _react_loop
    from agent.tools_lc import WIKI_TOOLS
    tool_map = {t.name: t for t in WIKI_TOOLS}
    model = FakeModel([ai("Done.")])
    original = [HumanMessage(content="go")]
    _react_loop(model, original, tool_map, max_iterations=5)
    assert len(original) == 1


def test_react_loop_raises_on_max_iterations(wiki_env):
    from agent.nodes import _react_loop
    from agent.tools_lc import WIKI_TOOLS
    tool_map = {t.name: t for t in WIKI_TOOLS}
    infinite = [ai("looping", [tc("list_wiki", {})]) for _ in range(12)]
    model = FakeModel(infinite)
    with pytest.raises(MaxIterationsError):
        _react_loop(model, [HumanMessage(content="go")], tool_map, max_iterations=3)


def test_react_loop_executes_tool_and_appends_result(wiki_env):
    from agent.nodes import _react_loop
    from agent.tools_lc import WIKI_TOOLS
    tool_map = {t.name: t for t in WIKI_TOOLS}
    model = FakeModel([
        ai("calling tool", [tc("list_wiki", {})]),
        ai("Done."),
    ])
    messages = _react_loop(model, [HumanMessage(content="go")], tool_map, max_iterations=5)
    tool_results = [m for m in messages if isinstance(m, ToolMessage)]
    assert len(tool_results) == 1


# ---------------------------------------------------------------------------
# ingest_node
# ---------------------------------------------------------------------------

def test_ingest_node_updates_state(wiki_env, base_state):
    from agent.nodes import build_ingest_node
    from agent.tools_lc import WIKI_TOOLS
    tool_map = {t.name: t for t in WIKI_TOOLS}
    model = FakeModel([ai("Ingest complete.")])
    node = build_ingest_node(model, tool_map)
    result = node(base_state("ingest", input_="sources/test.md"))
    assert result["output"] == "Ingest complete."
    assert len(result["messages"]) > 0


# ---------------------------------------------------------------------------
# query_node
# ---------------------------------------------------------------------------

def test_query_node_no_filing_prompt_when_not_indicated(base_state):
    from agent.nodes import build_query_node
    from agent.tools_lc import WIKI_TOOLS
    tool_map = {t.name: t for t in WIKI_TOOLS}
    model = FakeModel([ai("Here is your answer.")])
    prompted = []
    node = build_query_node(model, tool_map, prompter=lambda _: prompted.append(True) or "n")
    result = node(base_state("query", input_="What is X?"))
    assert result["output"] == "Here is your answer."
    assert prompted == []


def test_query_node_prompts_user_when_filing_indicated(base_state):
    from agent.nodes import build_query_node
    from agent.tools_lc import WIKI_TOOLS
    tool_map = {t.name: t for t in WIKI_TOOLS}
    model = FakeModel([
        ai("Here is your answer, which could be filed as a new wiki page."),
        ai("Filed successfully."),
    ])
    replies = ["y"]
    node = build_query_node(model, tool_map, prompter=lambda _: replies.pop(0))
    result = node(base_state("query", input_="Compare X and Y"))
    assert result["output"] == "Filed successfully."


def test_query_node_skips_filing_on_no(base_state):
    from agent.nodes import build_query_node
    from agent.tools_lc import WIKI_TOOLS
    tool_map = {t.name: t for t in WIKI_TOOLS}
    model = FakeModel([ai("Answer that could be filed as a new wiki page.")])
    node = build_query_node(model, tool_map, prompter=lambda _: "n")
    result = node(base_state("query", input_="What is X?"))
    assert result["output"] == "Answer that could be filed as a new wiki page."


# ---------------------------------------------------------------------------
# lint_node
# ---------------------------------------------------------------------------

def test_lint_node_returns_output(wiki_env, base_state):
    from agent.nodes import build_lint_node
    from agent.tools_lc import WIKI_TOOLS
    tool_map = {t.name: t for t in WIKI_TOOLS}
    model = FakeModel([ai("3 issues found.")])
    node = build_lint_node(model, tool_map)
    result = node(base_state("lint"))
    assert result["output"] == "3 issues found." 
