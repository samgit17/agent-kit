import pytest
from agent.tools_lc import read_file, write_file, list_wiki, WIKI_TOOLS


def test_read_file_delegates(wiki_env, tmp_path):
    p = tmp_path / "source.md"
    p.write_text("# Source")
    assert read_file.invoke({"path": str(p)}) == "# Source"


def test_write_file_delegates_and_confirms(wiki_env):
    p = wiki_env / "page.md"
    result = write_file.invoke({"path": str(p), "content": "# Page"})
    assert p.read_text() == "# Page"
    assert str(p) in result


def test_write_file_rejects_outside_wiki(wiki_env, tmp_path):
    outside = tmp_path / "evil.md"
    with pytest.raises(ValueError, match="restricted to"):
        write_file.invoke({"path": str(outside), "content": "bad"})


def test_list_wiki_delegates(wiki_env):
    (wiki_env / "a.md").write_text("a")
    (wiki_env / "b.md").write_text("b")
    result = list_wiki.invoke({})
    assert str(wiki_env / "a.md") in result
    assert str(wiki_env / "b.md") in result


def test_wiki_tools_export():
    names = {t.name for t in WIKI_TOOLS}
    assert names == {"read_file", "write_file", "list_wiki"}
