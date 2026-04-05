import pytest
from unittest.mock import MagicMock, patch
from agent.tools import read_file, write_file, list_wiki, SUPPORTED_EXTENSIONS


# ---------------------------------------------------------------------------
# write_file / list_wiki — unchanged behaviour
# ---------------------------------------------------------------------------

def test_write_and_read(wiki_env):
    p = wiki_env / "page.md"
    write_file(str(p), "# Hello")
    assert read_file(str(p)) == "# Hello"


def test_write_creates_parents(wiki_env):
    p = wiki_env / "sub" / "dir" / "page.md"
    write_file(str(p), "content")
    assert p.exists()


def test_write_rejects_path_outside_wiki(wiki_env, tmp_path):
    outside = tmp_path / "sources" / "evil.md"
    with pytest.raises(ValueError, match="restricted to"):
        write_file(str(outside), "bad")


def test_list_wiki_returns_sorted_md(wiki_env):
    (wiki_env / "b.md").write_text("b")
    (wiki_env / "a.md").write_text("a")
    (wiki_env / "ignore.txt").write_text("x")
    result = list_wiki(str(wiki_env))
    assert result == [str(wiki_env / "a.md"), str(wiki_env / "b.md")]


def test_list_wiki_recurses(wiki_env):
    sub = wiki_env / "entities"
    sub.mkdir()
    (sub / "foo.md").write_text("foo")
    result = list_wiki(str(wiki_env))
    assert str(sub / "foo.md") in result


def test_read_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_file(str(tmp_path / "missing.md"))


# ---------------------------------------------------------------------------
# read_file — format dispatch
# ---------------------------------------------------------------------------

def test_read_file_text_extensions(tmp_path):
    for ext in [".md", ".txt", ".rst", ".html"]:
        p = tmp_path / f"file{ext}"
        p.write_text(f"content for {ext}", encoding="utf-8")
        assert read_file(str(p)) == f"content for {ext}"


def test_read_file_rejects_unsupported_extension(tmp_path):
    p = tmp_path / "file.xyz"
    p.write_text("content")
    with pytest.raises(ValueError, match="Unsupported file type"):
        read_file(str(p))


def test_read_file_unsupported_extension_lists_supported(tmp_path):
    p = tmp_path / "file.xlsx"
    p.write_text("content")
    with pytest.raises(ValueError) as exc:
        read_file(str(p))
    for ext in SUPPORTED_EXTENSIONS:
        assert ext in str(exc.value)


def test_read_file_pdf_delegates_to_docling(tmp_path):
    p = tmp_path / "doc.pdf"
    p.write_bytes(b"%PDF-1.4 fake")

    mock_doc = MagicMock()
    mock_doc.document.export_to_markdown.return_value = "# Extracted PDF"
    mock_converter = MagicMock()
    mock_converter.convert.return_value = mock_doc

    with patch("agent.tools._converter_factory", return_value=mock_converter):
        result = read_file(str(p))

    assert result == "# Extracted PDF"
    mock_converter.convert.assert_called_once_with(str(p))


def test_read_file_docx_delegates_to_docling(tmp_path):
    p = tmp_path / "doc.docx"
    p.write_bytes(b"PK fake docx")

    mock_doc = MagicMock()
    mock_doc.document.export_to_markdown.return_value = "# Extracted DOCX"
    mock_converter = MagicMock()
    mock_converter.convert.return_value = mock_doc

    with patch("agent.tools._converter_factory", return_value=mock_converter):
        result = read_file(str(p))

    assert result == "# Extracted DOCX"


def test_read_file_raises_helpful_error_if_docling_missing(tmp_path):
    p = tmp_path / "doc.pdf"
    p.write_bytes(b"%PDF-1.4 fake")

    with patch.dict("sys.modules", {"docling.document_converter": None}):
        with pytest.raises(ImportError, match="pip install docling"):
            read_file(str(p))
