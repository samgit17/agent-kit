"""
Pure I/O functions. No LangChain imports. All file writes are path-confined to WIKI_DIR.
fetch_url: httpx first, Playwright fallback on empty body or non-200.
"""
from pathlib import Path

from agent.config import WIKI_DIR

SUPPORTED_EXTENSIONS = {".md", ".txt", ".rst", ".html", ".pdf", ".docx"}


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _safe_wiki_path(path: str) -> Path:
    wiki_root = Path(WIKI_DIR).resolve()
    target = Path(path).resolve()
    if not str(target).startswith(str(wiki_root)):
        raise ValueError(f"Write path restricted to {WIKI_DIR}: {path}")
    return target


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def _converter_factory(path: str):
    try:
        from docling.document_converter import DocumentConverter
    except ImportError:
        raise ImportError("pip install docling to read PDF/DOCX files")
    return DocumentConverter()


def _read_with_docling(path: str) -> str:
    converter = _converter_factory(path)
    result = converter.convert(path)
    return result.document.export_to_markdown()


def read_file(path: str) -> str:
    p = Path(path)
    ext = p.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{ext}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )
    if ext in {".pdf", ".docx"}:
        return _read_with_docling(path)
    return p.read_text(encoding="utf-8")


def write_file(path: str, content: str) -> str:
    target = _safe_wiki_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return f"Written: {path}"


def list_wiki(wiki_dir: str = WIKI_DIR) -> str:
    wiki_root = Path(wiki_dir)
    if not wiki_root.exists():
        return "(wiki directory does not exist yet)"
    files = sorted(wiki_root.rglob("*.md"))
    if not files:
        return "(no wiki pages yet)"
    return [str(f) for f in files]


# ---------------------------------------------------------------------------
# URL fetching — httpx first, Playwright fallback
# ---------------------------------------------------------------------------

def _is_likely_dynamic(html: str) -> bool:
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        body = soup.find("body")
        return not body or len(body.get_text(strip=True)) < 200
    except Exception:
        return False


def _extract_text(html: str) -> str:
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    main = soup.find("article") or soup.find("main") or soup.find("body")
    return main.get_text(separator="\n", strip=True) if main else soup.get_text(strip=True)


def _fetch_with_httpx(url: str) -> tuple:
    import httpx
    resp = httpx.get(url, follow_redirects=True, timeout=15)
    return resp.status_code, resp.text


def _fetch_with_playwright(url: str) -> str:
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url, wait_until="networkidle", timeout=30000)
        html = page.content()
        browser.close()
    return html


def fetch_url(url: str) -> str:
    try:
        status, html = _fetch_with_httpx(url)
        if status == 200 and not _is_likely_dynamic(html):
            return _extract_text(html)
    except Exception:
        pass
    try:
        html = _fetch_with_playwright(url)
        return _extract_text(html)
    except Exception as e:
        return f"[fetch_url error] Could not retrieve {url}: {e}"
