# LLM Wiki

**A LangGraph agent that builds and maintains a personal knowledge base from your documents and URLs.**

Drop in a source — file or URL. The agent reads it, extracts what matters, and integrates it into a structured wiki of interlinked markdown files — updating pages, cross-referencing entities, and keeping everything consistent. Ask questions against it. Save answers back into the wiki. Run a health-check when it gets large.

---

## What It Does

```
ingest <file_or_url>
  → Fetch URL content (httpx → Playwright fallback) or read file
  → Extract entities, concepts, relationships
  → Create / update wiki pages
  → Update index.md  (last)
  → Append to log.md (last)

query "<question>" [--save]
  → Read index.md
  → Fetch relevant pages
  → Synthesise answer with citations
  → Optionally file answer back into wiki
      without --save: prompts if LLM signals a filing opportunity
      with --save:    LLM decides new page vs. update existing (non-interactive)

lint
  → Scan all wiki pages
  → Flag contradictions, orphans, missing cross-references
  → Write versioned lint report (wiki/lint-<timestamp>.md)
  → Print findings to stdout
```

---

## Quickstart

```bash
# 1. Install
pip install -r requirements.txt

# 2. Install Playwright browser (required for JS-heavy URL ingestion)
playwright install chromium

# 3. Configure
cp .env.example .env
# Edit .env — set LLM_API_KEY and LLM_MODEL

# 4. Ingest a file
python run.py ingest sources/my-article.md

# 5. Ingest a URL
python run.py ingest https://example.com/article

# 6. Ask a question
python run.py query "What are the main themes across my sources?"

# 7. Ask and save the answer back into the wiki
python run.py query "Compare X and Y" --save

# 8. Health-check
python run.py lint
```

---

## Configuration

| Variable | Required | Default | Description |
|---|---|---|---|
| `LLM_API_KEY` | ✅ | — | API key for your LLM provider |
| `LLM_MODEL` | ✅ | — | Model name e.g. `gpt-4o`, `llama3.1:8b` |
| `LLM_BASE_URL` | ❌ | OpenAI | Override for Ollama (`http://localhost:11434/v1`) or vLLM |
| `WIKI_DIR` | ❌ | `wiki` | Directory where the agent writes wiki pages |
| `WIKI_MAX_ITERATIONS` | ❌ | `10` | Max ReAct loop iterations per operation |

---

## Wiki Structure

```
wiki/
├── index.md              # catalog of all pages
├── log.md                # append-only operation history
├── overview.md           # high-level synthesis
├── entities/             # people, orgs, systems
├── concepts/             # ideas, patterns, frameworks
├── sources/              # one summary per ingested document
├── answers/              # saved query answers (--save)
└── lint-<timestamp>.md
```

See [`SCHEMA.md`](./SCHEMA.md) for full page format and conventions.

---

## URL Ingestion

Static pages are fetched with `httpx`. JavaScript-heavy or dynamic pages fall back to Playwright automatically — no configuration needed.

```bash
python run.py ingest https://some-article.com
python run.py ingest https://docs.some-library.com/guide
```

---

## Compound Loop (--save)

Every `query --save` files the answer back into the wiki. The LLM decides whether to create a new page or extend an existing one. Knowledge accumulates across sessions.

```bash
python run.py query "What gaps exist in my understanding of X?" --save
python run.py query "Summarise the key tradeoffs between A and B" --save
```

Run `lint` periodically — compounding errors are real, and the health-check catches them before they pile up.

---

## Extending This Template

- **Batch ingest** — wrap `ingest_node` in a loop over a directory of files or a list of URLs
- **Search** — replace `index.md` navigation with BM25 or vector search as the wiki grows beyond ~100 pages
- **Obsidian** — point `WIKI_DIR` at your Obsidian vault; the agent writes standard markdown with wikilinks
- **Scheduled lint** — cron the lint operation weekly to catch drift automatically

---

## Stack

LangGraph · LangChain · httpx · BeautifulSoup · Playwright · Python · OpenAI-compatible LLMs