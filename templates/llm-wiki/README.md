# LLM Wiki

**A LangGraph agent that builds and maintains a personal knowledge base from your documents.**

Drop in a source. The agent reads it, extracts what matters, and integrates it into a structured wiki of interlinked markdown files — updating pages, cross-referencing entities, and keeping everything consistent. Ask questions against it. Run a health-check when it gets large.

---

## What It Does

```
ingest <source>
  → Read source document
  → Extract entities, concepts, relationships
  → Create / update wiki pages
  → Update index.md  (last)
  → Append to log.md (last)

query "<question>"
  → Read index.md
  → Fetch relevant pages
  → Synthesise answer with citations
  → Optionally file analysis as a new wiki page

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
pip install -r requirements-dev.txt  # includes pytest

# 2. Configure
cp .env.example .env
# Edit .env — set LLM_API_KEY and LLM_MODEL

# 3. Ingest your first source
python run.py ingest sources/my-article.md   # .md .txt .rst .html
python run.py ingest sources/paper.pdf          # PDF via docling
python run.py ingest sources/report.docx        # DOCX via docling

# 4. Ask a question
python run.py query "What are the main themes across my sources?"

# 5. Health-check
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
| `SOURCES_DIR` | ❌ | `sources` | Directory for your raw source documents |
| `WIKI_MAX_ITERATIONS` | ❌ | `10` | Max ReAct loop iterations per operation |

---

## Wiki Structure

```
wiki/
├── index.md          # catalog of all pages
├── log.md            # append-only operation history
├── overview.md       # high-level synthesis
├── entities/         # people, orgs, systems
├── concepts/         # ideas, patterns, frameworks
├── sources/          # one summary per ingested document
└── lint-<timestamp>.md
```

See [`SCHEMA.md`](./SCHEMA.md) for full page format and conventions.

---

## Extending This Template

- **Different output formats** — add a node that converts wiki pages to Marp slides or matplotlib charts
- **Batch ingest** — wrap `ingest_node` in a parallel subgraph for processing many sources at once
- **Search** — replace `index.md` navigation with BM25 or vector search as the wiki grows beyond ~100 pages
- **Obsidian** — point `WIKI_DIR` at your Obsidian vault; the agent writes standard markdown with wikilinks

---

## Stack

LangGraph · LangChain · Python · OpenAI-compatible LLMs
