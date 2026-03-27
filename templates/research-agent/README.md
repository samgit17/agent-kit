# Research Agent

**A multi-step LangGraph agent that researches a topic, synthesises findings, and returns a structured report.**

Give it a question. It plans search queries, gathers information across multiple steps, reasons over the results, verifies its answer, and outputs a cited markdown report. Runs locally with Ollama or against OpenAI — switchable via `.env`.

---

## What It Does

```
User question
  → Planner        (breaks question into 3–5 search queries)
  → Searcher       (executes each query via Tavily or DuckDuckGo)
  → Synthesiser    (reasons over gathered results, builds draft answer)
  → Verifier       (scores faithfulness, flags uncertainty)
  → Formatter      (outputs structured markdown report with citations)
```

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Edit .env — set LLM_PROVIDER, API keys, and search provider

# 3. Run
python run.py --query "What are the latest developments in agentic AI security?"

# Or interactive mode
python run.py
```

## Configuration

All configuration is in `.env`. See `.env.example` for all options.

| Variable | Required | Description |
|---|---|---|
| `LLM_PROVIDER` | ✅ | `openai` or `ollama` |
| `OPENAI_API_KEY` | if OpenAI | Your OpenAI API key |
| `OPENAI_MODEL` | if OpenAI | e.g. `gpt-4o` |
| `OLLAMA_BASE_URL` | if Ollama | e.g. `http://localhost:11434` |
| `OLLAMA_MODEL` | if Ollama | e.g. `llama3.1:8b` |
| `SEARCH_PROVIDER` | ✅ | `tavily` or `duckduckgo` |
| `TAVILY_API_KEY` | if Tavily | Your Tavily API key |
| `MAX_SEARCH_QUERIES` | ❌ | Default: `5` |
| `MAX_SEARCH_RESULTS` | ❌ | Default: `3` per query |
| `VERIFIER_THRESHOLD` | ❌ | Min confidence to accept answer. Default: `0.7` |

## Output

The agent writes a markdown report to `output/report.md`:

```markdown
# Research Report: <your question>

## Summary
...

## Findings

### Finding 1 — <topic>
...
[Source: <url>]

## Confidence Score: 0.87
## Queries used: 4
## Sources: 9
```

## Extending This Template

- **Add a new tool** — create a file in `tools/` and register it in `graph.py`
- **Change the LLM** — update `LLM_PROVIDER` in `.env`; the `llm_client.py` abstraction handles the rest
- **Add memory** — replace the in-memory state with a Redis-backed checkpointer in `graph.py`
- **Add a UI** — the `run.py` entry point exposes a `run_research(query: str)` function you can call from any web framework

## Architecture

See [`graph.py`](./graph.py) for the full LangGraph state graph definition.

```
ResearchState
  ├── query: str
  ├── search_queries: list[str]
  ├── search_results: list[SearchResult]
  ├── draft_answer: str
  ├── citations: list[Citation]
  ├── confidence_score: float
  ├── uncertainty_flagged: bool
  └── final_report: str
```
