# Contributing a New Template

Each template lives in its own directory under `templates/` and must be fully self-contained.

---

## Template Requirements

Every template must include:

| File | Purpose |
|---|---|
| `README.md` | Standalone docs — what it does, how to run it, how to extend it |
| `requirements.txt` | All dependencies — no imports from outside the template directory |
| `.env.example` | All required env vars with placeholder values and comments |
| `run.py` | Single entry point — `python run.py` must produce visible output |
| `graph.py` | LangGraph state graph definition |
| `agents/` | One file per agent node |
| `tools/` | External tool wrappers (search, scraper, etc.) |

---

## Checklist Before Submitting a PR

- [ ] Template runs end-to-end with `python run.py` after following the README
- [ ] Works with both `LLM_PROVIDER=openai` and `LLM_PROVIDER=ollama`
- [ ] No imports from other templates or a `shared/` directory
- [ ] `.env` is in `.gitignore` — no secrets committed
- [ ] Root `README.md` table updated with the new template
- [ ] `docs/module-catalog.md` updated if new reusable patterns were introduced

---

## Template Naming Convention

```
templates/<noun>-agent/        ← e.g. research-agent, scraper-agent
templates/<noun>-pipeline/     ← e.g. ingest-pipeline
templates/<verb>-bot/          ← e.g. support-bot
```

Keep names short, lowercase, hyphenated.
