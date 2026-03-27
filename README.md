# AgentKit 🤖

**Pre-built LangGraph agent templates for indie hackers and micro-SaaS builders.**

Stop rebuilding the same agent scaffolding from scratch. AgentKit gives you production-ready, fully self-contained templates you can clone, configure, and ship.

---

## Templates

| Template | What it does | LLM | Status |
|---|---|---|---|
| [research-agent](./templates/research-agent/) | Multi-step web research → structured report | Ollama / OpenAI | ✅ Ready |
| *coming soon* | | | 🔜 |
| *coming soon* | | | 🔜 |

---

## Philosophy

- **Self-contained** — every template has its own `requirements.txt`, `.env.example`, and `README.md`. No shared dependencies between templates. Clone one, delete the rest.
- **LangGraph-first** — all orchestration uses [LangGraph](https://github.com/langchain-ai/langgraph). No custom agent loops to maintain.
- **Provider-agnostic** — templates support both local (Ollama) and cloud (OpenAI) LLMs, switchable via `.env`.
- **Runnable in under 5 minutes** — `pip install → configure .env → python run.py`.

---

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/samgit17/agent-kit.git
cd agent-kit

# 2. Pick a template
cd templates/research-agent

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure
cp .env.example .env
# Edit .env — set LLM_PROVIDER and API key or Ollama URL

# 5. Run
python run.py
```

---

## Adding a Template

See [CONTRIBUTING.md](./CONTRIBUTING.md).

---

## Author

Built by [Samuel Beera](https://www.linkedin.com/in/samuelbeera) — Enterprise AI Architect, LangGraph practitioner.
