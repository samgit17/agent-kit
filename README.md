# AgentKit 🤖

**Pre-built LangGraph agent templates for indie hackers and micro-SaaS builders.**

Stop rebuilding the same agent scaffolding from scratch. AgentKit gives you production-ready, fully self-contained templates you can clone, configure, and ship.

---

## Templates

| Template | What it does | LLM | Status |
|---|---|---|---|
| [research-agent](./templates/research-agent/) | Web research → report · ML experiment ratchet loop (Karpathy-style) | Ollama / OpenAI | ✅ Ready |
| [prompt-optimizer](./templates/prompt-optimizer/) | Ratchet loop for skill/prompt files — auto-improves against binary eval criteria | Ollama / OpenAI / Anthropic | ✅ Ready |
| [llm-wiki](./templates/llm-wiki/) | Builds and maintains a personal knowledge base from documents | Ollama / OpenAI | ✅ Ready |
| *coming soon* | | | 🔜 |

---

## Philosophy

- **Self-contained** — every template has its own `requirements.txt`, `.env.example`, and `README.md`. No shared dependencies between templates. Clone one, delete the rest.
- **LangGraph-first** — all orchestration uses [LangGraph](https://github.com/langchain-ai/langgraph). No custom agent loops to maintain.
- **Provider-agnostic** — templates support local (Ollama) and cloud (OpenAI / Anthropic) LLMs, switchable via `.env`.
- **Runnable in under 5 minutes** — `pip install → configure .env → python run.py`.

---

## Quick Start
```bash
# 1. Clone the repo
git clone https://github.com/samgit17/agent-kit.git
cd agent-kit

# 2. Pick a template
cd templates/prompt-optimizer   # or research-agent

# 3. Install dependencies
pip install -r requirements.txt # add -r requirements-dev.txt for tests

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

**Samuel Beera** — building in public
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://www.linkedin.com/in/sambeera)  — Enterprise AI Architect, LangGraph practitioner.

## Why agent-kit?
LangGraph is powerful but the setup cost kills momentum. 
AgentKit removes that tax.

## Status
🚧 Active development — launching May 30, 2026

## Follow the Build
I'm documenting every step publicly on LinkedIn — architecture decisions, 
mistakes, and weekly progress. Follow along if you're building with AI agents.

## Support
If this saves you time, drop a ⭐ — it helps others find it.



