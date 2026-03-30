# Research Agent

Inspired by Karpathy's autoresearch pattern. `program.md` is the only file you edit.

## Backends

| Backend | What it does | Metric | Ships ready? |
|---|---|---|---|
| `web` | Multi-step web research → markdown report | Confidence score | ✅ Yes — plug and play |
| `ml_experiment` | Ratchet loop: propose → train → eval → keep/revert | val_bpb | ✅ Yes — includes train.py + dataset |
| `prompt_engineering` | Optimize system prompts against an eval script | Accuracy / custom | 🔜 Planned — bring your own task + dataset |
| `code_optimization` | Optimize code against a benchmark script | Latency / custom | 🔜 Planned — bring your own code + benchmark |

### What "bring your own task" means

`web` and `ml_experiment` ship everything needed to run. The planned backends provide the ratchet loop infrastructure but require you to supply:

| Backend | You must provide |
|---|---|
| `prompt_engineering` | A labelled test dataset · An eval script that prints `METRIC: <float>` · A `system_prompt.txt` for the agent to edit |
| `code_optimization` | The code file for the agent to edit · A benchmark script that prints `METRIC: <float>` · A running local environment |

The agent handles the loop. You define the problem.

## What Can the Ratchet Loop Optimize?

Any task with a **fast, objective, measurable metric** is a candidate. The agent can't optimize what it can't measure — if the feedback loop takes days or requires human judgment, it won't work.

| Use Case | Metric | Loop speed | Viable? |
|---|---|---|---|
| ML model training (nanochat) | val_bpb | ~10 min/exp | ✅ Yes — ships with this template |
| System prompt quality | eval accuracy | ~30 sec/exp | ✅ Yes — planned backend |
| Code / API performance | p95 latency | ~30 sec/exp | ✅ Yes — planned backend |
| Marketing copy / A/B tests | conversion rate | days/exp | ❌ No — feedback loop too slow |
| Algorithmic trading | Sharpe ratio | fast on historical data | ❌ No — out of scope for this template |
| Pricing optimization | revenue | days/exp | ❌ No — feedback loop too slow |

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate  # Linux/Mac
python -m venv .venv && .venv\Scripts\activate      # Windows
pip install -r requirements.txt
cp .env.example .env          # configure LLM_PROVIDER and model
# copy the backend example you want:
copy program.web.md program.md           # web research
copy program.ml_experiment.md program.md # ML experiment
python run.py
```

## Folder Structure

```
research-agent/
├── program.md                    ← ONLY file you edit to control the agent
├── program.web.md                ← example: web backend
├── program.ml_experiment.md      ← example: ML experiment backend
├── run.py                        ← entry point: python run.py
├── program_parser.py             ← parses program.md into typed config
├── requirements.txt
├── .env.example                  ← copy to .env and configure
├── .gitignore
├── backends/
│   ├── llm.py                    ← shared LLM factory (Ollama / OpenAI)
│   ├── log.py                    ← shared Rich console + log()
│   ├── web/
│   │   ├── graph.py              ← LangGraph: planner→searcher→synthesiser→verifier→formatter
│   │   ├── models.py             ← ResearchState
│   │   └── nodes.py              ← node functions
│   └── ml_experiment/
│       ├── graph.py              ← LangGraph: proposer→executor→evaluator→committer→reporter
│       ├── models.py             ← MLResearchState, ExperimentRecord
│       ├── nodes.py              ← ratchet loop node functions
│       └── train.py              ← nanochat training script (agent edits this only)
├── output/
│   └── report.md                 ← web backend writes here (git-ignored)
└── tests/
    ├── test_log.py
    ├── test_run_config.py
    ├── test_web_nodes.py
    ├── test_ml_graph.py
    └── test_known_constraints.py
```

## Console Output

**Web backend:**
```
╭─ Research Agent v2 ─────────────────────────────────────────────────╮
│ Backend:  web                                                        │
│ Goal:     What are best practices for securing multi-agent LLMs?    │
│ LLM: OLLAMA  ·  Search: DUCKDUCKGO  ·  Started: 2026-03-29 09:15:22│
╰──────────────────────────────────────────────────────────────────────╯
[planner]     Generated 3 queries
[searcher]    Retrieved 9 results across 3 queries
[synthesiser] Draft complete (iteration 1/3)
[verifier]    Confidence: 68%  ✅
[formatter]   Report ready
[run]         Report saved to output/report.md
[run]         Finished at 2026-03-29 09:16:45 — elapsed 0:01:23
```

**ML experiment backend:**
```
╭─ Research Agent v2 ─────────────────────────────────────────────────╮
│ Backend:  ml_experiment                                              │
│ Goal:     Improve val_bpb on nanochat. Baseline: measure first.     │
│ LLM: OLLAMA  ·  Started: 2026-03-29 09:20:15                        │
╰──────────────────────────────────────────────────────────────────────╯
[run]       Resuming from git history — best val_bpb so far: 2.1780
[proposer]  Exp 1: Switch learning rate schedule to warmup_cosine
[evaluator] Exp 1 ✅ inf → 2.2186 | Switch learning rate schedule to warmup_cosine
[proposer]  Exp 2: Increase batch size from 8 to 16
[evaluator] Exp 2 ❌ 2.2688 (best=2.2186) | Increase batch size from 8 to 16
[reporter]  Results written to backends/ml_experiment/results.md
[run]       Finished at 2026-03-29 11:11:42 — elapsed 0:51:27
```

## ML Experiment Backend — Setup

The ratchet loop uses `git checkout` to revert bad experiments. `train.py`
must be committed before the first run:

```bash
cd backends/ml_experiment
git init
git add train.py
git commit -m "baseline"
```

Subsequent runs automatically resume from the best val_bpb in the git history —
no manual baseline tracking needed.

### GPU Selection

Set `gpu:` in `program.md` to select the CUDA device index:

```markdown
## constraints
gpu: 0    # 0 = 4070 Super (12GB) — use DEPTH<=4, batch_size<=8
gpu: 1    # 1 = 5060 Ti  (16GB)  — DEPTH<=6, batch_size<=16 are safe
```

### PyTorch for 5060 Ti (Blackwell)

Stable PyTorch does not support Blackwell. Install nightly:

```bash
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128
```

4070 Super works on stable PyTorch ≥2.4.

## program.md Reference

```markdown
## backend
web | ml_experiment

## goal
<what you want to achieve>

## success_criteria      # web only
- criterion 1

## directions            # ml_experiment only
- things to try

## known_constraints     # ml_experiment only — LLM will never propose these
- Do not increase CONTEXT_LEN — timeouts at 10min budget
- Do not increase DEPTH beyond 6 — OOM risk

## constraints
# web
max_iterations: 3
confidence_threshold: 0.6   # 0.0-1.0 — lower for local models

# ml_experiment
gpu: 0
minutes_per_experiment: 10
max_experiments: 20
revert_on_no_improvement: true
vram_budget_gb: 12
```

## Example program.md Files

```bash
copy program.web.md program.md           # Windows
copy program.ml_experiment.md program.md
```

### Web research
```markdown
## backend
web

## goal
What are the best practices for securing multi-agent LLM systems in 2025?

## success_criteria
- At least 5 credible sources
- Prefer sources from last 6 months
- Focus on agentic attack surfaces, not general LLM safety

## constraints
max_iterations: 3
confidence_threshold: 0.6
```

### ML experiment
```markdown
## backend
ml_experiment

## goal
Improve val_bpb on nanochat. Baseline: measure first.

## directions
- Learning rate schedule (cosine vs flat vs warmup_cosine)
- DEPTH param (4 vs 6)
- Batch size (8 vs 16)
- Weight decay values (0.1, 0.01, 0.001)

## known_constraints
- Do not increase CONTEXT_LEN — timeouts at 10min budget on home lab GPU

## constraints
gpu: 1
minutes_per_experiment: 10
max_experiments: 20
revert_on_no_improvement: true
vram_budget_gb: 16
```

### Prompt engineering (planned)
```markdown
## backend
prompt_engineering

## goal
Maximize accuracy on customer support ticket classification.
Baseline: 71% on held-out test set.

## directions
- Vary instruction detail (terse vs verbose)
- Add few-shot examples (0 vs 3 vs 5)
- Try chain-of-thought prefix
- Rephrase persona (assistant vs expert vs classifier)

## constraints
eval_script: evals/classify_tickets.py
metric: accuracy
max_experiments: 30
revert_on_no_improvement: true
```

### Code optimization (planned)
```markdown
## backend
code_optimization

## goal
Reduce p95 latency of /api/search endpoint.
Baseline: 142ms.

## directions
- Add result caching (Redis vs in-memory)
- Optimize DB query (index, reduce joins)
- Batch embedding lookups

## constraints
benchmark_script: benchmarks/search_benchmark.py
metric: p95_ms
lower_is_better: true
max_experiments: 20
revert_on_no_improvement: true
timeout_seconds: 30
```

## Tests

```bash
python -m pytest tests/ -v
# 22 tests — log, run config, web nodes, ml graph, known constraints
```
