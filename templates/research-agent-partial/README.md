# Research Agent v2

Inspired by Karpathy's autoresearch pattern. `program.md` is the only file you edit.

## Backends

| Backend | What it does | Metric |
|---|---|---|
| `web` | Multi-step web research → markdown report | Confidence score |
| `ml_experiment` | Ratchet loop: propose → train → eval → keep/revert | val_bpb |

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # configure LLM_PROVIDER and model
# edit program.md             # set backend, goal, constraints
python run.py
```

## Folder Structure

```
research-agent/                   ← rename from research-agent-v2 when replacing v1
├── program.md                    ← ONLY file you edit to control the agent
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
│       └── train.py             ← nanochat training script (agent edits this only)
├── output/
│   └── report.md                 ← web backend writes here (git-ignored)
└── tests/
    ├── test_log.py
    └── test_run_config.py
```

## Console Output

```
╭─ Research Agent v2 ──────────────────────╮
│ Backend:  web                             │
│ Goal:     What are best practices for…   │
│ LLM: OLLAMA  ·  Search: DUCKDUCKGO       │
╰───────────────────────────────────────────╯
[planner]     Generated 3 queries
[searcher]    Retrieved 9 results across 3 queries
[synthesiser] Draft complete (iteration 1/3)
[verifier]    Confidence: 82%  ✅
[formatter]   Report ready
[run]         Report saved to output/report.md
```

Web backend saves the final report to `output/report.md`.

## ML Experiment Backend — Setup

The ratchet loop uses `git checkout` to revert bad experiments. `train.py`
must be committed before the first run:

```bash
cd backends/ml_experiment
git init
git add train.py
git commit -m "baseline"
```

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

## constraints
# web
max_iterations: 3

# ml_experiment
gpu: 0
minutes_per_experiment: 10
max_experiments: 20
revert_on_no_improvement: true
vram_budget_gb: 12
```

## Tests

```bash
python -m pytest tests/ -v
```
