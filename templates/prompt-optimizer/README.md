# Prompt Optimizer

Applies Karpathy's autoresearch ratchet loop to any skill/prompt file.
The agent edits `skill.md`, a locked eval script scores outputs against
binary criteria, keeps improvements, reverts failures via git.

Inspired by the pattern described in
[The Ultimate Autoresearch Guide](https://news.aakashg.com).

## How It Works

```
Phase 1 (first run only):
  eval_builder  -- LLM writes eval.py from your criteria, then git-locks it
  baseline      -- runs eval.py once to establish starting score

Phase 2 (every run):
  proposer  -- LLM reads skill.md + history -> proposes ONE change
  executor  -- runs eval.py subprocess, captures SCORE: <float>
  evaluator -- keep if score improved, revert if not
  committer -- git commit (keep) or git checkout skill.md (revert)
              |
              +--> repeat until target_score or max_experiments
              |
  reporter  -- writes results.md
```

## Quick Start

```bash
python -m venv .venv && .venv\Scripts\activate   # Windows
pip install -r requirements.txt
cp .env.example .env        # set LLM_PROVIDER and model

# Git init required for revert to work
git init && git add skill.md && git commit -m "baseline"

python run.py
```

## Console Output

```
╭─ Prompt Optimizer ──────────────────────────────────────────╮
│ Backend:  prompt_optimizer                                   │
│ Goal:     Improve landing page copy to score 95%+...        │
│ Target:   skill.md                                           │
│ LLM: OLLAMA  --  Started: 2026-03-30 09:00:00               │
╰─────────────────────────────────────────────────────────────╯
[eval_builder] eval.py written (1243 chars)
[eval_locker]  eval.py committed and locked
[baseline]     Baseline SCORE: 41.33%
[proposer]     Round 1: Add specific number requirement to headline rule
[evaluator]    Round 1 ✅ 41.33% -> 68.00% | Add specific number...
[proposer]     Round 2: Add banned buzzword list to Writing Rules
[evaluator]    Round 2 ✅ 68.00% -> 79.33% | Add banned buzzword list...
[reporter]     Results written to results.md
[run]          Finished at 2026-03-30 09:04:12 -- elapsed 0:04:12
```

## LLM Providers

Set `LLM_PROVIDER` in `.env`:

| Provider | Value | Model env var |
|---|---|---|
| Ollama (local) | `ollama` | `OLLAMA_MODEL` |
| OpenAI | `openai` | `OPENAI_MODEL` |
| Anthropic | `anthropic` | `ANTHROPIC_MODEL` |

## program.md Reference

```markdown
## backend
prompt_optimizer

## goal
<what you want to achieve>

## target_file
skill.md            # the file the agent edits

## test_inputs
- "test scenario 1"
- "test scenario 2"
- "test scenario 3"

## eval_criteria           # 3-6 binary yes/no questions
- Does the headline include a specific number?
- Is the copy free of buzzwords?
- Does the CTA use a specific action verb?

## known_constraints       # agent will never violate these
- Do not remove the Headline/Subheadline/Body/CTA structure

## constraints
outputs_per_round: 10      # outputs scored per experiment
target_score: 0.95         # stop when this is reached
max_experiments: 20
revert_on_no_improvement: true
```

## Writing Good Eval Criteria

| Do | Don't |
|---|---|
| "Does the headline include a specific number?" | "Is the headline compelling?" |
| "Is the copy under 150 words?" | "Is the copy concise?" |
| "Does the CTA name a specific outcome?" | "Is the CTA strong?" |

Rules: binary only (pass/fail). 3-6 criteria. Lock the eval — never edit it mid-run.

## Example Use Cases

All use the same backend. Swap `skill.md` and `eval_criteria`.

| Use Case | Target file | Example criteria |
|---|---|---|
| Landing page copy | `landing-page-skill.md` | headline has number, no buzzwords, CTA is verb |
| Cold email | `cold-email-skill.md` | under 75 words, references role, ends with question |
| Ad copy | `ad-skill.md` | pain point in first 5 words, under 40 chars |
| Job description | `job-post-skill.md` | under 400 words, names specific project, includes salary |
| Video script | `script-skill.md` | curiosity gap in opener, single narrative thread |
| Agent system prompt | `system-prompt.md` | identifies intent, includes next action, handles edge cases |

## Folder Structure

```
prompt-optimizer/
├── program.md              -- ONLY file you edit to control the agent
├── skill.md                -- the skill/prompt file the agent improves
├── eval.py                 -- generated on first run, then git-locked
├── run.py
├── program_parser.py
├── requirements.txt
├── .env.example
├── .gitignore
├── backends/
│   ├── llm.py              -- shared LLM factory
│   ├── log.py              -- shared Rich console
│   └── prompt_optimizer/
│       ├── graph.py
│       ├── models.py
│       └── nodes.py
└── tests/
    ├── test_parser.py
    ├── test_nodes.py
    └── test_graph.py
```

## Tests

```bash
python -m pytest tests/ -v
# 17 tests
```
