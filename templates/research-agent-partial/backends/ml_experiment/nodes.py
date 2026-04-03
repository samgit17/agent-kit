"""
backends/ml_experiment/nodes.py — The ratchet loop.

proposer   → LLM reads train.py + history → proposes a diff to train.py
executor   → runs train.py subprocess, captures VAL_BPB, enforces timeout
evaluator  → compare val_bpb to best; decide keep or revert
committer  → git commit (keep) or git checkout train.py (revert)
reporter   → writes results.md
"""

from __future__ import annotations
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from langchain_core.messages import HumanMessage
from .models import MLResearchState, ExperimentRecord
from backends.llm import get_llm, strip_json_fence
from backends.log import log

TRAIN_PY = Path(__file__).parent / "train.py"
VAL_BPB_RE = re.compile(r"VAL_BPB:\s*([0-9]+\.[0-9]+)")


def proposer_node(state: MLResearchState) -> dict:
    train_src = TRAIN_PY.read_text(encoding="utf-8")
    directions_text = "\n".join(f"- {d}" for d in state.directions)

    history_text = ""
    if state.history:
        rows = []
        for r in state.history[-10:]:  # last 10 to stay within context
            status = f"val_bpb={r.val_bpb:.4f} ({'kept' if r.kept else 'reverted'})" if r.val_bpb else f"FAILED: {r.error}"
            rows.append(f"Exp {r.experiment_num}: {r.proposal} → {status}")
        history_text = "\n".join(rows)

    prompt = f"""You are an ML research agent optimizing a language model training script.

GOAL: {state.goal}
BEST VAL_BPB SO FAR: {state.best_val_bpb if state.best_val_bpb != float('inf') else 'not yet measured'}
EXPERIMENT: {state.current_experiment + 1} of {state.max_experiments}
VRAM BUDGET: {state.vram_budget_gb}GB

RESEARCH DIRECTIONS TO EXPLORE:
{directions_text}

EXPERIMENT HISTORY:
{history_text or 'No experiments yet.'}

CURRENT train.py:
```python
{train_src}
```

Propose ONE specific code change to train.py that is likely to improve val_bpb.
- Do not change anything in the evaluation logic or VAL_BPB print statement.
- Stay within the VRAM budget — do not increase DEPTH or batch_size beyond what fits in {state.vram_budget_gb}GB.
- Do not repeat a change that was already tried and reverted.

Respond with ONLY a JSON object:
{{
  "description": "one-line human-readable description of the change",
  "old_code": "exact string to replace (must exist verbatim in train.py)",
  "new_code": "replacement string"
}}"""

    response = get_llm(temperature=0.3).invoke([HumanMessage(content=prompt)])
    proposal = json.loads(strip_json_fence(response.content))

    # Apply the patch to train.py
    src = TRAIN_PY.read_text(encoding="utf-8")
    if proposal["old_code"] not in src:
        log("[proposer]", f"Patch failed: old_code not found in train.py", style="red")
        return {
            "current_proposal": proposal["description"],
            "last_error": f"Patch failed: old_code not found in train.py: {proposal['old_code']!r}",
            "last_val_bpb": None,
        }

    patched = src.replace(proposal["old_code"], proposal["new_code"], 1)
    TRAIN_PY.write_text(patched, encoding="utf-8")
    log("[proposer]", f"Exp {state.current_experiment + 1}: {proposal['description']}")
    return {"current_proposal": proposal["description"], "last_error": None}


def executor_node(state: MLResearchState) -> dict:
    if state.last_error:
        # Patch failed in proposer — skip execution
        return {}

    timeout_secs = state.minutes_per_experiment * 60
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(state.gpu)}

    try:
        result = subprocess.run(
            [sys.executable, str(TRAIN_PY)],
            capture_output=True,
            text=True,
            timeout=timeout_secs,
            env=env,
            cwd=str(TRAIN_PY.parent),
        )
        output = result.stdout + result.stderr
        match = VAL_BPB_RE.search(output)
        if match:
            return {"last_val_bpb": float(match.group(1)), "last_error": None}
        else:
            return {
                "last_val_bpb": None,
                "last_error": f"No VAL_BPB in output. stderr: {result.stderr[-500:]}",
            }
    except subprocess.TimeoutExpired:
        log("[executor]", f"Timed out after {state.minutes_per_experiment}min", style="red")
        return {"last_val_bpb": None, "last_error": f"Training timed out after {state.minutes_per_experiment}min"}
    except Exception as e:
        log("[executor]", str(e), style="red")
        return {"last_val_bpb": None, "last_error": str(e)}


def evaluator_node(state: MLResearchState) -> dict:
    exp_num = state.current_experiment + 1
    val_bpb = state.last_val_bpb
    kept = False
    new_best = state.best_val_bpb

    if val_bpb is not None and val_bpb < state.best_val_bpb:
        kept = True
        new_best = val_bpb
        log("[evaluator]", f"Exp {exp_num} ✅ {state.best_val_bpb:.4f} → {val_bpb:.4f} | {state.current_proposal}", style="green")
    elif val_bpb is not None:
        log("[evaluator]", f"Exp {exp_num} ❌ {val_bpb:.4f} (best={state.best_val_bpb:.4f}) | {state.current_proposal}", style="yellow")
    else:
        log("[evaluator]", f"Exp {exp_num} 💥 {state.last_error}", style="red")

    record = ExperimentRecord(
        experiment_num=exp_num,
        proposal=state.current_proposal,
        val_bpb=val_bpb,
        kept=kept,
        error=state.last_error,
    )
    return {
        "history": state.history + [record],
        "best_val_bpb": new_best,
        "current_experiment": exp_num,
    }


def committer_node(state: MLResearchState) -> dict:
    last = state.history[-1]

    if last.kept:
        msg = f"exp {last.experiment_num}: {last.proposal} (val_bpb={last.val_bpb:.4f})"
        add = subprocess.run(
            ["git", "add", str(TRAIN_PY)],
            cwd=str(TRAIN_PY.parent),
            capture_output=True,
        )
        if add.returncode != 0:
            log("[committer]", f"git add failed: {add.stderr.decode()}", style="red")
            return {}
        commit = subprocess.run(
            ["git", "commit", "-m", msg],
            cwd=str(TRAIN_PY.parent),
            capture_output=True,
        )
        if commit.returncode != 0:
            log("[committer]", f"git commit failed: {commit.stderr.decode()}", style="red")
    elif state.revert_on_no_improvement:
        revert = subprocess.run(
            ["git", "checkout", "--", str(TRAIN_PY.name)],
            cwd=str(TRAIN_PY.parent),
            capture_output=True,
        )
        if revert.returncode != 0:
            log("[committer]", f"git checkout failed: {revert.stderr.decode()}", style="red")
    return {}


def should_continue(state: MLResearchState) -> str:
    if state.current_experiment >= state.max_experiments:
        return "report"
    return "propose"


def reporter_node(state: MLResearchState) -> dict:
    successful = [r for r in state.history if r.kept]
    failed = [r for r in state.history if not r.kept]
    total = len(state.history)

    lines = [
        f"# ML Experiment Report",
        f"",
        f"**Goal:** {state.goal}",
        f"**GPU:** CUDA:{state.gpu} | **Budget:** {state.minutes_per_experiment}min/exp",
        f"**Total experiments:** {total} | **Kept:** {len(successful)} | **Reverted/Failed:** {len(failed)}",
        f"**Best val_bpb:** {state.best_val_bpb:.4f}" if state.best_val_bpb != float("inf") else "**Best val_bpb:** N/A",
        f"",
        f"## Kept Changes",
    ]
    for r in successful:
        lines.append(f"- Exp {r.experiment_num}: `{r.proposal}` → val_bpb={r.val_bpb:.4f}")

    lines += ["", "## Reverted / Failed"]
    for r in failed:
        status = f"val_bpb={r.val_bpb:.4f}" if r.val_bpb else f"ERROR: {r.error}"
        lines.append(f"- Exp {r.experiment_num}: `{r.proposal}` → {status}")

    lines += ["", f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*"]
    report = "\n".join(lines)

    results_path = TRAIN_PY.parent / "results.md"
    results_path.write_text(report, encoding="utf-8")
    log("[reporter]", f"Results written to {results_path}")
    return {"report": report}
