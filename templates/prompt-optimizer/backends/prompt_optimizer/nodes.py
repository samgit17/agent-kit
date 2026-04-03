"""
backends/prompt_optimizer/nodes.py — Ratchet loop for prompt/skill optimization.

Nodes:
  eval_builder  → generates eval.py from static template (no LLM), then git-locks it
  baseline      → runs eval.py once to establish best_score
  proposer      → LLM reads skill.md + history → proposes ONE change
  executor      → runs eval.py subprocess, captures SCORE
  evaluator     → keep or revert
  committer     → git commit (keep) or git checkout (revert)
  reporter      → writes results.md
"""

from __future__ import annotations
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from langchain_core.messages import HumanMessage

from .models import PromptOptimizerState, RoundRecord
from backends.llm import get_llm, strip_json_fence
from backends.log import log

SCORE_RE = re.compile(r"SCORE:\s*([0-9]+\.[0-9]+)")


def _skill_path(state: PromptOptimizerState) -> Path:
    return Path(__file__).parent.parent.parent / state.target_file


def _eval_path() -> Path:
    return Path(__file__).parent.parent.parent / "eval.py"


def _run_eval() -> float | None:
    """Run eval.py subprocess and return parsed SCORE, or None on failure."""
    result = subprocess.run(
        [sys.executable, str(_eval_path())],
        capture_output=True, text=True, cwd=str(_eval_path().parent),
    )
    output = result.stdout + result.stderr
    match = SCORE_RE.search(output)
    if match:
        return float(match.group(1))
    log("[eval]", f"No SCORE in output. stderr: {result.stderr[-300:]}", style="red")
    return None


# ── Phase 1: Eval setup ───────────────────────────────────────────────────────

def should_build_eval(state: PromptOptimizerState) -> str:
    """Routes to build only if eval.py doesn't exist yet."""
    return "build" if not state.eval_built else "skip"


def eval_builder_node(state: PromptOptimizerState) -> dict:
    """Generates eval.py from a static template by injecting criteria as data.
    No LLM call — instant and deterministic."""
    if state.eval_built:
        return {}  # skip — eval.py already locked in git

    from .eval_template import EVAL_TEMPLATE

    script = EVAL_TEMPLATE.format(
        skill_file=state.target_file,
        test_inputs=state.test_inputs,
        eval_criteria=state.eval_criteria,
        outputs_per_round=state.outputs_per_round,
    )

    _eval_path().write_text(script, encoding="utf-8")
    log("[eval_builder]", f"eval.py generated ({len(script)} chars)")
    return {"eval_built": True}


def eval_locker_node(state: PromptOptimizerState) -> dict:
    """Git commit eval.py so it can never be accidentally modified."""
    eval_py = _eval_path()
    cwd = str(eval_py.parent)
    subprocess.run(["git", "add", eval_py.name], cwd=cwd, capture_output=True)
    result = subprocess.run(
        ["git", "commit", "-m", "lock: eval harness -- do not modify"],
        cwd=cwd, capture_output=True,
    )
    if result.returncode == 0:
        log("[eval_locker]", "eval.py committed and locked")
    else:
        log("[eval_locker]", "eval.py already committed", style="dim")
    return {}


def baseline_node(state: PromptOptimizerState) -> dict:
    """Run eval.py once to get baseline score."""
    log("[baseline]", "Running baseline evaluation...")
    score = _run_eval()
    if score is not None:
        log("[baseline]", f"Baseline SCORE: {score:.2%}")
        return {"best_score": score}
    log("[baseline]", "Could not parse baseline score — defaulting to 0.0", style="red")
    return {"best_score": 0.0}


# ── Phase 2: Ratchet loop ─────────────────────────────────────────────────────

def proposer_node(state: PromptOptimizerState) -> dict:
    skill_src = _skill_path(state).read_text(encoding="utf-8")

    # Build history summary
    history_text = ""
    if state.history:
        rows = []
        for r in state.history[-10:]:
            if r.score is not None:
                status = f"score={r.score:.2%} ({'KEPT' if r.kept else 'REVERTED'})"
            else:
                status = f"FAILED: {r.error}"
            rows.append(f"Round {r.round_num}: {r.proposal} -> {status}")
        history_text = "\n".join(rows)

    # Explicit list of reverted patches — deduplicated by old_text for semantic blocking
    reverted = [(r.proposal, r.old_text) for r in state.history if not r.kept]
    seen_old_texts: set[str] = set()
    reverted_unique = []
    for proposal, old_text in reverted:
        if old_text not in seen_old_texts:
            seen_old_texts.add(old_text)
            reverted_unique.append((proposal, old_text))

    if reverted_unique:
        do_not_retry = "\n".join(
            f"- {proposal} [patched: {old_text!r}]"
            for proposal, old_text in reverted_unique
        )
    else:
        do_not_retry = "None yet."

    constraints_text = ""
    if state.known_constraints:
        constraints_text = "\nKNOWN CONSTRAINTS -- never violate these:\n" + \
                           "\n".join(f"- {c}" for c in state.known_constraints)

    criteria_text = "\n".join(f"- {c}" for c in state.eval_criteria)

    prompt = f"""You are optimizing a skill/prompt file to maximize its eval score.

GOAL: {state.goal}
CURRENT BEST SCORE: {state.best_score:.2%}
TARGET SCORE: {state.target_score:.2%}
ROUND: {state.current_round + 1} of {state.max_experiments}
{constraints_text}

EVAL CRITERIA (binary yes/no per output):
{criteria_text}

EXPERIMENT HISTORY (last 10 rounds):
{history_text or 'No experiments yet.'}

ALREADY TRIED AND FAILED -- DO NOT PROPOSE THESE AGAIN:
{do_not_retry}

CURRENT {state.target_file}:
{skill_src}

Propose ONE specific text change to {state.target_file} that will improve the eval score.
Rules:
- ONE targeted change only
- MUST NOT repeat any proposal from the "ALREADY TRIED AND FAILED" list above
- MUST target a different criterion or a different part of the file than failed rounds
- Do not remove required sections or change the file structure

Respond with ONLY a JSON object:
{{
  "description": "one-line description of the change",
  "old_text": "exact string to replace (must exist verbatim in the file)",
  "new_text": "replacement string"
}}"""

    response = get_llm(temperature=0.3).invoke([HumanMessage(content=prompt)])
    stripped = strip_json_fence(response.content)
    try:
        proposal = json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        log("[proposer]", f"LLM returned invalid JSON -- skipping round. Raw: {stripped[:100]!r}", style="red")
        return {
            "current_proposal": "invalid JSON from LLM",
            "current_old_text": "",
            "proposer_failed": True,
            "last_error": "LLM returned invalid JSON",
            "last_score": None,
        }

    if proposal["old_text"] not in skill_src:
        log("[proposer]", "Patch failed: old_text not found in skill file", style="red")
        return {
            "current_proposal": proposal["description"],
            "current_old_text": proposal["old_text"],
            "proposer_failed": True,
            "last_error": "Patch failed: old_text not found",
            "last_score": None,
        }

    patched = skill_src.replace(proposal["old_text"], proposal["new_text"], 1)
    _skill_path(state).write_text(patched, encoding="utf-8")
    log("[proposer]", f"Round {state.current_round + 1}: {proposal['description']}")
    return {
        "current_proposal": proposal["description"],
        "current_old_text": proposal["old_text"],
        "proposer_failed": False,
        "last_error": None,
    }


def executor_node(state: PromptOptimizerState) -> dict:
    if state.proposer_failed:
        return {}  # proposer didn't write to disk — nothing to evaluate

    score = _run_eval()
    if score is not None:
        return {"last_score": score, "last_error": None}
    return {
        "last_score": None,
        "last_error": "eval.py produced no SCORE output",
    }


def evaluator_node(state: PromptOptimizerState) -> dict:
    round_num = state.current_round + 1

    # Proposer failed before writing — don't pollute history, just advance round
    if state.proposer_failed:
        log("[evaluator]", f"Round {round_num} ⚠️ proposer failed — skipping", style="yellow")
        return {"current_round": round_num}

    score = state.last_score
    kept = False
    new_best = state.best_score

    if score is not None and score > state.best_score:
        kept = True
        new_best = score
        log("[evaluator]", f"Round {round_num} ✅ {state.best_score:.2%} -> {score:.2%} | {state.current_proposal}", style="green")
    elif score is not None:
        log("[evaluator]", f"Round {round_num} ❌ {score:.2%} (best={state.best_score:.2%}) | {state.current_proposal}", style="yellow")
    else:
        log("[evaluator]", f"Round {round_num} 💥 {state.last_error}", style="red")

    record = RoundRecord(
        round_num=round_num,
        proposal=state.current_proposal,
        old_text=state.current_old_text,
        score=score,
        kept=kept,
        error=state.last_error,
    )
    return {
        "history": state.history + [record],
        "best_score": new_best,
        "current_round": round_num,
    }


def committer_node(state: PromptOptimizerState) -> dict:
    if state.proposer_failed:
        return {}  # nothing was written to disk

    last = state.history[-1]
    skill = _skill_path(state)
    cwd = str(skill.parent)

    if last.kept:
        msg = f"round {last.round_num}: {last.proposal} (score={last.score:.4f})"
        add = subprocess.run(["git", "add", skill.name], cwd=cwd, capture_output=True)
        if add.returncode != 0:
            log("[committer]", f"git add failed: {add.stderr.decode()}", style="red")
            return {}
        commit = subprocess.run(["git", "commit", "-m", msg], cwd=cwd, capture_output=True)
        if commit.returncode != 0:
            log("[committer]", f"git commit failed: {commit.stderr.decode()}", style="red")
    elif state.revert_on_no_improvement:
        revert = subprocess.run(
            ["git", "checkout", "--", skill.name], cwd=cwd, capture_output=True
        )
        if revert.returncode != 0:
            log("[committer]", f"git checkout failed: {revert.stderr.decode()}", style="red")
    return {}


def should_continue(state: PromptOptimizerState) -> str:
    if state.current_round >= state.max_experiments:
        return "report"
    if state.best_score >= state.target_score:
        return "report"
    return "propose"


def reporter_node(state: PromptOptimizerState) -> dict:
    kept = [r for r in state.history if r.kept]
    failed = [r for r in state.history if not r.kept]

    lines = [
        "# Prompt Optimizer Report",
        "",
        f"**Goal:** {state.goal}",
        f"**Target file:** `{state.target_file}`",
        f"**Total rounds:** {len(state.history)} | **Kept:** {len(kept)} | **Reverted/Failed:** {len(failed)}",
        f"**Best score:** {state.best_score:.2%} (target: {state.target_score:.2%})",
        "",
        "## Kept Changes",
    ]
    for r in kept:
        lines.append(f"- Round {r.round_num}: `{r.proposal}` -> score={r.score:.2%}")

    lines += ["", "## Reverted / Failed"]
    for r in failed:
        status = f"score={r.score:.2%}" if r.score is not None else f"ERROR: {r.error}"
        lines.append(f"- Round {r.round_num}: `{r.proposal}` -> {status}")

    lines += ["", f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*"]
    report = "\n".join(lines)

    results_path = _eval_path().parent / "results.md"
    results_path.write_text(report, encoding="utf-8")
    log("[reporter]", f"Results written to {results_path}")
    return {"report": report}
