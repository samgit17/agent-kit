from __future__ import annotations
import re
import subprocess
from pathlib import Path

from langgraph.graph import StateGraph, END

from .models import PromptOptimizerState
from .nodes import (
    eval_builder_node,
    eval_locker_node,
    baseline_node,
    proposer_node,
    executor_node,
    evaluator_node,
    committer_node,
    reporter_node,
    should_build_eval,
    should_continue,
)
from program_parser import ProgramConfig

# Parses score from commit messages: "round 3: ... (score=0.8500)"
_COMMIT_SCORE_RE = re.compile(r"score=([0-9]+\.[0-9]+)")


def _read_best_score_from_git(target_file: str) -> float:
    """Parse best score from git commit history of the target skill file."""
    skill = Path(__file__).parent.parent.parent / target_file
    result = subprocess.run(
        ["git", "log", "--oneline", skill.name],
        cwd=str(skill.parent),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return 0.0

    best = 0.0
    for line in result.stdout.splitlines():
        match = _COMMIT_SCORE_RE.search(line)
        if match:
            val = float(match.group(1))
            if val > best:
                best = val
    return best


def _eval_exists() -> bool:
    return (Path(__file__).parent.parent.parent / "eval.py").exists()


def build_graph(cfg: ProgramConfig):
    g = StateGraph(PromptOptimizerState)

    g.add_node("eval_builder", eval_builder_node)
    g.add_node("eval_locker", eval_locker_node)
    g.add_node("baseline", baseline_node)
    g.add_node("proposer", proposer_node)
    g.add_node("executor", executor_node)
    g.add_node("evaluator", evaluator_node)
    g.add_node("committer", committer_node)
    g.add_node("reporter", reporter_node)

    # Phase 1: eval_builder is a no-op if eval_built=True, routes to skip
    g.set_entry_point("eval_builder")
    g.add_conditional_edges(
        "eval_builder",
        should_build_eval,
        {"build": "eval_locker", "skip": "baseline"},
    )
    g.add_edge("eval_locker", "baseline")

    # Phase 2: ratchet loop
    g.add_edge("baseline", "proposer")
    g.add_edge("proposer", "executor")
    g.add_edge("executor", "evaluator")
    g.add_edge("evaluator", "committer")
    g.add_conditional_edges(
        "committer",
        should_continue,
        {"propose": "proposer", "report": "reporter"},
    )
    g.add_edge("reporter", END)

    return g.compile()


def build_initial_state(cfg: ProgramConfig) -> dict:
    best = _read_best_score_from_git(cfg.po.target_file)
    return {
        "goal": cfg.goal,
        "target_file": cfg.po.target_file,
        "test_inputs": cfg.po.test_inputs,
        "eval_criteria": cfg.po.eval_criteria,
        "known_constraints": cfg.po.known_constraints,
        "outputs_per_round": cfg.po.outputs_per_round,
        "target_score": cfg.po.target_score,
        "max_experiments": cfg.po.max_experiments,
        "revert_on_no_improvement": cfg.po.revert_on_no_improvement,
        "eval_built": _eval_exists(),
        "best_score": best,
    }
