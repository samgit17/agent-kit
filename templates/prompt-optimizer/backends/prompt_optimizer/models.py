from __future__ import annotations
from pydantic import BaseModel, Field


class RoundRecord(BaseModel):
    round_num: int
    proposal: str
    old_text: str = ""     # the exact text that was patched — used for semantic deduplication
    score: float | None    # None = run failed
    kept: bool
    error: str | None = None


class PromptOptimizerState(BaseModel):
    # Config from program.md
    goal: str
    target_file: str
    test_inputs: list[str]
    eval_criteria: list[str]
    known_constraints: list[str]
    outputs_per_round: int
    target_score: float
    max_experiments: int
    revert_on_no_improvement: bool

    # Runtime state
    eval_built: bool = False
    best_score: float = 0.0
    current_round: int = 0
    history: list[RoundRecord] = Field(default_factory=list)
    current_proposal: str = ""
    current_old_text: str = ""   # old_text from last proposer patch
    proposer_failed: bool = False  # True when proposer errored before writing to disk
    last_score: float | None = None
    last_error: str | None = None
    report: str = ""
