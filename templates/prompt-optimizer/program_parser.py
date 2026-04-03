"""
program_parser.py — Parses program.md into typed config.

Supports backends: web | ml_experiment | prompt_optimizer
"""

from __future__ import annotations
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class POConfig:
    target_file: str = "skill.md"
    eval_criteria: list = field(default_factory=list)
    test_inputs: list = field(default_factory=list)
    known_constraints: list = field(default_factory=list)
    outputs_per_round: int = 5
    target_score: float = 0.95
    max_experiments: int = 20
    revert_on_no_improvement: bool = True


@dataclass
class ProgramConfig:
    backend: Literal["prompt_optimizer"] = "prompt_optimizer"
    goal: str = ""
    po: POConfig = field(default_factory=POConfig)


def parse_program(path: str | Path = "program.md") -> ProgramConfig:
    text = Path(path).read_text(encoding="utf-8")
    text = text.split("---")[0]

    sections: dict[str, list[str]] = {}
    current: str | None = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or (line.startswith("#") and not line.startswith("##")):
            continue
        if line.startswith("## "):
            current = line[3:].strip().lower().replace(" ", "_")
            sections[current] = []
        elif current is not None and not line.startswith("#"):
            sections[current].append(line)

    cfg = ProgramConfig()

    if "backend" in sections and sections["backend"]:
        val = sections["backend"][0].strip()
        if val == "prompt_optimizer":
            cfg.backend = val  # type: ignore[assignment]

    if "goal" in sections:
        cfg.goal = " ".join(sections["goal"])

    if "known_constraints" in sections:
        cfg.po.known_constraints = [l.lstrip("- ").strip() for l in sections["known_constraints"] if l.strip()]

    if "target_file" in sections and sections["target_file"]:
        cfg.po.target_file = sections["target_file"][0].strip()

    if "eval_criteria" in sections:
        cfg.po.eval_criteria = [l.lstrip("- ").strip() for l in sections["eval_criteria"] if l.strip()]

    if "test_inputs" in sections:
        cfg.po.test_inputs = [l.lstrip("- ").strip() for l in sections["test_inputs"] if l.strip()]

    if "constraints" in sections:
        for line in sections["constraints"]:
            if ":" not in line:
                continue
            key, _, val_raw = line.partition(":")
            key = key.strip().lower()
            val = val_raw.strip()
            if key == "outputs_per_round":
                val_int = int(val)
                if val_int < 3:
                    warnings.warn(
                        f"outputs_per_round={val_int} is too low for reliable scoring — "
                        "scores will be noisy. Minimum recommended: 3.",
                        UserWarning, stacklevel=2,
                    )
                cfg.po.outputs_per_round = val_int
            elif key == "target_score":
                cfg.po.target_score = float(val)
            elif key == "max_experiments":
                cfg.po.max_experiments = int(val)
            elif key == "revert_on_no_improvement":
                cfg.po.revert_on_no_improvement = val.lower() == "true"

    return cfg
