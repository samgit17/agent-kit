"""
program_parser.py — Parses program.md into a typed ProgramConfig.

Sections are delimited by ## headings. Lines starting with # are comments.
Everything after --- is ignored (used for inline documentation in program.md).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class WebConfig:
    max_iterations: int = 3
    confidence_threshold: float = 0.6


@dataclass
class MLConfig:
    gpu: int = 0
    minutes_per_experiment: int = 10
    max_experiments: int = 20
    revert_on_no_improvement: bool = True
    vram_budget_gb: int = 12


@dataclass
class ProgramConfig:
    backend: Literal["web", "ml_experiment"] = "web"
    goal: str = ""
    success_criteria: list[str] = field(default_factory=list)
    directions: list[str] = field(default_factory=list)
    web: WebConfig = field(default_factory=WebConfig)
    ml: MLConfig = field(default_factory=MLConfig)


def parse_program(path: str | Path = "program.md") -> ProgramConfig:
    text = Path(path).read_text(encoding="utf-8")

    # Strip everything after ---
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

    if "backend" in sections:
        val = sections["backend"][0].strip()
        if val in ("web", "ml_experiment"):
            cfg.backend = val  # type: ignore[assignment]

    if "goal" in sections:
        cfg.goal = " ".join(sections["goal"])

    if "success_criteria" in sections:
        cfg.success_criteria = [
            l.lstrip("- ").strip() for l in sections["success_criteria"] if l.strip()
        ]

    if "directions" in sections:
        cfg.directions = [
            l.lstrip("- ").strip() for l in sections["directions"] if l.strip()
        ]

    if "constraints" in sections:
        for line in sections["constraints"]:
            if ":" not in line:
                continue
            key, _, val_raw = line.partition(":")
            key = key.strip().lower()
            val = val_raw.strip()
            # Web constraints
            if key == "max_iterations":
                cfg.web.max_iterations = int(val)
            elif key == "confidence_threshold":
                cfg.web.confidence_threshold = float(val)
            # ML constraints
            elif key == "gpu":
                cfg.ml.gpu = int(val)
            elif key == "minutes_per_experiment":
                cfg.ml.minutes_per_experiment = int(val)
            elif key == "max_experiments":
                cfg.ml.max_experiments = int(val)
            elif key == "revert_on_no_improvement":
                cfg.ml.revert_on_no_improvement = val.lower() == "true"
            elif key == "vram_budget_gb":
                cfg.ml.vram_budget_gb = int(val)

    return cfg
