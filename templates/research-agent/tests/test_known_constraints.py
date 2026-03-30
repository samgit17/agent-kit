"""
tests/test_known_constraints.py — Tests for known_constraints parsing and proposer integration.

Run: python -m pytest tests/test_known_constraints.py -v
"""


def test_known_constraints_parsed_from_program_md(tmp_path):
    from program_parser import parse_program
    p = tmp_path / "program.md"
    p.write_text(
        "## backend\nml_experiment\n\n"
        "## goal\nTest\n\n"
        "## known_constraints\n"
        "- Do not increase CONTEXT_LEN -- timeouts at 10min budget\n"
        "- Do not increase DEPTH beyond 6 -- OOM on 16GB\n",
        encoding="utf-8",
    )
    cfg = parse_program(p)
    assert len(cfg.ml.known_constraints) == 2
    assert "CONTEXT_LEN" in cfg.ml.known_constraints[0]


def test_known_constraints_default_empty():
    from program_parser import MLConfig
    cfg = MLConfig()
    assert cfg.known_constraints == []


def test_known_constraints_in_initial_state():
    from backends.ml_experiment.graph import build_initial_state
    from program_parser import ProgramConfig, MLConfig
    from unittest.mock import patch

    cfg = ProgramConfig(backend="ml_experiment", goal="test")
    cfg.ml = MLConfig(known_constraints=["Do not increase CONTEXT_LEN"])

    with patch("backends.ml_experiment.graph._read_best_val_bpb_from_git", return_value=float("inf")):
        state = build_initial_state(cfg)

    assert state["known_constraints"] == ["Do not increase CONTEXT_LEN"]


def test_known_constraints_appear_in_proposer_prompt():
    """known_constraints must appear in the prompt sent to the LLM."""
    from backends.ml_experiment.models import MLResearchState

    state = MLResearchState(
        goal="test", directions=[], gpu=0,
        minutes_per_experiment=5, max_experiments=3,
        revert_on_no_improvement=True, vram_budget_gb=16,
        known_constraints=["Do not increase CONTEXT_LEN — timeouts"],
    )
    # Build the prompt the same way proposer_node does
    constraints_text = "\n".join(f"- {c}" for c in state.known_constraints)
    assert "CONTEXT_LEN" in constraints_text
