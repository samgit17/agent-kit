"""tests/test_graph.py — graph startup and initial state."""

from unittest.mock import patch


def test_initial_state_from_config():
    from backends.prompt_optimizer.graph import build_initial_state
    from program_parser import ProgramConfig, POConfig

    cfg = ProgramConfig(backend="prompt_optimizer", goal="Improve copy")
    cfg.po = POConfig(
        target_file="skill.md",
        eval_criteria=["Does headline have number?"],
        test_inputs=["An AI tool"],
        outputs_per_round=5,
        target_score=0.9,
        max_experiments=5,
    )

    with patch("backends.prompt_optimizer.graph._read_best_score_from_git", return_value=0.0), \
         patch("backends.prompt_optimizer.graph._eval_exists", return_value=False):
        state = build_initial_state(cfg)

    assert state["target_file"] == "skill.md"
    assert state["outputs_per_round"] == 5
    assert state["target_score"] == 0.9
    assert state["best_score"] == 0.0


def test_read_best_score_returns_zero_on_no_history():
    from backends.prompt_optimizer.graph import _read_best_score_from_git

    mock_result = type("R", (), {"returncode": 0, "stdout": "abc1234 baseline\n"})()
    with patch("subprocess.run", return_value=mock_result):
        score = _read_best_score_from_git("skill.md")
    assert score == 0.0


def test_read_best_score_parses_from_commits():
    from backends.prompt_optimizer.graph import _read_best_score_from_git

    fake_log = (
        "abc1234 round 3: add number to headline (score=0.85)\n"
        "def5678 round 1: force pain point (score=0.68)\n"
        "baseline\n"
    )
    mock_result = type("R", (), {"returncode": 0, "stdout": fake_log})()
    with patch("subprocess.run", return_value=mock_result):
        score = _read_best_score_from_git("skill.md")
    assert score == 0.85


def test_resume_uses_git_history_not_zero():
    from backends.prompt_optimizer.graph import build_initial_state
    from program_parser import ProgramConfig

    cfg = ProgramConfig(backend="prompt_optimizer", goal="test")

    with patch("backends.prompt_optimizer.graph._read_best_score_from_git", return_value=0.75), \
         patch("backends.prompt_optimizer.graph._eval_exists", return_value=True):
        state = build_initial_state(cfg)

    assert state["best_score"] == 0.75
    assert state["eval_built"] is True
