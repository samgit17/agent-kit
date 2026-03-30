"""
tests/test_ml_graph.py — Tests for ML experiment graph startup behaviour.

Run: python -m pytest tests/test_ml_graph.py -v
"""

from unittest.mock import patch


def test_read_best_val_bpb_from_git_parses_commits():
    from backends.ml_experiment.graph import _read_best_val_bpb_from_git

    fake_log = (
        "f271dca exp 3: weight_decay (val_bpb=2.2346)\n"
        "b38fe32 exp 5: N_HEADS 8->12 (val_bpb=2.1780)\n"
        "08578cd exp 3: D_MODEL 512->768 (val_bpb=2.2035)\n"
        "0b18be3 exp 1: warmup_cosine (val_bpb=2.2186)\n"
        "b1af392 baseline\n"
    )
    mock_result = type("R", (), {"returncode": 0, "stdout": fake_log})()

    with patch("subprocess.run", return_value=mock_result):
        best = _read_best_val_bpb_from_git()

    assert best == 2.1780


def test_read_best_val_bpb_returns_inf_on_no_history():
    from backends.ml_experiment.graph import _read_best_val_bpb_from_git

    mock_result = type("R", (), {"returncode": 0, "stdout": "b1af392 baseline\n"})()

    with patch("subprocess.run", return_value=mock_result):
        best = _read_best_val_bpb_from_git()

    assert best == float("inf")


def test_read_best_val_bpb_returns_inf_on_git_failure():
    from backends.ml_experiment.graph import _read_best_val_bpb_from_git

    mock_result = type("R", (), {"returncode": 1, "stdout": ""})()

    with patch("subprocess.run", return_value=mock_result):
        best = _read_best_val_bpb_from_git()

    assert best == float("inf")


def test_build_initial_state_includes_best_val_bpb():
    from backends.ml_experiment.graph import build_initial_state, _read_best_val_bpb_from_git
    from program_parser import ProgramConfig

    cfg = ProgramConfig(backend="ml_experiment", goal="test")

    with patch("backends.ml_experiment.graph._read_best_val_bpb_from_git", return_value=2.1780):
        state = build_initial_state(cfg)

    assert state["best_val_bpb"] == 2.1780
