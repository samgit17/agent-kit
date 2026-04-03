"""tests/test_nodes.py — prompt_optimizer node logic."""

import tempfile
import warnings
from unittest.mock import patch
from backends.prompt_optimizer.models import PromptOptimizerState, RoundRecord


def _base_state(**kwargs) -> PromptOptimizerState:
    defaults = dict(
        goal="Improve copy",
        target_file="skill.md",
        test_inputs=["An AI tool"],
        eval_criteria=["Does headline have a number?"],
        known_constraints=[],
        outputs_per_round=5,
        target_score=0.95,
        max_experiments=5,
        revert_on_no_improvement=True,
    )
    defaults.update(kwargs)
    return PromptOptimizerState(**defaults)


def _program_md_with(content: str) -> str:
    """Write content to a temp program.md file and return its path."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".md", delete=False, encoding="utf-8"
    ) as f:
        f.write(content)
        return f.name


# ── should_continue ───────────────────────────────────────────────────────────

def test_should_continue_routes_to_report_when_max_reached():
    from backends.prompt_optimizer.nodes import should_continue
    state = _base_state(current_round=5, max_experiments=5)
    assert should_continue(state) == "report"


def test_should_continue_routes_to_report_when_target_met():
    from backends.prompt_optimizer.nodes import should_continue
    state = _base_state(best_score=0.96, target_score=0.95, current_round=2, max_experiments=10)
    assert should_continue(state) == "report"


def test_should_continue_routes_to_propose_when_running():
    from backends.prompt_optimizer.nodes import should_continue
    state = _base_state(best_score=0.7, target_score=0.95, current_round=2, max_experiments=10)
    assert should_continue(state) == "propose"


# ── eval_builder ──────────────────────────────────────────────────────────────

def test_should_build_eval_when_not_built():
    from backends.prompt_optimizer.nodes import should_build_eval
    state = _base_state(eval_built=False)
    assert should_build_eval(state) == "build"


def test_should_skip_eval_when_already_built():
    from backends.prompt_optimizer.nodes import should_build_eval
    state = _base_state(eval_built=True)
    assert should_build_eval(state) == "skip"


def test_eval_builder_is_noop_when_already_built(tmp_path):
    from backends.prompt_optimizer.nodes import eval_builder_node
    state = _base_state(eval_built=True)
    result = eval_builder_node(state)
    assert result == {}
    assert not (tmp_path / "eval.py").exists()


# ── proposer DO NOT RETRY ─────────────────────────────────────────────────────

def test_proposer_do_not_retry_excludes_reverted_and_includes_kept(tmp_path):
    """proposer_node must build DO NOT RETRY from reverted only, not kept."""
    from backends.prompt_optimizer.nodes import proposer_node

    skill = tmp_path / "skill.md"
    skill.write_text("## Headline\nBuy now.", encoding="utf-8")

    reverted = RoundRecord(round_num=1, proposal="Add number to headline",
                           old_text="Buy now.", score=0.8, kept=False)
    kept = RoundRecord(round_num=2, proposal="Remove buzzword",
                       old_text="Buy now.", score=0.93, kept=True)
    state = _base_state(
        target_file=str(skill),
        history=[reverted, kept],
        current_round=2,
    )

    captured_prompt = {}

    def fake_invoke(messages):
        captured_prompt["text"] = messages[0].content
        class R:
            content = '{"description":"test","old_text":"Buy now.","new_text":"Save 10 hours."}'
        return R()

    with patch("backends.prompt_optimizer.nodes.get_llm") as mock_llm:
        mock_llm.return_value.invoke = fake_invoke
        proposer_node(state)

    prompt = captured_prompt["text"]
    assert "Add number to headline" in prompt
    assert "Buy now." in prompt   # old_text included for semantic blocking
    assert "Remove buzzword" not in prompt.split("ALREADY TRIED")[1]


def test_proposer_do_not_retry_deduplicates_repeated_reverts(tmp_path):
    """Same old_text reverted 5 times should appear once in DO NOT RETRY."""
    from backends.prompt_optimizer.nodes import proposer_node

    skill = tmp_path / "skill.md"
    skill.write_text("## Headline\nBuy now.", encoding="utf-8")

    repeated = [
        RoundRecord(round_num=i, proposal="Add number",
                    old_text="Buy now.", score=0.8, kept=False)
        for i in range(1, 6)
    ]
    state = _base_state(target_file=str(skill), history=repeated, current_round=5)

    captured_prompt = {}

    def fake_invoke(messages):
        captured_prompt["text"] = messages[0].content
        class R:
            content = '{"description":"test","old_text":"Buy now.","new_text":"Buy 10."}'
        return R()

    with patch("backends.prompt_optimizer.nodes.get_llm") as mock_llm:
        mock_llm.return_value.invoke = fake_invoke
        proposer_node(state)

    do_not_retry_section = captured_prompt["text"].split("ALREADY TRIED AND FAILED")[1]
    assert do_not_retry_section.count("Add number") == 1


def test_round_record_stores_old_text():
    """RoundRecord must store old_text for semantic deduplication."""
    record = RoundRecord(round_num=1, proposal="Add number",
                         old_text="Buy now.", score=0.8, kept=False)
    assert record.old_text == "Buy now."


def test_eval_template_uses_eval_model_env_var():
    """eval.py template must use EVAL_MODEL, not OLLAMA_MODEL, for scoring."""
    from backends.prompt_optimizer.eval_template import EVAL_TEMPLATE
    # EVAL_MODEL should be referenced before OLLAMA_MODEL in LLM setup
    assert "EVAL_MODEL" in EVAL_TEMPLATE
    eval_model_pos = EVAL_TEMPLATE.index("EVAL_MODEL")
    ollama_model_pos = EVAL_TEMPLATE.index("OLLAMA_MODEL")
    assert eval_model_pos < ollama_model_pos


# ── evaluator ─────────────────────────────────────────────────────────────────

def test_proposer_handles_invalid_llm_response(tmp_path):
    """proposer_node must not crash when LLM returns unparseable content."""
    from backends.prompt_optimizer.nodes import proposer_node

    skill = tmp_path / "skill.md"
    skill.write_text("## Headline\nBuy now.", encoding="utf-8")
    state = _base_state(target_file=str(skill))

    for bad_content in [
        "<think>\nLet me think...\n</think>",   # think-only, no JSON
        "   ",                                   # whitespace only
        "Here is my suggestion: change X to Y", # prose, not JSON
    ]:
        def fake_invoke(messages, _c=bad_content):
            class R:
                content = _c
            return R()

        with patch("backends.prompt_optimizer.nodes.get_llm") as mock_llm:
            mock_llm.return_value.invoke = fake_invoke
            result = proposer_node(state)

        assert result["last_error"] is not None
        assert result["last_score"] is None


# ── evaluator ─────────────────────────────────────────────────────────────────

def test_evaluator_does_not_record_failed_proposer_rounds():
    """Rounds where proposer failed (no file write) must NOT appear in history."""
    from backends.prompt_optimizer.nodes import evaluator_node
    state = _base_state(
        best_score=0.8,
        last_score=None,
        last_error="LLM returned invalid JSON",
        current_proposal="invalid JSON from LLM",
        current_old_text="",
        proposer_failed=True,
        current_round=0,
    )
    result = evaluator_node(state)
    assert result.get("history", state.history) == state.history  # no new records
    assert result["current_round"] == 1  # round still advances


def test_evaluator_records_real_experiments():
    """Rounds where proposer successfully patched must appear in history."""
    from backends.prompt_optimizer.nodes import evaluator_node
    state = _base_state(
        best_score=0.8,
        last_score=0.6,
        last_error=None,
        current_proposal="Change headline",
        current_old_text="Buy now.",
        proposer_failed=False,
        current_round=0,
    )
    result = evaluator_node(state)
    assert len(result["history"]) == 1


def test_evaluator_keeps_improvement():
    from backends.prompt_optimizer.nodes import evaluator_node
    state = _base_state(best_score=0.6, last_score=0.75, current_round=0,
                        current_proposal="Added number to headline")
    result = evaluator_node(state)
    assert result["best_score"] == 0.75
    assert result["history"][-1].kept is True


def test_evaluator_reverts_regression():
    from backends.prompt_optimizer.nodes import evaluator_node
    state = _base_state(best_score=0.75, last_score=0.6, current_round=0,
                        current_proposal="Removed number")
    result = evaluator_node(state)
    assert result["best_score"] == 0.75
    assert result["history"][-1].kept is False


def test_evaluator_handles_failed_run():
    from backends.prompt_optimizer.nodes import evaluator_node
    state = _base_state(best_score=0.5, last_score=None, last_error="Timeout",
                        current_round=0, current_proposal="bad change")
    result = evaluator_node(state)
    assert result["history"][-1].kept is False
    assert result["history"][-1].error == "Timeout"


# ── outputs_per_round warning ─────────────────────────────────────────────────

def test_outputs_per_round_warning_below_3():
    from program_parser import parse_program
    p = _program_md_with("## backend\nprompt_optimizer\n\n## goal\ntest\n\n## constraints\noutputs_per_round: 1\n")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        parse_program(p)
    assert any("outputs_per_round" in str(warning.message) for warning in w)


def test_outputs_per_round_no_warning_at_3():
    from program_parser import parse_program
    p = _program_md_with("## backend\nprompt_optimizer\n\n## goal\ntest\n\n## constraints\noutputs_per_round: 3\n")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        parse_program(p)
    assert not any("outputs_per_round" in str(warning.message) for warning in w)
