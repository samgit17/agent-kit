"""tests/test_parser.py — program.md parsing for prompt-optimizer."""

import tempfile


def _write_md(content: str) -> str:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
        f.write(content)
        return f.name


def test_backend_parsed():
    from program_parser import parse_program
    cfg = parse_program(_write_md("## backend\nprompt_optimizer\n\n## goal\nImprove copy.\n"))
    assert cfg.backend == "prompt_optimizer"


def test_target_file_parsed():
    from program_parser import parse_program
    cfg = parse_program(_write_md("## backend\nprompt_optimizer\n\n## goal\ntest\n\n## target_file\nskill.md\n"))
    assert cfg.po.target_file == "skill.md"


def test_eval_criteria_parsed():
    from program_parser import parse_program
    cfg = parse_program(_write_md(
        "## backend\nprompt_optimizer\n\n## goal\ntest\n\n"
        "## eval_criteria\n"
        "- Does the headline include a number?\n"
        "- Is the copy free of buzzwords?\n"
    ))
    assert len(cfg.po.eval_criteria) == 2
    assert "number" in cfg.po.eval_criteria[0]


def test_test_inputs_parsed():
    from program_parser import parse_program
    cfg = parse_program(_write_md(
        "## backend\nprompt_optimizer\n\n## goal\ntest\n\n"
        "## test_inputs\n"
        "- An AI productivity tool\n"
        "- A B2B SaaS CRM\n"
    ))
    assert len(cfg.po.test_inputs) == 2


def test_constraints_parsed():
    from program_parser import parse_program
    cfg = parse_program(_write_md(
        "## backend\nprompt_optimizer\n\n## goal\ntest\n\n"
        "## constraints\n"
        "outputs_per_round: 5\ntarget_score: 0.90\nmax_experiments: 15\n"
    ))
    assert cfg.po.outputs_per_round == 5
    assert cfg.po.target_score == 0.90
    assert cfg.po.max_experiments == 15


def test_po_config_defaults():
    from program_parser import POConfig
    cfg = POConfig()
    assert cfg.outputs_per_round == 5   # stable signal near target score
    assert cfg.target_score == 0.95
    assert cfg.max_experiments == 20
    assert cfg.target_file == "skill.md"
    assert cfg.eval_criteria == []
    assert cfg.test_inputs == []
