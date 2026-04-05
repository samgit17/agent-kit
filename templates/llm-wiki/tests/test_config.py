import pytest


def test_validate_env_passes_when_vars_set(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "gpt-4o")
    import importlib
    import agent.config as cfg
    importlib.reload(cfg)
    cfg.validate_env()  # should not raise


def test_validate_env_raises_on_missing_vars(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    import importlib
    import agent.config as cfg
    importlib.reload(cfg)
    with pytest.raises(EnvironmentError, match="LLM_API_KEY"):
        cfg.validate_env()


def test_validate_env_reports_all_missing(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    import importlib
    import agent.config as cfg
    importlib.reload(cfg)
    with pytest.raises(EnvironmentError) as exc:
        cfg.validate_env()
    assert "LLM_MODEL" in str(exc.value)


def test_max_iterations_default(monkeypatch):
    monkeypatch.delenv("WIKI_MAX_ITERATIONS", raising=False)
    import importlib
    import agent.config as cfg
    importlib.reload(cfg)
    assert cfg.MAX_ITERATIONS == 10


def test_max_iterations_override(monkeypatch):
    monkeypatch.setenv("WIKI_MAX_ITERATIONS", "5")
    import importlib
    import agent.config as cfg
    importlib.reload(cfg)
    assert cfg.MAX_ITERATIONS == 5
