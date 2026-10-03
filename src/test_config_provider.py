from __future__ import annotations

from pathlib import Path

from config import LabConfig, load_config
from model_provider import ProviderConfig, normalize_provider


def test_normalize_provider_maps_aliases_and_typos() -> None:
    assert normalize_provider("Anthropic") == "anthropic"
    assert normalize_provider("anthorpic") == "anthropic"
    assert normalize_provider("Custom") == "custom"
    assert normalize_provider("openai-compatible") == "custom"
    assert normalize_provider("GEMINI") == "gemini"
    assert normalize_provider("") == ""


def test_load_config_creates_state_dir_and_defaults(tmp_path: Path) -> None:
    cfg = load_config(base_dir=tmp_path)

    assert isinstance(cfg, LabConfig)
    assert cfg.base_dir == tmp_path.resolve()
    assert cfg.data_dir == tmp_path.resolve() / "data"
    assert cfg.state_dir.is_dir(), "load_config phải tạo state/ nếu chưa có"
    assert cfg.compact_threshold_tokens == 600
    assert cfg.compact_keep_messages == 4


def test_load_config_reads_env_overrides(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "Custom")
    monkeypatch.setenv("LLM_MODEL", "local-model-x")
    monkeypatch.setenv("CUSTOM_BASE_URL", "http://127.0.0.1:1234/v1")
    monkeypatch.setenv("COMPACT_THRESHOLD_TOKENS", "321")

    cfg = load_config(base_dir=tmp_path)

    assert cfg.model.provider == "custom"
    assert cfg.model.model_name == "local-model-x"
    assert cfg.model.base_url == "http://127.0.0.1:1234/v1"
    assert cfg.compact_threshold_tokens == 321


def test_judge_model_falls_back_to_main_model(tmp_path: Path) -> None:
    cfg = load_config(base_dir=tmp_path)

    assert isinstance(cfg.judge_model, ProviderConfig)
    assert cfg.judge_model.provider == cfg.model.provider
    assert cfg.judge_model.temperature == 0.0