from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from model_provider import ProviderConfig, normalize_provider


@dataclass
class LabConfig:
    """Shared configuration for the lab.

    Holds paths, compact-memory thresholds, and provider settings for the main model
    and the judge model. Values come from environment variables (optionally via `.env`),
    with defaults chosen so the offline benchmark runs with no setup at all.
    """

    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig


def load_config(base_dir: Path | None = None) -> LabConfig:
    """Build a :class:`LabConfig` from the environment.

    Recognised variables:
        LLM_PROVIDER              one of openai|custom|gemini|anthropic|ollama|openrouter
        LLM_MODEL                 model id passed to the provider
        LLM_TEMPERATURE          float, defaults to 0.0 for reproducibility
        JUDGE_MODEL               optional; falls back to LLM_MODEL
        CUSTOM_BASE_URL          OpenAI-compatible base URL (e.g. http://127.0.0.1:1234/v1)
        CUSTOM_API_KEY           key for the custom endpoint
        OPENAI_API_KEY           fallback key when provider=openai
        COMPACT_THRESHOLD_TOKENS compact-memory trigger, default 600
        COMPACT_KEEP_MESSAGES    recent messages kept verbatim, default 4

    A `.env` file in the repo root is loaded when python-dotenv is installed. `.env` is
    gitignored, so credentials never reach a commit.
    """

    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()

    try:
        from dotenv import load_dotenv

        load_dotenv(root / ".env")
    except ImportError:
        # python-dotenv is optional; plain environment variables still work.
        pass

    provider = normalize_provider(os.getenv("LLM_PROVIDER", "custom"))
    model_name = os.getenv("LLM_MODEL", "gpt-4o-mini")
    temperature = float(os.getenv("LLM_TEMPERATURE", "0.0"))
    api_key = os.getenv("CUSTOM_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("CUSTOM_BASE_URL") or os.getenv("OPENAI_BASE_URL")

    model = ProviderConfig(
        provider=provider,
        model_name=model_name,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url,
    )

    judge_name = os.getenv("JUDGE_MODEL") or model_name
    judge_model = ProviderConfig(
        provider=provider,
        model_name=judge_name,
        temperature=0.0,
        api_key=api_key,
        base_url=base_url,
    )

    state_dir = root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    return LabConfig(
        base_dir=root,
        data_dir=root / "data",
        state_dir=state_dir,
        compact_threshold_tokens=int(os.getenv("COMPACT_THRESHOLD_TOKENS", "600")),
        compact_keep_messages=int(os.getenv("COMPACT_KEEP_MESSAGES", "4")),
        model=model,
        judge_model=judge_model,
    )
