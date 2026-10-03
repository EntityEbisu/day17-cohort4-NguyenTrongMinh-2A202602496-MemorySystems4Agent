from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ProviderConfig:
    """Student TODO: define the provider configuration shared by the agents.

    Required providers for this lab:
    - openai
    - custom (OpenAI-compatible base URL)
    - gemini
    - anthropic
    - ollama
    - openrouter
    """

    provider: str
    model_name: str
    temperature: float
    api_key: str | None = None
    base_url: str | None = None


def normalize_provider(value: str) -> str:
    """Map a provider name or common typo/alias to a canonical provider id."""

    key = (value or "").strip().lower()
    aliases = {
        "anthropic": "anthropic",
        "anthorpic": "anthropic",          # common typo
        "claude": "anthropic",
        "openai": "openai",
        "custom": "custom",
        "openai-compatible": "custom",
        "openai_compatible": "custom",
        "local": "custom",
        "lmstudio": "custom",
        "gemini": "gemini",
        "google": "gemini",
        "ollama": "ollama",
        "openrouter": "openrouter",
    }
    return aliases.get(key, key)


def build_chat_model(config: ProviderConfig):
    """Instantiate a real chat model for the selected provider.

    All provider imports are lazy and local to their branch. This module must stay
    importable without LangChain installed, because the whole lab runs offline by
    default and only touches a provider when live mode is explicitly requested.
    """

    provider = normalize_provider(config.provider)

    if provider in ("openai", "custom"):
        from langchain_openai import ChatOpenAI

        kwargs: dict[str, object] = {
            "model": config.model_name,
            "temperature": config.temperature,
        }
        if config.api_key:
            kwargs["api_key"] = config.api_key
        # `custom` means an OpenAI-compatible endpoint; a base_url is mandatory there.
        if provider == "custom" or config.base_url:
            kwargs["base_url"] = config.base_url
        return ChatOpenAI(**kwargs)

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=config.model_name,
            temperature=config.temperature,
            google_api_key=config.api_key,
        )

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        # Note: Anthropic uses `model=`, not `model_name=`.
        return ChatAnthropic(
            model=config.model_name,
            temperature=config.temperature,
            api_key=config.api_key,
        )

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=config.model_name,
            temperature=config.temperature,
            base_url=config.base_url,
        )

    if provider == "openrouter":
        from langchain_openrouter import ChatOpenRouter

        return ChatOpenRouter(
            model=config.model_name,
            temperature=config.temperature,
            openrouter_api_key=config.api_key,
        )

    raise ValueError(
        f"Unsupported provider: {config.provider!r}. "
        "Expected one of: openai, custom, gemini, anthropic, ollama, openrouter."
    )
