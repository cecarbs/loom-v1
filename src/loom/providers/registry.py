"""Provider name → adapter. Adapters are imported lazily so unused SDKs aren't loaded."""

from __future__ import annotations

from collections.abc import Callable

from loom.broker.secrets import SecretStore
from loom.providers.base import Provider
from loom.providers.profiles import ModelProfile

ProviderFactory = Callable[[ModelProfile, SecretStore], Provider]


def _anthropic(profile: ModelProfile, secrets: SecretStore) -> Provider:
    from loom.providers.anthropic import AnthropicProvider

    return AnthropicProvider(profile, secrets)


def _openai_compat(profile: ModelProfile, secrets: SecretStore) -> Provider:
    from loom.providers.openai_compat import OpenAICompatProvider

    return OpenAICompatProvider(profile, secrets)


def _openai_responses(profile: ModelProfile, secrets: SecretStore) -> Provider:
    from loom.providers.openai_responses import OpenAIResponsesProvider

    return OpenAIResponsesProvider(profile, secrets)


PROVIDERS: dict[str, ProviderFactory] = {
    "anthropic": _anthropic,
    "openai_compat": _openai_compat,
    "openai_responses": _openai_responses,
}


def create_provider(
    profile: ModelProfile, secrets: SecretStore, *, text_tools: bool = True
) -> Provider:
    """`text_tools`: wrap models without native tool calling in the text fallback."""
    try:
        factory = PROVIDERS[profile.provider]
    except KeyError:
        known = ", ".join(sorted(PROVIDERS))
        raise ValueError(f"unknown provider {profile.provider!r} (known: {known})") from None
    provider = factory(profile, secrets)
    if text_tools and not profile.capabilities.tools:
        from loom.providers.text_tools import TextToolCallProvider

        provider = TextToolCallProvider(provider)
    return provider
