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


PROVIDERS: dict[str, ProviderFactory] = {
    "anthropic": _anthropic,
    "openai_compat": _openai_compat,
}


def create_provider(profile: ModelProfile, secrets: SecretStore) -> Provider:
    try:
        factory = PROVIDERS[profile.provider]
    except KeyError:
        known = ", ".join(sorted(PROVIDERS))
        raise ValueError(f"unknown provider {profile.provider!r} (known: {known})") from None
    return factory(profile, secrets)
