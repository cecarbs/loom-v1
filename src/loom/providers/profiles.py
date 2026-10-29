"""Model profiles and capabilities; built-in defaults. See docs/02 "Model profiles"."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

from loom.config import LoomConfig, ModelEntry
from loom.providers.base import ThinkingConfig


@dataclass(frozen=True)
class ModelCapabilities:
    tools: bool = True
    parallel_tool_calls: bool = True
    images: bool = False
    thinking: bool = False
    prompt_cache: bool = False
    streaming: bool = True
    system_role: Literal["system", "developer", "user"] = "system"


@dataclass(frozen=True)
class Pricing:
    """USD per million tokens. Cache prices default to the input price when unknown."""

    input: float
    output: float
    cache_read: float | None = None
    cache_write: float | None = None

    def cost(
        self, input_tokens: int, output_tokens: int, cache_read: int, cache_write: int
    ) -> float:
        cr = self.input if self.cache_read is None else self.cache_read
        cw = self.input if self.cache_write is None else self.cache_write
        return (
            input_tokens * self.input
            + output_tokens * self.output
            + cache_read * cr
            + cache_write * cw
        ) / 1_000_000


@dataclass(frozen=True)
class ModelProfile:
    alias: str
    provider: str
    model: str
    base_url: str | None = None
    api_key_secret: str | None = None
    context_window: int = 32_768
    max_tokens: int = 8192
    capabilities: ModelCapabilities = field(default_factory=ModelCapabilities)
    headers: dict[str, str] = field(default_factory=dict)
    extra_body: dict[str, Any] = field(default_factory=dict)
    thinking: ThinkingConfig | None = None
    pricing: Pricing | None = None
    timeout_s: float = 600
    verify_tls: bool = True
    ca_bundle: Path | None = None


PROVIDER_CAPABILITIES: dict[str, ModelCapabilities] = {
    "anthropic": ModelCapabilities(images=True, thinking=True, prompt_cache=True),
    "openai_compat": ModelCapabilities(),
}

# Values from Anthropic's model table (2026-10): 1M context, 128K max output,
# $2 / $10 per MTok, cache reads $0.20, 5-minute cache writes 1.25x input.
BUILTIN_PROFILES: dict[str, ModelProfile] = {
    "sonnet": ModelProfile(
        alias="sonnet",
        provider="anthropic",
        model="claude-sonnet-5-5",
        context_window=1_000_000,
        max_tokens=64_000,
        capabilities=PROVIDER_CAPABILITIES["anthropic"],
        pricing=Pricing(input=2.0, output=10.0, cache_read=0.20, cache_write=2.50),
    ),
}

# Credential env vars the SDKs read when no secret is configured (docs/09).
PROVIDER_KEY_ENV: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai_compat": "OPENAI_API_KEY",
}


def resolve_profile(alias: str, config: LoomConfig) -> ModelProfile:
    """Merge a built-in profile (if any) with the config entry; only explicitly set keys win."""
    entry = config.models.get(alias)
    base = BUILTIN_PROFILES.get(alias)
    if entry is None and base is None:
        raise KeyError(f"unknown model alias {alias!r}")
    if entry is None:
        assert base is not None
        return _with_provider_secret(base, config)

    if base is None or entry.provider != base.provider:
        base = ModelProfile(
            alias=alias,
            provider=entry.provider,
            model=entry.model,
            capabilities=PROVIDER_CAPABILITIES.get(entry.provider, ModelCapabilities()),
        )
    return _with_provider_secret(_apply_entry(base, entry), config)


def _apply_entry(base: ModelProfile, entry: ModelEntry) -> ModelProfile:
    changes: dict[str, Any] = {}
    for name in entry.model_fields_set:
        value = getattr(entry, name)
        if name == "capabilities":
            caps = {k: v for k, v in value.model_dump().items() if v is not None}
            changes[name] = replace(base.capabilities, **caps)
        elif name == "thinking":
            changes[name] = ThinkingConfig(**value.model_dump()) if value else None
        elif name == "pricing":
            changes[name] = Pricing(**value.model_dump()) if value else None
        else:
            changes[name] = value
    return replace(base, **changes)


def _with_provider_secret(profile: ModelProfile, config: LoomConfig) -> ModelProfile:
    if profile.api_key_secret is not None:
        return profile
    provider_entry = config.providers.get(profile.provider)
    if provider_entry and provider_entry.api_key_secret:
        return replace(profile, api_key_secret=provider_entry.api_key_secret)
    return profile
