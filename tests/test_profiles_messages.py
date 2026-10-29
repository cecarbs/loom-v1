from loom.broker.secrets import SecretStore
from loom.config import BUILTIN_MODEL_ALIASES, LoomConfig
from loom.core.messages import (
    ImageBlock,
    Message,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    message_from_dict,
    message_to_dict,
)
from loom.providers.base import ThinkingConfig
from loom.providers.profiles import BUILTIN_PROFILES, resolve_profile


def cfg(data: dict) -> LoomConfig:
    return LoomConfig.model_validate(data)


def test_builtin_aliases_match_config() -> None:
    assert set(BUILTIN_PROFILES) == BUILTIN_MODEL_ALIASES


def test_builtin_sonnet_profile() -> None:
    p = resolve_profile("sonnet", LoomConfig())
    assert (p.provider, p.model) == ("anthropic", "claude-sonnet-5-5")
    assert p.context_window == 1_000_000
    assert p.capabilities.prompt_cache and p.capabilities.thinking and p.capabilities.images


def test_config_overrides_only_explicit_keys() -> None:
    p = resolve_profile(
        "sonnet",
        cfg(
            {
                "models": {
                    "sonnet": {
                        "provider": "anthropic",
                        "model": "claude-sonnet-5-5",
                        "context_window": 200_000,
                        "thinking": {"effort": "low"},
                        "capabilities": {"images": False},
                    }
                }
            }
        ),
    )
    assert p.context_window == 200_000
    assert p.max_tokens == 64_000  # not reset to ModelEntry's default 8192
    assert p.pricing is not None and p.pricing.input == 2.0
    assert p.thinking == ThinkingConfig(effort="low")
    assert p.capabilities.images is False and p.capabilities.prompt_cache is True


def test_openai_compat_defaults_and_provider_secret() -> None:
    p = resolve_profile(
        "local",
        cfg(
            {
                "models": {"local": {"provider": "openai_compat", "model": "qwen3:8b"}},
                "providers": {"openai_compat": {"api_key_secret": "shared"}},
            }
        ),
    )
    assert p.capabilities.tools and not p.capabilities.images and not p.capabilities.thinking
    assert p.context_window == 32_768 and p.pricing is None
    assert p.api_key_secret == "shared"


def test_secret_store_reads_env() -> None:
    store = SecretStore({"LOOM_SECRET_OPENAI_API_KEY": "k"})
    assert store.get("openai_api_key") == "k"
    assert store.get("missing") is None


def test_message_roundtrip() -> None:
    msg = Message(
        "user",
        [
            TextBlock("hi"),
            ToolUseBlock("t1", "read", {"path": "a"}),
            ToolResultBlock("t1", [TextBlock("x"), ImageBlock("image/png", "AAAA")], True),
            ThinkingBlock("hmm", "anthropic", {"type": "thinking", "signature": "s"}),
        ],
        {"model": "m"},
    )
    assert message_from_dict(message_to_dict(msg)) == msg
