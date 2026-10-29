"""Provider protocol, request type, and the provider event stream. See docs/02."""

from __future__ import annotations

import ssl
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol

from loom.core.events import Usage
from loom.core.messages import ImageBlock, Message, TextBlock, ToolResultBlock

if TYPE_CHECKING:
    from loom.broker.secrets import SecretStore
    from loom.providers.profiles import ModelProfile


@dataclass(frozen=True)
class ToolSchema:
    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass(frozen=True)
class ThinkingConfig:
    effort: Literal["low", "medium", "high", "xhigh", "max"] | None = None
    budget_tokens: int | None = None  # only for older models that still take a fixed budget


@dataclass
class ModelRequest:
    model: str
    system: str
    messages: list[Message]
    tools: list[ToolSchema]
    max_tokens: int
    temperature: float | None = None  # never sent to Anthropic
    thinking: ThinkingConfig | None = None
    cache: bool = True


# --- provider events ---------------------------------------------------------


@dataclass(frozen=True)
class TextDelta:
    text: str


@dataclass(frozen=True)
class ThinkingDelta:
    text: str


@dataclass(frozen=True)
class ToolUseStart:
    id: str
    name: str


@dataclass(frozen=True)
class ToolUseInputDelta:
    id: str
    partial_json: str


@dataclass(frozen=True)
class ToolUseEnd:
    id: str
    input: dict[str, Any]


@dataclass(frozen=True)
class MessageEnd:
    stop_reason: str
    usage: Usage
    final: Message


ErrorKind = Literal["rate_limit", "overloaded", "context_too_long", "auth", "other"]
RETRYABLE: frozenset[str] = frozenset({"rate_limit", "overloaded"})


@dataclass(frozen=True)
class ProviderError:
    kind: ErrorKind
    message: str
    retry_after: float | None = None


ProviderEvent = (
    TextDelta
    | ThinkingDelta
    | ToolUseStart
    | ToolUseInputDelta
    | ToolUseEnd
    | MessageEnd
    | ProviderError
)


class Provider(Protocol):
    name: str

    def stream(self, req: ModelRequest) -> AsyncIterator[ProviderEvent]: ...

    def count_tokens_estimate(self, messages: list[Message]) -> int: ...


# Set on a ToolUseBlock's input when the model sent arguments that aren't valid JSON.
INVALID_JSON_KEY = "__invalid_json__"


def estimate_tokens(messages: list[Message]) -> int:
    """Cheap heuristic: ~4 characters per token."""
    chars = 0
    for msg in messages:
        for block in msg.content:
            chars += len(str(block))
    return chars // 4


def image_notice() -> TextBlock:
    return TextBlock("[image omitted: model does not accept images]")


def without_images(blocks: list[Any]) -> list[Any]:
    """Replace image blocks (also inside tool results) with a text notice."""
    out: list[Any] = []
    for b in blocks:
        if isinstance(b, ImageBlock):
            out.append(image_notice())
        elif isinstance(b, ToolResultBlock) and not isinstance(b.content, str):
            content = [image_notice() if isinstance(c, ImageBlock) else c for c in b.content]
            out.append(ToolResultBlock(b.tool_use_id, content, b.is_error))  # type: ignore[arg-type]
        else:
            out.append(b)
    return out


def resolve_headers(headers: dict[str, str], secrets: SecretStore) -> dict[str, str]:
    """Resolve `secret:<name>` header values from the secret store."""
    resolved: dict[str, str] = {}
    for key, value in headers.items():
        if value.startswith("secret:"):
            secret = secrets.get(value.removeprefix("secret:"))
            if secret is None:
                raise LookupError(f"header {key!r}: secret {value!r} is not set")
            value = secret
        resolved[key] = value
    return resolved


class ProviderSetupError(Exception):
    """The profile can't be turned into a working client (e.g. an unreadable ca_bundle)."""


def tls_verify(profile: ModelProfile) -> bool | ssl.SSLContext:
    """`verify` for the HTTP client: False, True (system CAs) or a context with `ca_bundle`."""
    if not profile.verify_tls:
        return False
    if profile.ca_bundle is None:
        return True
    path = profile.ca_bundle.expanduser()
    try:
        return ssl.create_default_context(cafile=str(path))
    except (OSError, ssl.SSLError) as e:
        raise ProviderSetupError(f"ca_bundle {path}: {e}") from e


def api_key_for(profile: ModelProfile, secrets: SecretStore) -> str | None:
    return secrets.get(profile.api_key_secret) if profile.api_key_secret else None
