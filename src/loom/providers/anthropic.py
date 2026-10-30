"""Anthropic Messages API adapter (streaming, prompt caching, thinking replay). See docs/02."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import anthropic
import httpx2

from loom.broker.secrets import SecretStore, env_name
from loom.core.events import Usage
from loom.core.messages import (
    Block,
    ImageBlock,
    Message,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from loom.providers.base import (
    MessageEnd,
    ModelRequest,
    ProviderError,
    ProviderEvent,
    ProviderSetupError,
    TextDelta,
    ThinkingDelta,
    ToolUseEnd,
    ToolUseInputDelta,
    ToolUseStart,
    api_key_for,
    estimate_tokens,
    resolve_headers,
    tls_verify,
    without_images,
)
from loom.providers.profiles import ModelProfile

PROVIDER = "anthropic"
_CACHE = {"type": "ephemeral"}


class AnthropicProvider:
    name = PROVIDER

    def __init__(
        self,
        profile: ModelProfile,
        secrets: SecretStore,
        *,
        http_client: httpx2.AsyncClient | None = None,
    ) -> None:
        self.profile = profile
        self._secrets = secrets
        self._http_client = http_client
        self._client: anthropic.AsyncAnthropic | None = None

    def _get_client(self) -> anthropic.AsyncAnthropic:
        if self._client is None:
            kwargs: dict[str, Any] = {
                "base_url": self.profile.base_url,
                "timeout": self.profile.timeout_s,
                "max_retries": 0,  # the agent loop owns retries
                "default_headers": resolve_headers(self.profile.headers, self._secrets) or None,
                "http_client": self._http_client,
            }
            if self._http_client is None and (verify := tls_verify(self.profile)) is not True:
                kwargs["http_client"] = anthropic.DefaultAsyncHttpxClient(verify=verify)
            # Without a configured secret, the SDK's own chain applies (ANTHROPIC_API_KEY,
            # `ant auth login` profiles, ...).
            if api_key := api_key_for(self.profile, self._secrets):
                kwargs["api_key"] = api_key
            self._client = anthropic.AsyncAnthropic(**kwargs)
        return self._client

    def _no_credentials_message(self) -> str:
        name = self.profile.api_key_secret
        if name is None:
            return (
                "no Anthropic API key: set [providers.anthropic] api_key_secret and store it "
                "with `loom secrets set <name>`, or set ANTHROPIC_API_KEY"
            )
        return (
            f"no Anthropic API key: secret {name!r} is not set (`loom secrets set {name}`, "
            f"or {env_name(name)}) and ANTHROPIC_API_KEY is unset"
        )

    async def stream(self, req: ModelRequest) -> AsyncIterator[ProviderEvent]:
        params = build_params(req, self.profile)
        extra_body = self.profile.extra_body or None
        tool_ids: dict[int, str] = {}
        try:
            client = self._get_client()
            async with client.messages.stream(**params, extra_body=extra_body) as stream:
                async for event in stream:
                    if event.type == "content_block_start":
                        block = event.content_block
                        if block.type == "tool_use":
                            tool_ids[event.index] = block.id
                            yield ToolUseStart(block.id, block.name)
                    elif event.type == "content_block_delta":
                        delta = event.delta
                        if delta.type == "text_delta":
                            yield TextDelta(delta.text)
                        elif delta.type == "thinking_delta":
                            yield ThinkingDelta(delta.thinking)
                        elif delta.type == "input_json_delta" and event.index in tool_ids:
                            yield ToolUseInputDelta(tool_ids[event.index], delta.partial_json)
                final = await stream.get_final_message()
        except Exception as e:  # SDK and transport errors become ProviderErrors
            if _no_credentials(e):
                yield ProviderError("auth", self._no_credentials_message())
                return
            if not isinstance(
                e, anthropic.AnthropicError | httpx2.HTTPError | LookupError | ProviderSetupError
            ):
                raise
            yield map_error(e)
            return

        message = from_anthropic(final.model_dump(mode="json"))
        for tool_use in message.tool_uses():
            yield ToolUseEnd(tool_use.id, tool_use.input)
        yield MessageEnd(message.meta["stop_reason"], usage_from(final.usage), message)

    def count_tokens_estimate(self, messages: list[Message]) -> int:
        return estimate_tokens(messages)


# --- request ---------------------------------------------------------------


def build_params(req: ModelRequest, profile: ModelProfile) -> dict[str, Any]:
    caps = profile.capabilities
    cache = req.cache and caps.prompt_cache
    messages = [m for m in (to_anthropic(m, images=caps.images) for m in req.messages) if m]

    system: dict[str, Any] = {"type": "text", "text": req.system}
    if cache:
        system["cache_control"] = _CACHE
        if messages:
            last = messages[-1]["content"][-1]
            if last["type"] not in ("thinking", "redacted_thinking"):
                last["cache_control"] = _CACHE

    params: dict[str, Any] = {
        "model": req.model,
        "max_tokens": req.max_tokens,
        "system": [system],
        "messages": messages,
    }
    if req.tools:
        params["tools"] = [
            {"name": t.name, "description": t.description, "input_schema": t.input_schema}
            for t in req.tools
        ]
    if req.thinking and caps.thinking:
        if req.thinking.effort:
            params["thinking"] = {"type": "adaptive", "display": "summarized"}
            params["output_config"] = {"effort": req.thinking.effort}
        elif req.thinking.budget_tokens:
            params["thinking"] = {"type": "enabled", "budget_tokens": req.thinking.budget_tokens}
    return params


def to_anthropic(msg: Message, *, images: bool) -> dict[str, Any] | None:
    blocks = msg.content if images else without_images(msg.content)
    content = [c for c in (_block(b) for b in blocks) if c is not None]
    if not content:
        return None
    return {"role": msg.role, "content": content}


def _block(block: Block) -> dict[str, Any] | None:
    match block:
        case TextBlock(text):
            return {"type": "text", "text": text} if text else None
        case ToolUseBlock(id_, name, input_):
            return {"type": "tool_use", "id": id_, "name": name, "input": input_}
        case ToolResultBlock(tool_use_id, content, is_error):
            body: Any = content
            if not isinstance(content, str):
                body = [b for b in (_block(c) for c in content) if b is not None]
            return {
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": body,
                "is_error": is_error,
            }
        case ImageBlock(media_type, data):
            return {
                "type": "image",
                "source": {"type": "base64", "media_type": media_type, "data": data},
            }
        case ThinkingBlock(provider=provider, opaque=opaque):
            # Replay verbatim; the API drops blocks the current model can't read.
            return dict(opaque) if provider == PROVIDER else None


# --- response --------------------------------------------------------------


def from_anthropic(data: dict[str, Any]) -> Message:
    content: list[Block] = []
    for block in data.get("content", []):
        kind = block.get("type")
        if kind == "text":
            content.append(TextBlock(block["text"]))
        elif kind == "tool_use":
            content.append(ToolUseBlock(block["id"], block["name"], dict(block.get("input") or {})))
        elif kind in ("thinking", "redacted_thinking"):
            opaque = {k: v for k, v in block.items() if v is not None}
            content.append(ThinkingBlock(block.get("thinking") or "", PROVIDER, opaque))
    meta = {"model": data.get("model"), "stop_reason": data.get("stop_reason") or "end_turn"}
    return Message("assistant", content, meta)


def usage_from(usage: Any) -> Usage:
    return Usage(
        input_tokens=usage.input_tokens or 0,
        output_tokens=usage.output_tokens or 0,
        cache_read_tokens=getattr(usage, "cache_read_input_tokens", None) or 0,
        cache_write_tokens=getattr(usage, "cache_creation_input_tokens", None) or 0,
    )


def _no_credentials(e: Exception) -> bool:
    """The SDK raises a bare TypeError at request time when no key, token or credentials
    profile resolved."""
    return isinstance(e, TypeError) and "Could not resolve authentication method" in str(e)


def map_error(e: Exception) -> ProviderError:
    message = getattr(e, "message", None) or str(e)
    if isinstance(e, ProviderSetupError):
        return ProviderError("other", message)
    if isinstance(e, anthropic.RateLimitError):
        return ProviderError("rate_limit", message, _retry_after(e))
    if isinstance(
        e,
        anthropic.OverloadedError
        | anthropic.InternalServerError
        | anthropic.ServiceUnavailableError,
    ):
        return ProviderError("overloaded", message, _retry_after(e))
    if isinstance(e, anthropic.AuthenticationError | anthropic.PermissionDeniedError):
        return ProviderError("auth", message)
    if isinstance(e, anthropic.CredentialsError | LookupError):
        return ProviderError("auth", f"no Anthropic credentials: {message}")
    if isinstance(e, anthropic.BadRequestError | anthropic.RequestTooLargeError):
        lowered = message.lower()
        if "prompt is too long" in lowered or ("context" in lowered and "too long" in lowered):
            return ProviderError("context_too_long", message)
    if isinstance(e, anthropic.APIConnectionError):
        return ProviderError("other", f"cannot reach the Anthropic API: {message}")
    return ProviderError("other", message)


def _retry_after(e: Exception) -> float | None:
    response = getattr(e, "response", None)
    value = response.headers.get("retry-after") if response is not None else None
    try:
        return float(value) if value else None
    except ValueError:
        return None
