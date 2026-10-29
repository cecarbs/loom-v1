"""OpenAI Responses API adapter (docs/02 "Adapters").

Stateless: `store=false`, the whole history is sent each time, and reasoning items are kept as
`ThinkingBlock(provider="openai_responses", opaque=<item>)` with their encrypted content and
replayed only to this adapter. `function_call` / `function_call_output` items map to
`ToolUseBlock` / `ToolResultBlock`.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx2
import openai

from loom.broker.secrets import SecretStore
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
    INVALID_JSON_KEY,
    MessageEnd,
    ModelRequest,
    ProviderError,
    ProviderEvent,
    TextDelta,
    ThinkingDelta,
    ToolUseEnd,
    ToolUseInputDelta,
    ToolUseStart,
    estimate_tokens,
    image_notice,
)
from loom.providers.openai_compat import CLIENT_ERRORS, make_client, map_error, parse_args
from loom.providers.profiles import ModelCapabilities, ModelProfile

PROVIDER = "openai_responses"
_ERROR_KINDS = {
    "context_length_exceeded": "context_too_long",
    "rate_limit_exceeded": "rate_limit",
    "server_error": "overloaded",
}


class OpenAIResponsesProvider:
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
        self._client: openai.AsyncOpenAI | None = None

    async def stream(self, req: ModelRequest) -> AsyncIterator[ProviderEvent]:
        params = build_params(req, self.profile)
        final: dict[str, Any] | None = None
        try:
            if self._client is None:
                self._client = make_client(self.profile, self._secrets, self._http_client)
            response = await self._client.responses.create(
                **params, extra_body=self.profile.extra_body or None
            )
            if not params["stream"]:
                data: dict[str, Any] = response.model_dump(mode="json")
                final = data
                for event in _replay_events(data):
                    yield event
            else:
                call_ids: dict[str, str] = {}  # output item id -> call_id
                async for event in response:
                    kind = event.type
                    if kind == "response.output_text.delta":
                        yield TextDelta(event.delta)
                    elif kind in (
                        "response.reasoning_summary_text.delta",
                        "response.reasoning_text.delta",
                    ):
                        yield ThinkingDelta(event.delta)
                    elif kind == "response.output_item.added":
                        item = event.item
                        if item.type == "function_call":
                            call_ids[item.id or item.call_id] = item.call_id
                            yield ToolUseStart(item.call_id, item.name)
                    elif kind == "response.function_call_arguments.delta":
                        if event.item_id in call_ids:
                            yield ToolUseInputDelta(call_ids[event.item_id], event.delta)
                    elif kind in ("response.completed", "response.incomplete"):
                        final = event.response.model_dump(mode="json")
                    elif kind == "response.failed":
                        error = event.response.error
                        yield _stream_error(
                            error.code if error else None,
                            error.message if error else "response failed",
                        )
                        return
                    elif kind == "error":
                        yield _stream_error(event.code, event.message)
                        return
        except Exception as e:
            if not isinstance(e, CLIENT_ERRORS):
                raise
            yield map_error(e, self.profile.base_url)
            return

        if final is None:
            yield ProviderError("other", "response stream ended without a final response")
            return
        message = from_responses(final)
        for tool_use in message.tool_uses():
            yield ToolUseEnd(tool_use.id, tool_use.input)
        yield MessageEnd(message.meta["stop_reason"], usage_from(final.get("usage")), message)

    def count_tokens_estimate(self, messages: list[Message]) -> int:
        return estimate_tokens(messages)


# --- request ---------------------------------------------------------------


def build_params(req: ModelRequest, profile: ModelProfile) -> dict[str, Any]:
    caps = profile.capabilities
    params: dict[str, Any] = {
        "model": req.model,
        "input": to_responses(req.system, req.messages, caps),
        "max_output_tokens": req.max_tokens,
        "store": False,
        "stream": caps.streaming,
    }
    if caps.system_role == "system":
        params["instructions"] = req.system
    if req.tools:
        params["tools"] = [
            {
                "type": "function",
                "name": t.name,
                "description": t.description,
                "parameters": t.input_schema,
                "strict": False,
            }
            for t in req.tools
        ]
        if not caps.parallel_tool_calls:
            params["parallel_tool_calls"] = False
    if req.temperature is not None:
        params["temperature"] = req.temperature
    if caps.thinking:
        # Encrypted reasoning makes reasoning items replayable without server-side state.
        params["include"] = ["reasoning.encrypted_content"]
        if req.thinking and req.thinking.effort:
            params["reasoning"] = {"effort": req.thinking.effort, "summary": "auto"}
    return params


def to_responses(system: str, messages: list[Message], caps: ModelCapabilities) -> list[dict]:
    items: list[dict[str, Any]] = []
    prefix = ""
    if caps.system_role == "developer":
        items.append({"role": "developer", "content": system})
    elif caps.system_role == "user":
        prefix = system + "\n\n"

    for msg in messages:
        if msg.role == "assistant":
            for block in msg.content:
                if isinstance(block, TextBlock) and block.text:
                    items.append({"role": "assistant", "content": block.text})
                elif isinstance(block, ToolUseBlock):
                    items.append(
                        {
                            "type": "function_call",
                            "call_id": block.id,
                            "name": block.name,
                            "arguments": _dump_args(block.input),
                        }
                    )
                elif isinstance(block, ThinkingBlock) and _replayable(block, caps):
                    items.append(dict(block.opaque))
            continue

        parts: list[dict[str, Any]] = []
        for block in msg.content:
            if isinstance(block, ToolResultBlock):
                items.append(
                    {
                        "type": "function_call_output",
                        "call_id": block.tool_use_id,
                        "output": _result_output(block, caps),
                    }
                )
            elif isinstance(block, TextBlock | ImageBlock):
                parts.append(_part(block, caps))
        if parts:
            if prefix:
                parts.insert(0, {"type": "input_text", "text": prefix})
                prefix = ""
            items.append({"role": "user", "content": parts})
    return items


def _replayable(block: ThinkingBlock, caps: ModelCapabilities) -> bool:
    # Without stored state, a reasoning item can only be replayed with its encrypted content.
    return (
        block.provider == PROVIDER and caps.thinking and bool(block.opaque.get("encrypted_content"))
    )


def _result_output(block: ToolResultBlock, caps: ModelCapabilities) -> str | list[dict]:
    error = "[error] " if block.is_error else ""
    if isinstance(block.content, str):
        return error + block.content
    if not any(isinstance(b, ImageBlock) for b in block.content):
        return error + "".join(b.text for b in block.content if isinstance(b, TextBlock))
    parts = [_part(b, caps) for b in block.content]
    if error:
        parts.insert(0, {"type": "input_text", "text": error})
    return parts


def _part(block: TextBlock | ImageBlock, caps: ModelCapabilities) -> dict[str, Any]:
    if isinstance(block, ImageBlock):
        if not caps.images:
            return {"type": "input_text", "text": image_notice().text}
        url = f"data:{block.media_type};base64,{block.data_b64}"
        return {"type": "input_image", "image_url": url, "detail": "auto"}
    return {"type": "input_text", "text": block.text}


def _dump_args(args: dict[str, Any]) -> str:
    if INVALID_JSON_KEY in args:
        return str(args[INVALID_JSON_KEY])
    return json.dumps(args)


# --- response --------------------------------------------------------------


def from_responses(data: dict[str, Any]) -> Message:
    content: list[Block] = []
    for item in data.get("output") or []:
        kind = item.get("type")
        if kind == "message":
            text = "".join(
                part.get("text") or ""
                for part in item.get("content") or []
                if part.get("type") == "output_text"
            )
            if text:
                content.append(TextBlock(text))
        elif kind == "function_call":
            args = parse_args(item.get("arguments") or "")
            content.append(ToolUseBlock(item["call_id"], item["name"], args))
        elif kind == "reasoning":
            summary = "\n\n".join(s.get("text") or "" for s in item.get("summary") or [])
            opaque = {
                k: v for k, v in item.items() if v is not None and k not in ("status", "content")
            }
            content.append(ThinkingBlock(summary, PROVIDER, opaque))
    if any(isinstance(b, ToolUseBlock) for b in content):
        stop = "tool_use"
    elif (data.get("incomplete_details") or {}).get("reason") == "max_output_tokens":
        stop = "max_tokens"
    else:
        stop = "end_turn"
    return Message("assistant", content, {"model": data.get("model"), "stop_reason": stop})


def _replay_events(data: dict[str, Any]) -> list[ProviderEvent]:
    """Stream events for a non-streamed response (`streaming = false`)."""
    events: list[ProviderEvent] = []
    for block in from_responses(data).content:
        if isinstance(block, ThinkingBlock) and block.text:
            events.append(ThinkingDelta(block.text))
        elif isinstance(block, TextBlock):
            events.append(TextDelta(block.text))
        elif isinstance(block, ToolUseBlock):
            events.append(ToolUseStart(block.id, block.name))
    return events


def usage_from(usage: dict[str, Any] | None) -> Usage:
    if not usage:
        return Usage()
    cached = (usage.get("input_tokens_details") or {}).get("cached_tokens") or 0
    return Usage(
        input_tokens=(usage.get("input_tokens") or 0) - cached,
        output_tokens=usage.get("output_tokens") or 0,
        cache_read_tokens=cached,
    )


def _stream_error(code: str | None, message: str) -> ProviderError:
    kind = _ERROR_KINDS.get(code or "", "other")
    return ProviderError(kind, message)  # type: ignore[arg-type]
