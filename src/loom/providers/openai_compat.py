"""OpenAI-compatible Chat Completions adapter (OpenAI, OpenRouter, Ollama, vLLM, ...).

M1 core: streaming text and tool calls, tool messages, base_url / extra_body / timeout_s.
Capability shaping beyond images/system_role/parallel calls arrives in M7.
"""

from __future__ import annotations

import json
import os
import uuid
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
    api_key_for,
    estimate_tokens,
    image_notice,
    resolve_headers,
)
from loom.providers.profiles import ModelCapabilities, ModelProfile

_STOP_REASONS = {"tool_calls": "tool_use", "length": "max_tokens", "stop": "end_turn"}


class OpenAICompatProvider:
    name = "openai_compat"

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

    def _get_client(self) -> openai.AsyncOpenAI:
        if self._client is None:
            api_key = api_key_for(self.profile, self._secrets)
            if api_key is None and self.profile.base_url is None:
                api_key = os.environ.get("OPENAI_API_KEY")  # only for the official endpoint
            self._client = openai.AsyncOpenAI(
                api_key=api_key or "no-key",  # local servers ignore it; the SDK requires one
                base_url=self.profile.base_url,
                timeout=self.profile.timeout_s,
                max_retries=0,
                default_headers=resolve_headers(self.profile.headers, self._secrets) or None,
                http_client=self._http_client,
            )
        return self._client

    async def stream(self, req: ModelRequest) -> AsyncIterator[ProviderEvent]:
        params = build_params(req, self.profile)
        text: list[str] = []
        calls: dict[int, dict[str, Any]] = {}
        finish = "stop"
        usage = Usage()
        try:
            client = self._get_client()
            response = await client.chat.completions.create(
                **params, extra_body=self.profile.extra_body or None
            )
            async for chunk in response:
                if chunk.usage:
                    usage = _usage(chunk.usage)
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                delta = choice.delta
                if reasoning := _reasoning(delta):
                    yield ThinkingDelta(reasoning)
                if delta.content:
                    text.append(delta.content)
                    yield TextDelta(delta.content)
                for tc in delta.tool_calls or []:
                    for event in _accumulate(calls, tc):
                        yield event
                if choice.finish_reason:
                    finish = choice.finish_reason
        except Exception as e:
            if not isinstance(e, openai.OpenAIError | httpx2.HTTPError | LookupError):
                raise
            yield map_error(e, self.profile.base_url)
            return

        content: list[Block] = []
        if joined := "".join(text):
            content.append(TextBlock(joined))
        for index in sorted(calls):
            call = calls[index]
            tool_use = ToolUseBlock(call["id"], call["name"], _parse_args(call["args"]))
            content.append(tool_use)
            yield ToolUseEnd(tool_use.id, tool_use.input)
        if calls and finish == "stop":
            finish = "tool_calls"  # some servers report "stop" alongside tool calls
        stop_reason = _STOP_REASONS.get(finish, finish)
        meta = {"model": req.model, "stop_reason": stop_reason}
        yield MessageEnd(stop_reason, usage, Message("assistant", content, meta))

    def count_tokens_estimate(self, messages: list[Message]) -> int:
        return estimate_tokens(messages)


# --- request ---------------------------------------------------------------


def build_params(req: ModelRequest, profile: ModelProfile) -> dict[str, Any]:
    caps = profile.capabilities
    params: dict[str, Any] = {
        "model": req.model,
        "messages": to_openai(req.system, req.messages, caps),
        "max_tokens": req.max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if req.tools:
        params["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.input_schema,
                },
            }
            for t in req.tools
        ]
        if not caps.parallel_tool_calls:
            params["parallel_tool_calls"] = False
    if req.temperature is not None:
        params["temperature"] = req.temperature
    if req.thinking and caps.thinking and req.thinking.effort:
        params["reasoning_effort"] = req.thinking.effort
    return params


def to_openai(system: str, messages: list[Message], caps: ModelCapabilities) -> list[dict]:
    out: list[dict[str, Any]] = []
    prefix = ""
    if caps.system_role == "user":
        prefix = system + "\n\n"
    else:
        out.append({"role": caps.system_role, "content": system})

    for msg in messages:
        if msg.role == "assistant":
            text = msg.text()
            tool_calls = [
                {
                    "id": b.id,
                    "type": "function",
                    "function": {"name": b.name, "arguments": _dump_args(b.input)},
                }
                for b in msg.tool_uses()
            ]
            if not text and not tool_calls:
                continue
            entry: dict[str, Any] = {"role": "assistant", "content": text or None}
            if tool_calls:
                entry["tool_calls"] = tool_calls
            out.append(entry)
            continue

        # User message: tool results become role=tool messages; images from tool results
        # (which tool messages can't carry) and the user's own blocks follow as a user message.
        parts: list[dict[str, Any]] = []
        for block in msg.content:
            if isinstance(block, ToolResultBlock):
                text, images = _split_result(block)
                out.append({"role": "tool", "tool_call_id": block.tool_use_id, "content": text})
                parts.extend(_part(img, caps) for img in images)
            elif isinstance(block, TextBlock | ImageBlock):
                parts.append(_part(block, caps))
        if parts:
            if prefix:
                parts.insert(0, {"type": "text", "text": prefix})
                prefix = ""
            if all(p["type"] == "text" for p in parts):
                out.append({"role": "user", "content": "".join(p["text"] for p in parts)})
            else:
                out.append({"role": "user", "content": parts})
    return out


def _split_result(block: ToolResultBlock) -> tuple[str, list[ImageBlock]]:
    if isinstance(block.content, str):
        text, images = block.content, []
    else:
        text = "".join(b.text for b in block.content if isinstance(b, TextBlock))
        images = [b for b in block.content if isinstance(b, ImageBlock)]
    return (f"[error] {text}" if block.is_error else text), images


def _part(block: TextBlock | ImageBlock, caps: ModelCapabilities) -> dict[str, Any]:
    if isinstance(block, ImageBlock):
        if not caps.images:
            return {"type": "text", "text": image_notice().text}
        url = f"data:{block.media_type};base64,{block.data_b64}"
        return {"type": "image_url", "image_url": {"url": url}}
    return {"type": "text", "text": block.text}


def _dump_args(args: dict[str, Any]) -> str:
    if INVALID_JSON_KEY in args:
        return str(args[INVALID_JSON_KEY])
    return json.dumps(args)


# --- response --------------------------------------------------------------


def _accumulate(calls: dict[int, dict[str, Any]], tc: Any) -> list[ProviderEvent]:
    events: list[ProviderEvent] = []
    call = calls.setdefault(tc.index, {"id": None, "name": "", "args": "", "started": False})
    if tc.id:
        call["id"] = tc.id
    fn = tc.function
    if fn is not None and fn.name:
        call["name"] = fn.name
    if not call["started"] and call["name"]:
        call["id"] = call["id"] or f"call_{uuid.uuid4().hex[:12]}"
        call["started"] = True
        events.append(ToolUseStart(call["id"], call["name"]))
    if fn is not None and fn.arguments:
        call["args"] += fn.arguments
        if call["started"]:
            events.append(ToolUseInputDelta(call["id"], fn.arguments))
    return events


def _parse_args(raw: str) -> dict[str, Any]:
    if not raw.strip():
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {INVALID_JSON_KEY: raw}
    return value if isinstance(value, dict) else {INVALID_JSON_KEY: raw}


def _reasoning(delta: Any) -> str | None:
    # Not part of the OpenAI schema; Ollama sends `reasoning`, vLLM/DeepSeek `reasoning_content`.
    extra = getattr(delta, "model_extra", None) or {}
    value = extra.get("reasoning") or extra.get("reasoning_content")
    return value if isinstance(value, str) and value else None


def _usage(usage: Any) -> Usage:
    details = getattr(usage, "prompt_tokens_details", None)
    cached = (getattr(details, "cached_tokens", None) or 0) if details else 0
    prompt = usage.prompt_tokens or 0
    return Usage(
        input_tokens=prompt - cached,
        output_tokens=usage.completion_tokens or 0,
        cache_read_tokens=cached,
    )


def map_error(e: Exception, base_url: str | None) -> ProviderError:
    message = getattr(e, "message", None) or str(e)
    if isinstance(e, openai.RateLimitError):
        return ProviderError("rate_limit", message, _retry_after(e))
    if isinstance(e, openai.InternalServerError):
        return ProviderError("overloaded", message, _retry_after(e))
    if isinstance(e, openai.AuthenticationError | openai.PermissionDeniedError | LookupError):
        return ProviderError("auth", message)
    if isinstance(e, openai.BadRequestError) and "context" in message.lower():
        return ProviderError("context_too_long", message)
    if isinstance(e, openai.APIConnectionError):
        where = base_url or "the OpenAI API"
        return ProviderError("other", f"cannot connect to {where}: {message}")
    return ProviderError("other", message)


def _retry_after(e: Exception) -> float | None:
    response = getattr(e, "response", None)
    value = response.headers.get("retry-after") if response is not None else None
    try:
        return float(value) if value else None
    except ValueError:
        return None
