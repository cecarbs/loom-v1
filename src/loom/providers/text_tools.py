"""Text tool-call fallback for models without native tool calling (docs/02).

Wraps any adapter: tools are described in the system prompt, the model writes
`<tool_call>{"name": …, "input": {…}}</tool_call>`, and the wrapper turns those into ordinary
tool-use events. History stays canonical (real ToolUseBlock / ToolResultBlock); only the request
is rewritten, so switching to a native-tools model mid-session works.
"""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import AsyncIterator
from dataclasses import replace
from typing import Any

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
    Provider,
    ProviderError,
    ProviderEvent,
    TextDelta,
    ThinkingDelta,
    ToolSchema,
    ToolUseEnd,
    ToolUseInputDelta,
    ToolUseStart,
)

OPEN, CLOSE = "<tool_call>", "</tool_call>"
RESULT = "<tool_result"  # a model inventing its own results: drop the rest of the reply
INVALID_NAME = "invalid_tool_call"
_NAME_RE = re.compile(r'"name"\s*:\s*"([^"]+)"')
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$")


def tools_prompt(tools: list[ToolSchema]) -> str:
    lines = [
        "# Calling tools",
        "You call a tool by writing a tool call block in your reply, exactly like this:",
        f'{OPEN}{{"name": "<tool name>", "input": {{<arguments as JSON>}}}}{CLOSE}',
        "The content must be one valid JSON object. You may write several tool calls in one "
        "reply. After your tool calls, stop and wait: the results come back in the next "
        'message as <tool_result id="…">…</tool_result>. Never write tool results yourself.',
        "",
        "Available tools:",
    ]
    for tool in tools:
        lines += [
            f"## {tool.name}",
            tool.description,
            f"Input JSON schema: {json.dumps(tool.input_schema)}",
            "",
        ]
    return "\n".join(lines).strip()


class TextToolCallProvider:
    """A provider that adds text tool calls on top of `inner`, which gets no native tools."""

    def __init__(self, inner: Provider) -> None:
        self.inner = inner
        self.name = inner.name

    async def stream(self, req: ModelRequest) -> AsyncIterator[ProviderEvent]:
        parser = _Parser()
        thinking: list[ThinkingBlock] = []
        async for event in self.inner.stream(to_text_request(req)):
            match event:
                case TextDelta(chunk):
                    if text := parser.feed(chunk):
                        yield TextDelta(text)
                case ThinkingDelta() | ProviderError():
                    yield event
                case MessageEnd(stop_reason, usage, final):
                    if text := parser.finish():
                        yield TextDelta(text)
                    thinking = [b for b in final.content if isinstance(b, ThinkingBlock)]
                    content: list[Block] = [*thinking]
                    if joined := "".join(parser.text).strip():
                        content.append(TextBlock(joined))
                    for raw in parser.calls:
                        call = parse_call(raw)
                        content.append(call)
                        yield ToolUseStart(call.id, call.name)
                        yield ToolUseInputDelta(call.id, raw)
                        yield ToolUseEnd(call.id, call.input)
                    if parser.calls and stop_reason != "max_tokens":
                        stop_reason = "tool_use"
                    meta = {**final.meta, "stop_reason": stop_reason}
                    yield MessageEnd(stop_reason, usage, Message("assistant", content, meta))
                case _:
                    pass  # the inner adapter got no tools, so it emits no tool events

    def count_tokens_estimate(self, messages: list[Message]) -> int:
        return self.inner.count_tokens_estimate(messages)


def to_text_request(req: ModelRequest) -> ModelRequest:
    system = req.system
    if req.tools:
        system = f"{system}\n\n{tools_prompt(req.tools)}"
    messages = [Message(m.role, _as_text(m.content), m.meta) for m in req.messages]
    return replace(req, system=system, messages=messages, tools=[])


def _as_text(blocks: list[Block]) -> list[Block]:
    out: list[Block] = []
    for block in blocks:
        if isinstance(block, ToolUseBlock):
            if INVALID_JSON_KEY in block.input:
                body = str(block.input[INVALID_JSON_KEY])
            else:
                body = json.dumps({"name": block.name, "input": block.input})
            out.append(TextBlock(f"{OPEN}{body}{CLOSE}"))
        elif isinstance(block, ToolResultBlock):
            images: list[ImageBlock] = []
            if isinstance(block.content, str):
                text = block.content
            else:
                text = "".join(b.text for b in block.content if isinstance(b, TextBlock))
                images = [b for b in block.content if isinstance(b, ImageBlock)]
            attrs = f'id="{block.tool_use_id}"' + (' error="true"' if block.is_error else "")
            out.append(TextBlock(f"<tool_result {attrs}>{text}</tool_result>"))
            out.extend(images)
        else:
            out.append(block)
    return out


def parse_call(raw: str) -> ToolUseBlock:
    call_id = f"call_{uuid.uuid4().hex[:12]}"
    body = _FENCE_RE.sub("", raw.strip())
    try:
        data: Any = json.loads(body)
    except json.JSONDecodeError:
        data = None
    if isinstance(data, dict) and isinstance(data.get("name"), str):
        args = data.get("input", data.get("arguments", {}))
        if isinstance(args, dict):
            return ToolUseBlock(call_id, data["name"], args)
    match = _NAME_RE.search(body)
    return ToolUseBlock(call_id, match.group(1) if match else INVALID_NAME, {INVALID_JSON_KEY: raw})


class _Parser:
    """Splits streamed text into plain text (streamed on) and tool call bodies (buffered)."""

    def __init__(self) -> None:
        self.text: list[str] = []
        self.calls: list[str] = []
        self._buf = ""
        self._inside = False
        self._done = False

    def feed(self, chunk: str) -> str:
        self._buf += chunk
        out: list[str] = []
        while self._buf and not self._done:
            if self._inside:
                end = self._buf.find(CLOSE)
                if end < 0:
                    break
                self.calls.append(self._buf[:end])
                self._buf = self._buf[end + len(CLOSE) :]
                self._inside = False
                continue
            starts = [(i, tag) for tag in (OPEN, RESULT) if (i := self._buf.find(tag)) >= 0]
            if starts:
                start, tag = min(starts)
                out.append(self._buf[:start])
                self._buf = self._buf[start + len(tag) :]
                if tag == RESULT:
                    self._done = True
                else:
                    self._inside = True
                continue
            keep = _partial_tag_len(self._buf)
            out.append(self._buf[: len(self._buf) - keep])
            self._buf = self._buf[len(self._buf) - keep :]
            break
        if self._done:
            self._buf = ""
        return self._emit(out)

    def finish(self) -> str:
        """End of stream: an unclosed call is kept (as invalid JSON); held-back text flushes."""
        rest, self._buf = self._buf, ""
        if self._inside:
            self.calls.append(rest)
            return ""
        return self._emit([rest])

    def _emit(self, parts: list[str]) -> str:
        text = "".join(parts)
        if text:
            self.text.append(text)
        return text


def _partial_tag_len(buf: str) -> int:
    """Length of the longest suffix of `buf` that could start a tag."""
    for n in range(min(len(buf), len(OPEN)), 0, -1):
        tail = buf[-n:]
        if OPEN.startswith(tail) or RESULT.startswith(tail):
            return n
    return 0
