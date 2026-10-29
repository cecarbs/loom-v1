"""Scripted provider for tests: each `stream()` call plays the next scripted response."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field

from loom.core.events import Usage
from loom.core.messages import Block, Message, TextBlock, ToolUseBlock
from loom.providers.base import (
    MessageEnd,
    ModelRequest,
    ProviderError,
    ProviderEvent,
    TextDelta,
    ToolUseEnd,
    ToolUseInputDelta,
    ToolUseStart,
    estimate_tokens,
)


@dataclass
class FakeResponse:
    blocks: Sequence[Block]
    stop_reason: str | None = None  # default: tool_use if any tool calls, else end_turn
    usage: Usage = field(default_factory=lambda: Usage(input_tokens=100, output_tokens=20))
    chunk_delay: float = 0.0
    hang_after_text: bool = False  # stream text, then block forever (for cancellation tests)


Script = FakeResponse | Sequence[Block] | ProviderError


class FakeProvider:
    name = "fake"

    def __init__(self, script: Sequence[Script]) -> None:
        self._script = list(script)
        self.requests: list[ModelRequest] = []

    async def stream(self, req: ModelRequest) -> AsyncIterator[ProviderEvent]:
        self.requests.append(req)
        if not self._script:
            raise AssertionError("FakeProvider script exhausted")
        item = self._script.pop(0)
        if isinstance(item, ProviderError):
            yield item
            return
        resp = item if isinstance(item, FakeResponse) else FakeResponse(list(item))

        for block in resp.blocks:
            if isinstance(block, TextBlock):
                for word in _chunks(block.text):
                    await asyncio.sleep(resp.chunk_delay)
                    yield TextDelta(word)
                if resp.hang_after_text:
                    await asyncio.Event().wait()
            elif isinstance(block, ToolUseBlock):
                yield ToolUseStart(block.id, block.name)
                yield ToolUseInputDelta(block.id, json.dumps(block.input))
                yield ToolUseEnd(block.id, block.input)

        has_tools = any(isinstance(b, ToolUseBlock) for b in resp.blocks)
        stop = resp.stop_reason or ("tool_use" if has_tools else "end_turn")
        yield MessageEnd(stop, resp.usage, Message("assistant", list(resp.blocks)))

    def count_tokens_estimate(self, messages: list[Message]) -> int:
        return estimate_tokens(messages)


def _chunks(text: str) -> list[str]:
    words = text.split(" ")
    return [w + (" " if i < len(words) - 1 else "") for i, w in enumerate(words)]
