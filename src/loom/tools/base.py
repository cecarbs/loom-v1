"""Tool protocol, ToolResult, ToolContext, and ToolSet (dispatch + budget + logging)."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ValidationError

from loom.core.events import EventBus, ToolCallFinished, ToolCallStarted
from loom.core.messages import ImageBlock, TextBlock, ToolResultBlock, ToolUseBlock
from loom.providers.base import INVALID_JSON_KEY, ToolSchema
from loom.session.store import SessionLog
from loom.tools.budget import OutputBudget
from loom.tools.paths import PathPolicy, ReadSet

log = logging.getLogger(__name__)

PREVIEW_CHARS = 2000


@dataclass
class ToolResult:
    text: str
    is_error: bool = False
    images: list[ImageBlock] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)  # diff, exit_code, spill path, …


@dataclass
class ToolContext:
    workspace: Path
    paths: PathPolicy
    read_set: ReadSet
    bus: EventBus
    call_id: str = ""
    model_accepts_images: bool = True


class Tool(Protocol):
    name: str
    description: str

    @property
    def input_model(self) -> type[BaseModel]: ...  # read-only, so subclasses may narrow it

    async def run(self, args: Any, ctx: ToolContext) -> ToolResult: ...


# Consecutive calls to these tools may run concurrently.
CONCURRENT_TOOLS = frozenset({"read"})


class ToolSet:
    def __init__(
        self, tools: Sequence[Tool], budget: OutputBudget, session_log: SessionLog, bus: EventBus
    ) -> None:
        self._tools = {t.name: t for t in tools}
        self._budget = budget
        self._log = session_log
        self._bus = bus

    @property
    def budget(self) -> OutputBudget:
        return self._budget

    def names(self) -> list[str]:
        return list(self._tools)

    def schemas(self) -> list[ToolSchema]:
        return [
            ToolSchema(t.name, t.description, t.input_model.model_json_schema())
            for t in self._tools.values()
        ]

    async def run_all(
        self, calls: Sequence[ToolUseBlock], ctx: ToolContext, results: list[ToolResultBlock]
    ) -> None:
        """Run calls in order, appending to `results` as each finishes (so a cancelled batch
        keeps its completed results). Consecutive concurrent-safe calls run together."""
        i = 0
        while i < len(calls):
            j = i + 1
            if calls[i].name in CONCURRENT_TOOLS:
                while j < len(calls) and calls[j].name in CONCURRENT_TOOLS:
                    j += 1
            group = calls[i:j]
            if len(group) == 1:
                results.append(await self.run_one(group[0], ctx))
            else:
                results.extend(await asyncio.gather(*(self.run_one(c, ctx) for c in group)))
            i = j

    async def run_one(self, call: ToolUseBlock, ctx: ToolContext) -> ToolResultBlock:
        await self._bus.emit(ToolCallStarted(call.id, call.name, call.input))
        self._log.append("tool_call", id=call.id, name=call.name, args=call.input)
        start = time.monotonic()

        result = await self._execute(call, replace(ctx, call_id=call.id))
        text, spill = self._budget.apply(result.text)
        if spill:
            result.meta["spill_path"] = str(spill)
        duration_ms = int((time.monotonic() - start) * 1000)

        self._log.append(
            "tool_result",
            id=call.id,
            name=call.name,
            result=text,
            spill_path=str(spill) if spill else None,
            is_error=result.is_error,
            duration_ms=duration_ms,
        )
        await self._bus.emit(
            ToolCallFinished(
                call.id, text[:PREVIEW_CHARS], result.is_error, duration_ms, result.meta
            )
        )
        content: str | list[TextBlock | ImageBlock] = text
        if result.images:
            content = [TextBlock(text), *result.images]
        return ToolResultBlock(call.id, content, result.is_error)

    async def _execute(self, call: ToolUseBlock, ctx: ToolContext) -> ToolResult:
        tool = self._tools.get(call.name)
        if tool is None:
            return ToolResult(
                f"Unknown tool {call.name!r}. Available tools: {', '.join(self._tools)}.", True
            )
        if INVALID_JSON_KEY in call.input:
            return ToolResult(
                "The tool call arguments were not valid JSON. Resend the call with a valid JSON "
                "object.",
                True,
            )
        try:
            args = tool.input_model.model_validate(call.input)
        except ValidationError as e:
            problems = "\n".join(
                f"  {'.'.join(str(p) for p in err['loc']) or '<args>'}: {err['msg']}"
                for err in e.errors()
            )
            return ToolResult(f"Invalid arguments for {call.name}:\n{problems}", True)
        try:
            return await tool.run(args, ctx)
        except Exception as e:  # a tool bug must not end the turn
            log.exception("tool %s failed", call.name)
            return ToolResult(f"{call.name} failed: {type(e).__name__}: {e}", True)
