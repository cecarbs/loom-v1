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

from loom.broker.approval import Access, Approvals, denial_text
from loom.broker.broker import ToolBroker
from loom.broker.redact import Redactor
from loom.core.events import EventBus, Notice, ToolCallFinished, ToolCallStarted
from loom.core.messages import ImageBlock, TextBlock, ToolResultBlock, ToolUseBlock
from loom.kernel.client import KernelClient
from loom.providers.base import INVALID_JSON_KEY, ToolSchema
from loom.sandbox.base import SandboxBackend
from loom.session.checkpoints import CheckpointError, Checkpoints
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
    sandbox: SandboxBackend | None = None
    sandbox_env: dict[str, str] = field(default_factory=dict)
    kernel: KernelClient | None = None
    broker: ToolBroker | None = None
    workspace_writable: bool = True
    checkpoints: Checkpoints | None = None


class Tool(Protocol):
    name: str
    description: str

    @property
    def input_model(self) -> type[BaseModel]: ...  # read-only, so subclasses may narrow it

    async def run(self, args: Any, ctx: ToolContext) -> ToolResult: ...


class GatedTool(Tool, Protocol):
    """A tool the approval policy checks before it runs (all built-ins). ToolSet looks the
    method up by name, so tools without it (test doubles) always run."""

    def access(self, args: Any, ctx: ToolContext) -> Access: ...


# Consecutive calls to these tools may run concurrently.
CONCURRENT_TOOLS = frozenset({"read"})
# The first of these in a turn takes a checkpoint first (docs/07 "Checkpoints and undo").
MUTATING_TOOLS = frozenset({"write", "edit", "bash", "execute"})


class ToolSet:
    def __init__(
        self,
        tools: Sequence[Tool],
        budget: OutputBudget,
        session_log: SessionLog,
        bus: EventBus,
        approvals: Approvals | None = None,
        redactor: Redactor | None = None,
    ) -> None:
        self._tools = {t.name: t for t in tools}
        self._budget = budget
        self._log = session_log
        self._bus = bus
        self.approvals = approvals
        self.redactor = redactor

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
        if self.redactor is not None:
            result.text = self.redactor.text(result.text)
        kernel_ref = result.meta.get("kernel_ref")
        text, spill = self._budget.apply(
            result.text, kernel_ref=kernel_ref if isinstance(kernel_ref, int) else None
        )
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
        if INVALID_JSON_KEY in call.input:  # first: the name may be unreadable too
            return ToolResult(
                "The tool call arguments were not valid JSON. Resend the call with a valid JSON "
                "object.",
                True,
            )
        tool = self._tools.get(call.name)
        if tool is None:
            return ToolResult(
                f"Unknown tool {call.name!r}. Available tools: {', '.join(self._tools)}.", True
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
            access = getattr(tool, "access", None)
            if self.approvals is not None and access is not None:
                approvals = self.approvals
                decision = await approvals.check(call.name, access(args, ctx), call_id=call.id)
                if not decision.allowed:
                    return ToolResult(denial_text(call.name, decision, approvals.mode), True)
            if call.name in MUTATING_TOOLS:
                await self._checkpoint(ctx)
            return await tool.run(args, ctx)
        except Exception as e:  # a tool bug must not end the turn
            log.exception("tool %s failed", call.name)
            return ToolResult(f"{call.name} failed: {type(e).__name__}: {e}", True)

    async def _checkpoint(self, ctx: ToolContext) -> None:
        """A failed snapshot doesn't block the tool; the user is told undo won't cover it."""
        if ctx.checkpoints is None or not ctx.workspace_writable:
            return
        try:
            await ctx.checkpoints.ensure()
        except (CheckpointError, OSError) as e:
            self._log.append("error", text=f"checkpoint: {e}")
            await self._bus.emit(Notice(f"Checkpoint failed ({e}); /undo won't cover this turn."))
