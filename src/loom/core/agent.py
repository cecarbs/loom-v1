"""The agent loop: one user turn = model calls and tool batches until the model stops."""

from __future__ import annotations

import asyncio
import dataclasses
import time
from collections.abc import Sequence

from loom.core import events as ev
from loom.core.context import ContextBuilder
from loom.core.events import EventBus, Usage
from loom.core.messages import (
    Block,
    Message,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    interrupted_result,
    message_to_dict,
)
from loom.providers import base as pv
from loom.providers.profiles import ModelProfile
from loom.session.store import SessionLog
from loom.tools.base import ToolContext, ToolSet

MAX_TRIES = 4  # backoff 1s, 2s, 4s between tries unless the provider sends retry_after
TRUNCATED_TOOL_INPUT = (
    "Not run: your response hit max_tokens, so this tool call's input may be truncated. "
    "Retry with a smaller input (e.g. several smaller edits)."
)


class AgentLoop:
    def __init__(
        self,
        *,
        provider: pv.Provider,
        profile: ModelProfile,
        tools: ToolSet,
        tool_ctx: ToolContext,
        context: ContextBuilder,
        bus: EventBus,
        session_log: SessionLog,
        max_steps: int = 50,
    ) -> None:
        self.provider = provider
        self.profile = profile
        self.tools = tools
        self.tool_ctx = tool_ctx
        self.context = context
        self.bus = bus
        self.log = session_log
        self.max_steps = max_steps

        self.history: list[Message] = []
        self.usage = Usage()
        self.cost_usd: float | None = 0.0 if profile.pricing else None
        # Turn state used to repair history on cancellation.
        self._streaming: tuple[list[str], list[ToolUseBlock]] | None = None
        self._running: tuple[Sequence[ToolUseBlock], list[ToolResultBlock]] | None = None

    async def run_turn(self, user_input: str, attachments: Sequence[Block] = ()) -> None:
        try:
            self._append(Message("user", [TextBlock(user_input), *attachments]))
            await self.bus.emit(ev.TurnStarted())
            await self._steps()
        except asyncio.CancelledError:
            task = asyncio.current_task()
            if task is not None:
                task.uncancel()
            self._repair_after_cancel()
            self.log.append("turn_cancelled")
            await self.bus.emit(ev.TurnCancelled())
            return
        await self.bus.emit(ev.TurnFinished(self.usage))

    async def _steps(self) -> None:
        for _ in range(self.max_steps):
            request = self.context.build_request(self.history, self.tools.schemas())
            message = await self._stream_model(request)
            if message is None:
                return
            self._append(message)
            await self.bus.emit(ev.AssistantMessageDone(message))

            calls = message.tool_uses()
            if not calls:
                return
            if message.meta.get("stop_reason") == "max_tokens":
                truncated = [ToolResultBlock(c.id, TRUNCATED_TOOL_INPUT, True) for c in calls]
                self._append(Message("user", list(truncated)))
                continue

            results: list[ToolResultBlock] = []
            self._running = (calls, results)
            await self.tools.run_all(calls, self.tool_ctx, results)
            self._running = None
            self._append(Message("user", list(results)))
        await self.bus.emit(ev.Notice("Step limit reached; stopping."))

    async def _stream_model(self, request: pv.ModelRequest) -> Message | None:
        for attempt in range(MAX_TRIES):
            text: list[str] = []
            tool_uses: list[ToolUseBlock] = []
            names: dict[str, str] = {}
            self._streaming = (text, tool_uses)
            final: pv.MessageEnd | None = None
            error: pv.ProviderError | None = None
            start = time.monotonic()

            async for event in self.provider.stream(request):
                match event:
                    case pv.TextDelta(chunk):
                        text.append(chunk)
                        await self.bus.emit(ev.TextDelta(chunk))
                    case pv.ThinkingDelta(chunk):
                        await self.bus.emit(ev.ThinkingDelta(chunk))
                    case pv.ToolUseStart(id_, name):
                        names[id_] = name
                    case pv.ToolUseEnd(id_, input_):
                        tool_uses.append(ToolUseBlock(id_, names.get(id_, "?"), input_))
                    case pv.MessageEnd():
                        final = event
                    case pv.ProviderError():
                        error = event
            self._streaming = None
            latency_ms = int((time.monotonic() - start) * 1000)

            if final is not None:
                message = final.final
                message.meta.update(
                    model=request.model,
                    stop_reason=final.stop_reason,
                    usage=dataclasses.asdict(final.usage),
                )
                self.log.append(
                    "model_request",
                    model=request.model,
                    usage=dataclasses.asdict(final.usage),
                    latency_ms=latency_ms,
                    stop_reason=final.stop_reason,
                )
                await self._account(final.usage)
                return message

            error = error or pv.ProviderError("other", "model stream ended without a response")
            self.log.append("error", text=f"{error.kind}: {error.message}", attempt=attempt + 1)
            if error.kind in pv.RETRYABLE and attempt < MAX_TRIES - 1:
                delay = error.retry_after if error.retry_after is not None else 2.0**attempt
                await self.bus.emit(ev.Notice(f"Model {error.kind}; retrying in {delay:.0f}s."))
                await asyncio.sleep(delay)
                continue
            await self.bus.emit(ev.ErrorEvent(f"Model error ({error.kind}): {error.message}"))
            return None
        return None

    async def _account(self, usage: Usage) -> None:
        self.usage = self.usage + usage
        pricing = self.profile.pricing
        if pricing is not None and self.cost_usd is not None:
            self.cost_usd += pricing.cost(
                usage.input_tokens,
                usage.output_tokens,
                usage.cache_read_tokens,
                usage.cache_write_tokens,
            )
        await self.bus.emit(
            ev.UsageUpdated(
                input_tokens=self.usage.input_tokens,
                output_tokens=self.usage.output_tokens,
                cache_read=self.usage.cache_read_tokens,
                context_pct=usage.prompt_tokens / self.profile.context_window,
                cost_usd=self.cost_usd,
            )
        )

    def _append(self, message: Message) -> None:
        self.history.append(message)
        self.log.append("message", message=message_to_dict(message))

    def _repair_after_cancel(self) -> None:
        """Keep tool_use/tool_result pairing valid after a cancel (docs/02 "Cancellation")."""
        if self._streaming is not None:
            text, tool_uses = self._streaming
            content: list[Block] = []
            if joined := "".join(text):
                content.append(TextBlock(joined))
            content.extend(tool_uses)  # only complete tool uses were collected
            if content:
                self._append(Message("assistant", content, {"interrupted": True}))
            self._streaming = None
        if self._running is not None:
            calls, results = self._running
            done = {r.tool_use_id for r in results}
            results.extend(interrupted_result(c.id) for c in calls if c.id not in done)
            self._append(Message("user", list(results)))
            self._running = None
        # Anything else (e.g. cancelled between appending the assistant message and running
        # its tools): answer every unanswered tool use.
        last = self.history[-1] if self.history else None
        if last is not None and last.role == "assistant" and last.tool_uses():
            self._append(Message("user", [interrupted_result(c.id) for c in last.tool_uses()]))
