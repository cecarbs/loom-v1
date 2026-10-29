"""The agent loop: one user turn = model calls and tool batches until the model stops."""

from __future__ import annotations

import asyncio
import dataclasses
import time
from collections.abc import Sequence

from loom.config import ContextConfig
from loom.core import compaction
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
from loom.session.store import ResumedState, SessionLog, first_line
from loom.tools.base import ToolContext, ToolSet

MAX_TRIES = 4  # backoff 1s, 2s, 4s between tries unless the provider sends retry_after
TRUNCATED_TOOL_INPUT = (
    "Not run: your response hit max_tokens, so this tool call's input may be truncated. "
    "Retry with a smaller input (e.g. several smaller edits)."
)
RESUMED_NOTE = (
    "[loom: this session was resumed in a new process. The kernel is fresh: variables from "
    "earlier execute calls are gone.]"
)


class _Overflow:
    """`_stream_model` result: the provider said the prompt is too long."""


OVERFLOW = _Overflow()


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
        context_config: ContextConfig | None = None,
    ) -> None:
        self.provider = provider
        self.profile = profile
        self.tools = tools
        self.tool_ctx = tool_ctx
        self.context = context
        self.bus = bus
        self.log = session_log
        self.max_steps = max_steps
        self.context_config = context_config or ContextConfig()

        self.history: list[Message] = []
        self.turns = 0
        self.usage = Usage()
        self.cost_usd: float | None = 0.0 if profile.pricing else None
        # Told to the model with the next prompt (resume, /undo); kept in history and log.
        self.pending_notes: list[str] = []
        # (prompt tokens the provider reported, messages in that request): the base for
        # estimating the context size before the next request.
        self._last_prompt: tuple[int, int] | None = None
        self._compacted_len = 0  # history length right after the last compaction
        # Turn state used to repair history on cancellation.
        self._streaming: tuple[list[str], list[ToolUseBlock]] | None = None
        self._running: tuple[Sequence[ToolUseBlock], list[ToolResultBlock]] | None = None

    def switch_model(self, profile: ModelProfile, provider: pv.Provider) -> None:
        """Use another model from the next request on. History is kept; each adapter drops
        what its API can't take (e.g. another provider's thinking blocks)."""
        self.log.append(
            "model_switch", old=self.profile.alias, new=profile.alias, provider=profile.provider
        )
        self.profile = profile
        self.provider = provider
        self.context.profile = profile
        self.tool_ctx.model_accepts_images = profile.capabilities.images
        if profile.pricing is not None and self.cost_usd is None:
            self.cost_usd = 0.0  # requests to unpriced models aren't counted

    def restore(self, state: ResumedState) -> None:
        """Continue a resumed session's conversation (docs/07 "Resume")."""
        self.history = list(state.history)
        self.turns = state.turns
        self.usage = state.usage
        if state.cost_usd is not None:
            self.cost_usd = state.cost_usd
        if state.last_prompt_tokens:
            self._last_prompt = (state.last_prompt_tokens, len(self.history))
        self.pending_notes = list(state.notes)
        if state.used_kernel:
            self.pending_notes.append(RESUMED_NOTE)

    def add_note(self, text: str) -> None:
        """Tell the model `text` with the next prompt (logged, so it survives a resume)."""
        self.pending_notes.append(text)
        self.log.append("note", text=text)

    @property
    def context_pct(self) -> float:
        return self.context_tokens() / self.profile.context_window

    def context_tokens(self) -> int:
        """Provider-reported prompt tokens of the last request plus an estimate of what was
        added since; an estimate of everything when there's no report (yet)."""
        if self._last_prompt is not None and self._last_prompt[0] > 0:
            tokens, count = self._last_prompt
            return tokens + pv.estimate_tokens(self.history[count:])
        return pv.estimate_tokens(self.history) + len(self.context.system) // 4

    async def run_turn(self, user_input: str, attachments: Sequence[Block] = ()) -> None:
        try:
            notes = [TextBlock(n) for n in self.pending_notes]
            self.pending_notes = []
            self._append(Message("user", [TextBlock(user_input), *attachments, *notes]))
            self.turns += 1
            if self.tool_ctx.checkpoints is not None:
                self.tool_ctx.checkpoints.begin_turn(self.turns, first_line(user_input))
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
            await self._maybe_compact()
            request = self.context.build_request(self.history, self.tools.schemas())
            message = await self._stream_model(request, on_overflow="signal")
            if message is OVERFLOW:  # compact now, then one retry (docs/02 "Retries")
                if not await self.compact(mid_turn=True):
                    await self.bus.emit(
                        ev.ErrorEvent("The conversation is too long for the model.")
                    )
                    return
                request = self.context.build_request(self.history, self.tools.schemas())
                message = await self._stream_model(request)
            if not isinstance(message, Message):
                return
            if not message.content:
                # Nothing to show or replay (an empty assistant message is also invalid for
                # some APIs). Seen with small local models that only produce reasoning.
                await self.bus.emit(ev.Notice("The model returned an empty response."))
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

    async def _maybe_compact(self) -> None:
        threshold = self.context_config.compact_at * self.profile.context_window
        if len(self.history) > self._compacted_len and self.context_tokens() > threshold:
            await self.compact(mid_turn=True)

    async def compact(self, instructions: str = "", *, mid_turn: bool = False) -> bool:
        """Replace older history with a summary plus the kernel inventory (docs/07). Returns
        False if there was nothing to compact or the summary failed; history is then
        unchanged (also when cancelled)."""
        keep = self.context_config.keep_recent_turns
        cut = compaction.cut_point(self.history, keep, mid_turn=mid_turn)
        if cut is None:
            await self.bus.emit(ev.Notice("Nothing to compact yet."))
            return False
        await self.bus.emit(ev.Notice("Compacting the conversation…"))
        before = self.context_tokens()
        messages = compaction.summary_request(self.history[:cut], instructions)
        request = self.context.build_request(messages, self.tools.schemas())
        reply = await self._stream_model(request, purpose="compaction")
        summary = reply.text().strip() if isinstance(reply, Message) else ""
        if not summary:
            await self.bus.emit(ev.ErrorEvent("Compaction failed: the model gave no summary."))
            return False
        kernel = self.tool_ctx.kernel
        variables = await kernel.inventory() if kernel is not None else []
        message = compaction.summary_message(summary, variables)
        self.history = [message, *self.history[cut:]]
        self._compacted_len = len(self.history)
        self._last_prompt = None
        after = self.context_tokens()
        self.log.append(
            "compaction",
            summary=summary,
            message=message_to_dict(message),
            tokens_before=before,
            tokens_after=after,
            first_kept=cut,
            instructions=instructions or None,
        )
        await self.bus.emit(ev.Compacted(before, after))
        await self._emit_usage()
        return True

    async def _stream_model(
        self,
        request: pv.ModelRequest,
        *,
        purpose: str = "turn",
        on_overflow: str = "error",
    ) -> Message | _Overflow | None:
        """Stream one model response. A `compaction` request isn't shown or repaired on
        cancel. With `on_overflow="signal"`, a context-too-long error returns OVERFLOW."""
        live = purpose == "turn"
        for attempt in range(MAX_TRIES):
            text: list[str] = []
            tool_uses: list[ToolUseBlock] = []
            names: dict[str, str] = {}
            if live:
                self._streaming = (text, tool_uses)
            final: pv.MessageEnd | None = None
            error: pv.ProviderError | None = None
            start = time.monotonic()

            async for event in self.provider.stream(request):
                match event:
                    case pv.TextDelta(chunk):
                        text.append(chunk)
                        if live:
                            await self.bus.emit(ev.TextDelta(chunk))
                    case pv.ThinkingDelta(chunk):
                        if live:
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
                cost = self._account(final.usage)
                self.log.append(
                    "model_request",
                    model=request.model,
                    purpose=purpose,
                    settings=_settings(request),
                    usage=dataclasses.asdict(final.usage),
                    cost_usd=cost,
                    latency_ms=latency_ms,
                    stop_reason=final.stop_reason,
                )
                if live:
                    self._last_prompt = (final.usage.prompt_tokens, len(request.messages))
                await self._emit_usage()
                return message

            error = error or pv.ProviderError("other", "model stream ended without a response")
            self.log.append("error", text=f"{error.kind}: {error.message}", attempt=attempt + 1)
            if error.kind == "context_too_long" and on_overflow == "signal":
                return OVERFLOW
            if error.kind in pv.RETRYABLE and attempt < MAX_TRIES - 1:
                delay = error.retry_after if error.retry_after is not None else 2.0**attempt
                await self.bus.emit(ev.Notice(f"Model {error.kind}; retrying in {delay:.0f}s."))
                await asyncio.sleep(delay)
                continue
            await self.bus.emit(ev.ErrorEvent(f"Model error ({error.kind}): {error.message}"))
            return None
        return None

    def _account(self, usage: Usage) -> float | None:
        """Add a request's usage to the session totals; returns its cost if priced."""
        self.usage = self.usage + usage
        pricing = self.profile.pricing
        if pricing is None:
            return None
        cost = pricing.cost(
            usage.input_tokens,
            usage.output_tokens,
            usage.cache_read_tokens,
            usage.cache_write_tokens,
        )
        if self.cost_usd is not None:
            self.cost_usd += cost
        return cost

    async def _emit_usage(self) -> None:
        await self.bus.emit(
            ev.UsageUpdated(
                input_tokens=self.usage.input_tokens,
                output_tokens=self.usage.output_tokens,
                cache_read=self.usage.cache_read_tokens,
                context_pct=self.context_pct,
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


def _settings(request: pv.ModelRequest) -> dict[str, object]:
    """What was asked for, beyond the (reconstructible) prompt: makes the log checkable."""
    thinking = request.thinking
    return {
        "max_tokens": request.max_tokens,
        "thinking": {k: v for k, v in dataclasses.asdict(thinking).items() if v is not None}
        if thinking
        else None,
        "temperature": request.temperature,
        "cache": request.cache,
        "tools": len(request.tools),
        "messages": len(request.messages),
    }
