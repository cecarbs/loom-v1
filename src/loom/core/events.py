"""AgentEvents and the in-process event bus. The TUI subscribes; the core never imports it."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from loom.core.messages import Message

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.cache_read_tokens + other.cache_read_tokens,
            self.cache_write_tokens + other.cache_write_tokens,
        )

    @property
    def prompt_tokens(self) -> int:
        """Everything the model read: uncached + cache reads + cache writes."""
        return self.input_tokens + self.cache_read_tokens + self.cache_write_tokens


@dataclass(frozen=True)
class TurnStarted:
    pass


@dataclass(frozen=True)
class TextDelta:
    text: str


@dataclass(frozen=True)
class ThinkingDelta:
    text: str


@dataclass(frozen=True)
class AssistantMessageDone:
    message: Message


@dataclass(frozen=True)
class ToolCallStarted:
    id: str
    name: str
    input: dict[str, Any]


@dataclass(frozen=True)
class ToolCallOutput:
    id: str
    chunk: str


@dataclass(frozen=True)
class ToolCallFinished:
    id: str
    result_preview: str
    is_error: bool
    duration_ms: int
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ApprovalRequested:
    """The only event that expects a reply: answer with `EventBus.respond(request_id, ...)`."""

    request_id: str
    tool: str
    args_preview: str
    effect: str  # "read" | "write" | ...
    agent_id: str | None = None  # set for requests from sub-agents


@dataclass(frozen=True)
class Decision:
    action: Literal["allow_once", "allow_session", "deny"]
    message: str | None = None  # "deny with message": returned to the model as the error text
    by: Literal["user", "policy", "override", "cancel"] = "user"

    @property
    def allowed(self) -> bool:
        return self.action != "deny"


@dataclass(frozen=True)
class Notice:
    text: str


@dataclass(frozen=True)
class ErrorEvent:
    text: str


@dataclass(frozen=True)
class UsageUpdated:
    input_tokens: int
    output_tokens: int
    cache_read: int
    context_pct: float
    cost_usd: float | None  # session total; None when the model has no pricing


@dataclass(frozen=True)
class TurnCancelled:
    pass


@dataclass(frozen=True)
class TurnFinished:
    usage: Usage


AgentEvent = (
    TurnStarted
    | TextDelta
    | ThinkingDelta
    | AssistantMessageDone
    | ToolCallStarted
    | ToolCallOutput
    | ToolCallFinished
    | ApprovalRequested
    | Notice
    | ErrorEvent
    | UsageUpdated
    | TurnCancelled
    | TurnFinished
)

Subscriber = Callable[[AgentEvent], Awaitable[None]]


class EventBus:
    """Subscribers are awaited in order. A failing subscriber is logged, never fatal."""

    def __init__(self) -> None:
        self._subscribers: list[Subscriber] = []
        self._pending: dict[str, asyncio.Future[Decision]] = {}

    def subscribe(self, handler: Subscriber) -> Callable[[], None]:
        self._subscribers.append(handler)
        return lambda: self._subscribers.remove(handler)

    async def emit(self, event: AgentEvent) -> None:
        for handler in list(self._subscribers):
            try:
                await handler(event)
            except Exception:
                log.exception("event subscriber failed on %s", type(event).__name__)

    async def request_approval(self, event: ApprovalRequested) -> Decision:
        """Emit `event` and wait for a `respond()`. Never resolves to allow on its own."""
        future: asyncio.Future[Decision] = asyncio.get_running_loop().create_future()
        self._pending[event.request_id] = future
        try:
            await self.emit(event)
            return await future
        finally:
            self._pending.pop(event.request_id, None)

    def respond(self, request_id: str, decision: Decision) -> None:
        future = self._pending.get(request_id)
        if future is not None and not future.done():
            future.set_result(decision)
