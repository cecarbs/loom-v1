"""Sub-agents (docs/08): headless child loops, reachable from the kernel as `agents.*`.

Each one has its own history, kernel, transcript and private event bus. Its approvals go to
the main bus, labelled with its id. Children get a broker without the `agents` package, so
the depth limit holds by construction.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from loom.broker.registry import ToolPackage
from loom.core import events as ev
from loom.core.agent import AgentLoop
from loom.core.events import EventBus
from loom.core.messages import Message, ToolResultBlock
from loom.providers.base import Provider
from loom.providers.profiles import ModelProfile
from loom.session.store import SessionLog, first_line

log = logging.getLogger(__name__)

DEFAULT_MAX_TURNS = 15
TRANSCRIPT_RE = re.compile(r"^sa(\d+)\.jsonl$")
SUBAGENT_ROLE = (
    "# Sub-agent\n"
    "You are a sub-agent working on one task for another agent, which sees only your final "
    "message. Nobody can answer questions, so make reasonable assumptions and state them. End "
    "with a concise final answer: the result and the facts needed to use it, not a narrative "
    "of your steps. Your kernel is your own (the other agent's variables aren't in it), and "
    "`agents` isn't available: sub-agents can't start sub-agents."
)


@dataclass(frozen=True)
class ChildSpec:
    """What `make_loop` needs to build one sub-agent's AgentLoop."""

    id: str
    transcript: Path
    bus: EventBus  # private: the child's events don't reach the UI
    profile: ModelProfile
    provider: Provider
    max_turns: int


ChildFactory = Callable[[ChildSpec], AgentLoop]
ModelResolver = Callable[[str | None], tuple[ModelProfile, Provider]]


class SubAgentError(Exception):
    """A sub-agent ended without a final answer."""


@dataclass(frozen=True)
class _Outcome:
    id: str
    text: str = ""
    error: str = ""  # why there's no text; includes the transcript path


class SubAgents:
    def __init__(
        self,
        *,
        make_loop: ChildFactory,
        resolve_model: ModelResolver,
        agents_dir: Path,
        bus: EventBus,
        session_log: SessionLog,
        max_concurrency: int,
    ) -> None:
        self.make_loop = make_loop
        self.resolve_model = resolve_model
        self.agents_dir = agents_dir
        self.bus = bus
        self.log = session_log
        self.max_concurrency = max_concurrency
        self.parent: AgentLoop | None = None  # set once the main loop exists
        self._slots = asyncio.Semaphore(max_concurrency)
        self._count = 0
        self._active: dict[str, tuple[AgentLoop, asyncio.Task[None]]] = {}

    def package(self) -> ToolPackage:
        """`agents.run` and `agents.map`, as the kernel sees them."""
        package = ToolPackage(
            name="agents",
            description="Delegate tasks to sub-agents with their own history and kernel.",
            internal=True,
        )

        @package.tool(effect="read", unbounded=True)
        async def run(
            ctx, task: str, model: str | None = None, max_turns: int = DEFAULT_MAX_TURNS
        ) -> str:
            """Run a sub-agent on `task` and return its final answer.

            Its only input is the task text: it has its own kernel and can't see your
            variables. It has the same tools and approval mode as you. `model` is a model
            alias (default: the current model); `max_turns` limits its model calls. Raises
            ToolError if it ends without an answer."""
            outcome = await self._start(task, model, max_turns)
            if outcome.error:
                raise SubAgentError(outcome.error)
            return outcome.text

        @package.tool(effect="read", unbounded=True)
        async def map(
            ctx,
            tasks: list[str],
            concurrency: int | None = None,
            model: str | None = None,
            max_turns: int = DEFAULT_MAX_TURNS,
        ) -> list[str]:
            """Run one sub-agent per task, concurrently, and return their answers in order.

            `concurrency` can only lower the `[agents] max_concurrency` limit. A sub-agent
            that fails doesn't stop the others: its entry is "[sub-agent <id> failed: ...]"."""
            return await self._map(tasks, concurrency, model, max_turns)

        return package

    async def aclose(self) -> None:
        """Stop whatever a cancelled turn didn't get to clean up."""
        for agent_id, (loop, turn) in list(self._active.items()):
            turn.cancel()
            await asyncio.wait({turn})
            await _stop_kernel(loop)
            self._active.pop(agent_id, None)

    async def _map(
        self, tasks: Sequence[str], concurrency: int | None, model: str | None, max_turns: int
    ) -> list[str]:
        if concurrency is not None and concurrency < 1:
            raise ValueError("concurrency must be at least 1")
        _check_turns(max_turns)
        profile, provider = self.resolve_model(model)  # a bad alias fails before anything runs
        limit = asyncio.Semaphore(concurrency or self.max_concurrency)

        async def one(task: str) -> str:
            async with limit:
                outcome = await self._run(task, profile, provider, max_turns)
            return outcome.text or f"[sub-agent {outcome.id} failed: {outcome.error}]"

        # A TaskGroup, not gather: cancelling gather returns at the first cancelled child, with
        # the others' kernels still stopping.
        async with asyncio.TaskGroup() as group:
            running = [group.create_task(one(t)) for t in tasks]
        return [r.result() for r in running]

    async def _start(self, task: str, model: str | None, max_turns: int) -> _Outcome:
        _check_turns(max_turns)
        profile, provider = self.resolve_model(model)
        return await self._run(task, profile, provider, max_turns)

    async def _run(
        self, task: str, profile: ModelProfile, provider: Provider, max_turns: int
    ) -> _Outcome:
        async with self._slots:
            agent_id = self._new_id()
            transcript = self.agents_dir / f"{agent_id}.jsonl"
            errors: list[str] = []
            spec = ChildSpec(
                agent_id, transcript, self._child_bus(agent_id, errors), profile, provider,
                max_turns,
            )  # fmt: skip
            await self.bus.emit(ev.Notice(f"Sub-agent {agent_id} started: {first_line(task, 60)}"))
            start = time.monotonic()
            try:
                loop = self.make_loop(spec)
            except Exception as e:
                log.exception("sub-agent %s failed to start", agent_id)
                return await self._failed(agent_id, f"{type(e).__name__}: {e}", transcript)
            turn = asyncio.create_task(loop.run_turn(task), name=f"subagent-{agent_id}")
            self._active[agent_id] = (loop, turn)
            status = "cancelled"
            outcome = _Outcome(agent_id, error="cancelled")
            try:
                # Shielded: run_turn absorbs a cancel and returns normally, which would hide
                # it from this task.
                await asyncio.shield(turn)
                outcome = _outcome(agent_id, loop, errors, max_turns, transcript)
                status = "failed" if outcome.error else "done"
            except asyncio.CancelledError:
                turn.cancel()
                await asyncio.wait({turn})
                raise
            except Exception as e:  # a bug in the child must not take the parent's turn down
                log.exception("sub-agent %s crashed", agent_id)
                outcome = _Outcome(agent_id, error=f"{type(e).__name__}: {e} ({transcript})")
                status = "failed"
            finally:
                await self._record(loop, spec, status, start)
                await _stop_kernel(loop)
                self._active.pop(agent_id, None)
            await self.bus.emit(ev.Notice(_summary(outcome, start)))
            return outcome

    def _new_id(self) -> str:
        """`sa1`, `sa2`, … A resumed session continues after its existing transcripts."""
        self.agents_dir.mkdir(parents=True, exist_ok=True)
        if not self._count:
            found = (TRANSCRIPT_RE.match(p.name) for p in self.agents_dir.iterdir())
            self._count = max((int(m[1]) for m in found if m), default=0)
        self._count += 1
        return f"sa{self._count}"

    def _child_bus(self, agent_id: str, errors: list[str]) -> EventBus:
        """The child's events stop here, except for what the user should hear about."""
        bus = EventBus()

        async def forward(event: ev.AgentEvent) -> None:
            match event:
                case ev.Notice(text):
                    await self.bus.emit(ev.Notice(f"[{agent_id}] {text}"))
                case ev.ErrorEvent(text):
                    errors.append(text)

        bus.subscribe(forward)
        return bus

    async def _failed(self, agent_id: str, reason: str, transcript: Path) -> _Outcome:
        outcome = _Outcome(agent_id, error=reason)
        self.log.append("subagent", id=agent_id, status="failed", error=reason, usage={})
        await self.bus.emit(ev.Notice(f"Sub-agent {agent_id} failed: {reason}"))
        return outcome

    async def _record(self, loop: AgentLoop, spec: ChildSpec, status: str, start: float) -> None:
        """The parent log's reference to the transcript, with the usage that session totals
        (and a resume) need; the child's `model_request`s are only in its transcript."""
        self.log.append(
            "subagent",
            id=spec.id,
            status=status,
            model=spec.profile.alias,
            steps=sum(m.role == "assistant" for m in loop.history),
            usage=dataclasses.asdict(loop.usage),
            cost_usd=loop.cost_usd,
            duration_ms=int((time.monotonic() - start) * 1000),
            transcript=str(spec.transcript),
        )
        if self.parent is not None:
            await self.parent.add_usage(loop.usage, loop.cost_usd)


def _check_turns(max_turns: int) -> None:
    if max_turns < 1:
        raise ValueError("max_turns must be at least 1")


async def _stop_kernel(loop: AgentLoop) -> None:
    if loop.tool_ctx.kernel is not None:
        await loop.tool_ctx.kernel.shutdown()


def _outcome(
    agent_id: str, loop: AgentLoop, errors: Sequence[str], max_turns: int, transcript: Path
) -> _Outcome:
    last = loop.history[-1] if loop.history else None
    if last is not None and last.role == "assistant" and not last.tool_uses():
        if text := last.text().strip():
            return _Outcome(agent_id, text)
    reason = _failure(last, errors, max_turns)
    return _Outcome(agent_id, error=f"{reason} (transcript: {transcript})")


def _failure(last: Message | None, errors: Sequence[str], max_turns: int) -> str:
    if errors:
        return errors[-1]
    if last is not None and any(isinstance(b, ToolResultBlock) for b in last.content):
        return f"stopped at max_turns={max_turns} without a final answer"
    return "the model returned an empty response"


def _summary(outcome: _Outcome, start: float) -> str:
    if outcome.error:
        return f"Sub-agent {outcome.id} failed: {outcome.error}"
    return f"Sub-agent {outcome.id} finished in {time.monotonic() - start:.1f}s"
