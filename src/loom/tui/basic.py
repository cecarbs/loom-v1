"""Temporary plain-print UI (M1) to exercise the loop. Replaced by the inline TUI in M2."""

from __future__ import annotations

import asyncio
import json
import queue
import signal
import sys
import threading
from typing import TextIO

from loom.core import events as ev
from loom.core.agent import AgentLoop
from loom.core.events import EventBus

DIM, RED, YELLOW, RESET = "\033[2m", "\033[31m", "\033[33m", "\033[0m"


class BasicRenderer:
    def __init__(self, out: TextIO | None = None) -> None:
        self.out = out or sys.stdout
        self.color = self.out.isatty()
        self._in_text = False
        self._thinking = False
        self._last_usage: ev.UsageUpdated | None = None

    def _style(self, text: str, style: str) -> str:
        return f"{style}{text}{RESET}" if self.color else text

    def _line(self, text: str = "") -> None:
        if self._in_text:
            self.out.write("\n")
            self._in_text = False
        self.out.write(text + "\n")
        self.out.flush()

    async def __call__(self, event: ev.AgentEvent) -> None:
        match event:
            case ev.TextDelta(text):
                self._thinking = False
                self.out.write(text)
                self.out.flush()
                self._in_text = True
            case ev.ThinkingDelta():
                if not self._thinking:
                    self._line(self._style("thinking…", DIM))
                    self._thinking = True
            case ev.ToolCallStarted(_, name, input_):
                self._line(f"▸ {name} {_short_args(input_)}")
            case ev.ToolCallFinished(_, preview, is_error, duration_ms, _):
                first = preview.strip().splitlines()[0] if preview.strip() else ""
                text = f"  └ {first[:120]}  ({duration_ms / 1000:.1f}s)"
                self._line(self._style(text, RED) if is_error else self._style(text, DIM))
            case ev.Notice(text):
                self._line(self._style(f"· {text}", YELLOW))
            case ev.ErrorEvent(text):
                self._line(self._style(f"✗ {text}", RED))
            case ev.UsageUpdated():
                self._last_usage = event
            case ev.TurnCancelled():
                self._line(self._style("[cancelled]", YELLOW))
            case ev.TurnFinished():
                self._line()
                if self._last_usage is not None:
                    u = self._last_usage
                    cost = "—" if u.cost_usd is None else f"${u.cost_usd:.4f}"
                    self._line(self._style(f"[ctx {u.context_pct:.0%} · {cost}]", DIM))
            case _:
                pass


def _short_args(args: dict[str, object]) -> str:
    text = json.dumps(args, ensure_ascii=False)
    return text if len(text) <= 100 else text[:97] + "..."


class LineReader:
    """Reads stdin lines on a daemon thread, one prompt at a time, so a blocked input()
    never holds up shutdown."""

    def __init__(self) -> None:
        self._requests: queue.Queue[tuple[str, asyncio.Future[str | None]]] = queue.Queue()
        self._loop = asyncio.get_running_loop()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        while True:
            prompt, future = self._requests.get()
            try:
                line: str | None = input(prompt)
            except (EOFError, OSError):  # OSError: stdin closed or unavailable
                line = None
            self._loop.call_soon_threadsafe(_resolve, future, line)

    async def readline(self, prompt: str) -> str | None:
        future: asyncio.Future[str | None] = self._loop.create_future()
        self._requests.put((prompt, future))
        return await future


def _resolve(future: asyncio.Future[str | None], value: str | None) -> None:
    if not future.done():
        future.set_result(value)


async def run_basic(agent: AgentLoop, bus: EventBus) -> None:
    bus.subscribe(BasicRenderer())
    reader = LineReader()
    loop = asyncio.get_running_loop()
    print(
        f"loom · {agent.profile.alias} ({agent.profile.model}) · Ctrl+C cancels a turn, "
        "Ctrl+D or /exit quits"
    )
    while True:
        line = await reader.readline("> ")
        if line is None or line.strip() in ("/exit", "/quit"):
            return
        if not line.strip():
            continue
        task = asyncio.create_task(agent.run_turn(line))
        loop.add_signal_handler(signal.SIGINT, task.cancel)
        try:
            await task
        finally:
            loop.remove_signal_handler(signal.SIGINT)
