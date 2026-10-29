"""Rich renderer driven only by bus events (docs/06 "Rendering").

Terminal ownership: while a turn runs the renderer owns the terminal and may show one
transient `Live` region (streaming text, thinking, or tool output). `close()` stops it and
must run before the input prompt is shown again. Approval prompts suspend the region.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Literal, Protocol

from rich.console import Console, ConsoleOptions, RenderableType, RenderResult
from rich.live import Live
from rich.markdown import Markdown
from rich.segment import Segment
from rich.syntax import Syntax
from rich.text import Text

from loom.core import events as ev

PREVIEW_LINES = 8
OUTPUT_TAIL_LINES = 12
REFRESH_PER_SECOND = 8

Region = Literal["text", "thinking", "output"]


class Approver(Protocol):
    async def ask(self, request: ev.ApprovalRequested) -> ev.Decision: ...


@dataclass
class Status:
    """What the bottom toolbar shows; updated from events, reset on a new session."""

    context_pct: float = 0.0
    cost_usd: float | None = None
    kernel_state: str = "off"
    kernel_vars: int | None = None

    def kernel(self) -> str:
        """`kernel idle (3 vars)`; plain `kernel off` before the first execute."""
        if self.kernel_vars is None or self.kernel_state in ("off", "dead", "starting"):
            return f"kernel {self.kernel_state}"
        plural = "" if self.kernel_vars == 1 else "s"
        return f"kernel {self.kernel_state} ({self.kernel_vars} var{plural})"


class _Tail:
    """Render only the last lines that fit, so a growing Live region never exceeds the
    screen (taller regions can't be redrawn in place and leave duplicates in scrollback)."""

    def __init__(self, renderable: RenderableType, max_lines: int | None = None) -> None:
        self.renderable = renderable
        self.max_lines = max_lines

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        limit = self.max_lines or max(console.size.height - 3, 3)
        lines = console.render_lines(self.renderable, options, pad=False)
        if len(lines) > limit:
            yield Text("…", style="dim")
            yield Segment.line()
            lines = lines[-(limit - 1) :]
        for line in lines:
            yield from line
            yield Segment.line()


class Renderer:
    def __init__(self, console: Console, approver: Approver, bus: ev.EventBus) -> None:
        self.console = console
        self.approver = approver
        self.bus = bus
        self.status = Status()
        self.show_thinking = False
        self._region: Region | None = None
        self._buffer: list[str] = []
        self._live: Live | None = None

    @property
    def active(self) -> bool:
        return self._live is not None

    async def __call__(self, event: ev.AgentEvent) -> None:
        match event:
            case ev.TurnStarted():
                pass
            case ev.ThinkingDelta(text):
                self._feed("thinking", text)
            case ev.TextDelta(text):
                self._feed("text", text)
            case ev.AssistantMessageDone():
                self.close()
            case ev.ToolCallStarted(_, name, input_):
                self.close()
                self._tool_header(name, input_)
            case ev.ToolCallOutput(_, chunk):
                self._feed("output", chunk)
            case ev.ToolCallFinished():
                self.close()
                self._tool_result(event)
            case ev.ApprovalRequested():
                with self.suspended():
                    decision = await self.approver.ask(event)
                self.bus.respond(event.request_id, decision)
            case ev.KernelStatus(state, variables):
                self.status.kernel_state = state
                self.status.kernel_vars = variables
            case ev.Compacted(before, after):
                self.close()
                self.console.print(
                    Text(f"· Compacted: ~{before:,} → ~{after:,} tokens", style="dim")
                )
            case ev.Notice(text):
                self.close()
                self.console.print(Text(f"· {text}", style="yellow"))
            case ev.ErrorEvent(text):
                self.close()
                self.console.print(Text(f"✗ {text}", style="red"))
            case ev.UsageUpdated(context_pct=pct, cost_usd=cost):
                self.status.context_pct = pct
                self.status.cost_usd = cost
            case ev.TurnCancelled():
                self.close()
                self.console.print(Text("[cancelled]", style="yellow"))
            case ev.TurnFinished():
                self.close()
                if not self.console.is_terminal:  # no toolbar when piped; print the status
                    self.console.print(Text(status_line(self.status), style="dim"))

    # --- the live region ---------------------------------------------------------

    def _feed(self, region: Region, text: str) -> None:
        if self._region != region:
            self.close()
            self._region = region
            self._start_live()
        self._buffer.append(text)

    def _start_live(self) -> None:
        if not self.console.is_terminal:
            return
        self._live = Live(
            get_renderable=self._renderable,
            console=self.console,
            refresh_per_second=REFRESH_PER_SECOND,
            transient=True,
            redirect_stdout=False,
            redirect_stderr=False,
        )
        self._live.start()

    def _renderable(self) -> RenderableType:
        text = "".join(self._buffer)
        match self._region:
            case "text":
                return _Tail(Markdown(text))
            case "thinking" if self.show_thinking:
                return _Tail(Text(text, style="dim italic"))
            case "thinking":
                return Text("thinking…", style="dim")
            case "output":
                return _Tail(Text(text.rstrip("\n"), style="dim"), OUTPUT_TAIL_LINES)
            case _:
                return Text("")

    def close(self) -> None:
        """Stop the live region and print its final form to scrollback."""
        if self._region is None:
            return
        if self._live is not None:
            self._live.stop()
            self._live = None
        text = "".join(self._buffer)
        region, self._region, self._buffer = self._region, None, []
        if region == "text" and text.strip():
            self.console.print(Markdown(text))
        elif region == "thinking":
            if self.show_thinking:
                self.console.print(Text(text.strip(), style="dim italic"))
            else:
                self.console.print(Text("thinking…", style="dim"))
        # Streamed tool output is transient; ToolCallFinished prints the preview.

    @contextmanager
    def suspended(self) -> Iterator[None]:
        """Hand the terminal to a prompt, then resume the live region if there was one.

        Resuming uses a new Live: a restarted one remembers its old height and would move
        the cursor up over whatever the prompt printed."""
        live = self._live
        if live is not None:
            live.stop()
            self._live = None
        try:
            yield
        finally:
            if live is not None and self._region is not None and self._live is None:
                self._start_live()

    # --- tool calls --------------------------------------------------------------

    def _tool_header(self, name: str, input_: dict[str, object]) -> None:
        code = input_.get("code") if name == "execute" else None
        header = Text("▸ ", style="cyan", no_wrap=True, overflow="ellipsis")
        header.append(name, style="bold")
        if not isinstance(code, str):
            header.append(f"  {short_args(input_)}", style="dim")
        self.console.print(header)
        if isinstance(code, str):
            self.console.print(Syntax(code, "python", theme="ansi_dark", padding=(0, 2)))

    def _tool_result(self, event: ev.ToolCallFinished) -> None:
        duration = f"{event.duration_ms / 1000:.1f}s"
        diff = event.meta.get("diff")
        if isinstance(diff, str) and diff and not event.is_error:
            self.console.print(Text(f"  └ {first_line(event.result_preview)}  {duration}", "dim"))
            self.console.print(Syntax(diff.rstrip("\n"), "diff", theme="ansi_dark", padding=(0, 4)))
            return
        lines = event.result_preview.rstrip("\n").splitlines() or [""]
        shown = lines[:PREVIEW_LINES]
        if len(lines) > PREVIEW_LINES:
            shown.append(f"… {len(lines) - PREVIEW_LINES} more lines")
        style = "red" if event.is_error else "dim"
        body = Text(no_wrap=True, overflow="ellipsis")
        for i, line in enumerate(shown):
            prefix = "  └ " if i == 0 else "    "
            body.append(prefix + line, style=style)
            if i == 0:
                body.append(f"  {duration}", style="dim")
            if i < len(shown) - 1:
                body.append("\n")
        self.console.print(body)


def short_args(args: dict[str, object], limit: int = 100) -> str:
    text = json.dumps(args, ensure_ascii=False)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def first_line(text: str) -> str:
    stripped = text.strip()
    return stripped.splitlines()[0] if stripped else ""


def format_cost(cost: float | None) -> str:
    return "—" if cost is None else f"${cost:.2f}" if cost >= 1 else f"${cost:.4f}"


def status_line(status: Status, model: str = "", *rest: str) -> str:
    """`sonnet · ctx 18% · $0.04 · none · ask · kernel idle (3 vars)` (the parts given)."""
    parts = [model, f"ctx {status.context_pct:.0%}", format_cost(status.cost_usd), *rest]
    return " · ".join(p for p in parts if p)
