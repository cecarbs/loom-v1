"""The inline TUI's main loop (docs/06): prompt → turn → prompt, never both at once."""

from __future__ import annotations

import asyncio
import contextlib
import signal
import sys
import termios
from collections.abc import Coroutine, Iterator
from typing import Any, TextIO

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from loom.app import App
from loom.session.store import loom_home
from loom.tui.approvals import ApprovalPrompt, DenyAll
from loom.tui.commands import CommandContext, CommandRegistry, builtin_commands
from loom.tui.input import CTRL_C, LineSource, LoomCompleter, PipedInput, PromptInput
from loom.tui.render import Approver, Renderer, status_line


class InlineUI:
    def __init__(
        self,
        app: App,
        *,
        console: Console | None = None,
        source: LineSource | None = None,
        approver: Approver | None = None,
        commands: CommandRegistry | None = None,
    ) -> None:
        self.app = app
        self.console = console or Console(highlight=False)
        interactive = sys.stdin.isatty() and self.console.is_terminal
        self.commands = commands or builtin_commands()
        if approver is None:
            approver = ApprovalPrompt(self.console) if interactive else DenyAll(self.console)
        self.renderer = Renderer(self.console, approver, app.bus)
        self.renderer.status.cost_usd = app.agent.cost_usd
        app.bus.subscribe(self.renderer)
        if source is None:
            completer = LoomCompleter(app.workspace, self.commands.commands)
            source = (
                PromptInput(completer, self.toolbar, history_path=loom_home() / "history")
                if interactive
                else PipedInput()
            )
        self.source = source

    def toolbar(self) -> str:
        return " " + status_line(
            self.renderer.status,
            self.app.profile.alias,
            self.app.backend.describe(),
            self.app.mode,
            self.renderer.status.kernel(),
        )

    async def run(self) -> None:
        profile = self.app.profile
        self.console.print(
            Text.assemble(
                ("loom", "bold"),
                f" · {profile.alias} ({profile.model}) · ",
                ("/help for commands, Ctrl+D to quit", "dim"),
            )
        )
        for notice in self.app.notices:
            self.console.print(Text(f"· {notice}", style="yellow"))
        ctrl_c_pending = False
        while True:
            line = await self.source.read()
            if line is None:
                return
            if line == CTRL_C:
                if ctrl_c_pending:
                    return
                ctrl_c_pending = True
                self.console.print(Text("(Ctrl+C again to quit)", style="dim"))
                continue
            ctrl_c_pending = False
            if not line.strip():
                continue
            if CommandRegistry.is_command(line):
                ctx = CommandContext(self.app, self.console, self.renderer, self.commands)
                await self.commands.run(ctx, line)
                if ctx.exit_requested:
                    return
                continue
            self.console.print(
                Panel(Text(line), title="you", title_align="left", border_style="dim")
            )
            await self.run_turn(self.app.submit(line))

    async def run_turn(self, turn: Coroutine[Any, Any, None]) -> None:
        """Run a turn with Ctrl+C (SIGINT) cancelling it, then give the terminal back."""
        task = asyncio.create_task(turn)
        loop = asyncio.get_running_loop()
        try:
            loop.add_signal_handler(signal.SIGINT, task.cancel)
            handler_installed = True
        except (NotImplementedError, RuntimeError):  # not the main thread / no signals
            handler_installed = False
        try:
            with no_echo(sys.stdin):
                await task
        finally:
            if handler_installed:
                loop.remove_signal_handler(signal.SIGINT)
            self.renderer.close()


@contextlib.contextmanager
def no_echo(stream: TextIO) -> Iterator[None]:
    """Turn off terminal echo while a turn runs. The echoed `^C` (or anything typed) would
    land inside the Live region and shift the cursor, so the region's erase misses a line.
    Ctrl+C still raises SIGINT. Keys typed during the turn are discarded afterwards
    (type-ahead isn't supported in v1)."""
    try:
        fd = stream.fileno()
        saved = termios.tcgetattr(fd)
    except (OSError, ValueError, termios.error, AttributeError):  # not a terminal
        yield
        return
    quiet = list(saved)
    quiet[3] &= ~(termios.ECHO | termios.ECHOCTL)  # lflags
    termios.tcsetattr(fd, termios.TCSANOW, quiet)
    try:
        yield
    finally:
        termios.tcflush(fd, termios.TCIFLUSH)
        termios.tcsetattr(fd, termios.TCSANOW, saved)
