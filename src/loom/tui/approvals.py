"""Approval prompt (docs/06 "Approvals"): single-key answer, one request at a time."""

from __future__ import annotations

import asyncio

from prompt_toolkit import PromptSession
from prompt_toolkit.application import Application
from prompt_toolkit.input import Input
from prompt_toolkit.key_binding import KeyBindings, KeyPressEvent
from prompt_toolkit.layout import Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.output import Output
from rich.console import Console
from rich.text import Text

from loom.core.events import ApprovalRequested, Decision

CHOICES = "  [y] allow once   [a] allow this tool for session   [n] deny   [m] deny with message"
_KEYS = {"y": "allow_once", "a": "allow_session", "n": "deny", "m": "message"}


class ApprovalPrompt:
    def __init__(
        self, console: Console, *, input: Input | None = None, output: Output | None = None
    ) -> None:
        self.console = console
        self._input = input
        self._output = output
        self._lock = asyncio.Lock()  # concurrent requests are shown one at a time

    async def ask(self, request: ApprovalRequested) -> Decision:
        async with self._lock:
            self._show(request)
            key = await self._read_key()
            if key == "message":
                message = await self._read_message()
                decision = Decision("deny", message or None)
            elif key in ("allow_once", "allow_session"):
                decision = Decision(key)
            else:
                decision = Decision("deny")
            self.console.print(Text(f"  → {describe(decision)}", style="dim"))
            return decision

    def _show(self, request: ApprovalRequested) -> None:
        title = Text("⚠ ", style="yellow")
        if request.agent_id:
            title.append(f"[{request.agent_id}] ", style="magenta")
        title.append(request.tool, style="bold")
        title.append(f"  [{request.effect}]", style="yellow")
        self.console.print(title)
        for line in request.args_preview.splitlines():
            self.console.print(Text(f"  {line}"), soft_wrap=True)
        self.console.print(Text(CHOICES, style="dim"))

    async def _read_key(self) -> str:
        bindings = KeyBindings()

        def answer(value: str):
            def handler(event: KeyPressEvent) -> None:
                event.app.exit(result=value)

            return handler

        for key, value in _KEYS.items():
            bindings.add(key)(answer(value))
            bindings.add(key.upper())(answer(value))
        # Ctrl+C / Ctrl+D / Esc never allow.
        for key in ("c-c", "c-d", "escape"):
            bindings.add(key)(answer("deny"))
        bindings.add("<any>")(lambda event: None)

        app: Application[str] = Application(
            layout=Layout(Window(FormattedTextControl("  > "), height=1)),
            key_bindings=bindings,
            erase_when_done=True,
            input=self._input,
            output=self._output,
        )
        return await app.run_async()

    async def _read_message(self) -> str:
        session: PromptSession[str] = PromptSession(input=self._input, output=self._output)
        try:
            return (await session.prompt_async("  message to the model: ")).strip()
        except (KeyboardInterrupt, EOFError):
            return ""


class DenyAll:
    """Used when there's no terminal to ask on (piped input): approvals resolve as deny."""

    def __init__(self, console: Console) -> None:
        self.console = console

    async def ask(self, request: ApprovalRequested) -> Decision:
        self.console.print(Text(f"⚠ {request.tool}: denied (no terminal to ask)", "yellow"))
        return Decision("deny", by="policy")


def describe(decision: Decision) -> str:
    match decision.action:
        case "allow_once":
            return "allowed once"
        case "allow_session":
            return "allowed for this session"
        case _:
            return f"denied: {decision.message}" if decision.message else "denied"
