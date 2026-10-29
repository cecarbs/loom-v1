"""User input (docs/06 "Input"): a prompt_toolkit prompt, or plain lines when piped."""

from __future__ import annotations

import asyncio
import os
import sys
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Protocol

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import AnyFormattedText
from prompt_toolkit.history import FileHistory, History, InMemoryHistory
from prompt_toolkit.input import Input
from prompt_toolkit.input.ansi_escape_sequences import ANSI_SEQUENCES
from prompt_toolkit.key_binding import KeyBindings, KeyPressEvent
from prompt_toolkit.keys import Keys
from prompt_toolkit.output import Output
from prompt_toolkit.patch_stdout import patch_stdout

from loom.tui.commands import Command

# What a Ctrl+C on an empty prompt returns, so the loop can count two in a row.
CTRL_C = "\x03"
MAX_PATH_COMPLETIONS = 200
_SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv"}


class LineSource(Protocol):
    async def read(self) -> str | None:
        """The next submitted input; CTRL_C for Ctrl+C on an empty prompt; None to quit."""
        ...


class LoomCompleter(Completer):
    """`/` completes command names at the start of the input; `@` completes workspace paths."""

    def __init__(self, workspace: Path, commands: Callable[[], Sequence[Command]]) -> None:
        self.workspace = workspace.resolve()
        self.commands = commands

    def get_completions(
        self, document: Document, complete_event: CompleteEvent
    ) -> Iterable[Completion]:
        before = document.text_before_cursor
        if before.startswith("/") and " " not in before:
            for command in self.commands():
                for name in (command.name, *command.aliases):
                    if name.startswith(before[1:]):
                        yield Completion(
                            name, -len(before) + 1, display=f"/{name}", display_meta=command.help
                        )
            return
        word = document.get_word_before_cursor(WORD=True)
        if word.startswith("@"):
            yield from self._paths(word[1:])

    def _paths(self, partial: str) -> Iterable[Completion]:
        head, _, prefix = partial.rpartition("/")
        directory = (self.workspace / head).resolve()
        if not directory.is_relative_to(self.workspace) or not directory.is_dir():
            return
        try:
            entries = sorted(os.scandir(directory), key=lambda e: e.name)
        except OSError:
            return
        count = 0
        for entry in entries:
            name = entry.name
            if not name.startswith(prefix) or (name.startswith(".") and not prefix.startswith(".")):
                continue
            is_dir = entry.is_dir()
            if is_dir and name in _SKIP_DIRS:
                continue
            text = f"{head}/{name}" if head else name
            yield Completion(
                text + ("/" if is_dir else ""),
                -len(partial),
                display=name + ("/" if is_dir else ""),
            )
            count += 1
            if count >= MAX_PATH_COMPLETIONS:
                return


def register_modified_keys() -> None:
    """Teach prompt_toolkit's parser the modified-key encodings some terminals send (xterm
    modifyOtherKeys `CSI 27;<mod>;<code>~` and `CSI <code>;<mod>u`, seen in Ghostty).
    Unknown sequences would otherwise be inserted as text, e.g. `[27;2;27~` for Shift+Esc.
    Any modified Enter inserts a newline (like Alt+Enter); prompt_toolkit's own table maps
    Shift/Ctrl+Enter to plain Enter, which would submit. Must run before the first prompt:
    the parser caches prefix lookups."""
    newline = (Keys.Escape, Keys.ControlM)  # handled by the Alt+Enter binding
    for mod in range(2, 9):  # 2 = Shift, 3 = Alt, 5 = Ctrl, and combinations
        for code, key in (
            (13, newline),
            (27, Keys.Ignore),
            (9, Keys.BackTab if mod == 2 else Keys.ControlI),
            (127, Keys.ControlH),
        ):
            ANSI_SEQUENCES[f"\x1b[27;{mod};{code}~"] = key
            ANSI_SEQUENCES[f"\x1b[{code};{mod}u"] = key


register_modified_keys()


def key_bindings() -> KeyBindings:
    bindings = KeyBindings()

    @bindings.add("escape", "enter")
    @bindings.add("c-j")
    def _newline(event: KeyPressEvent) -> None:
        event.current_buffer.insert_text("\n")

    @bindings.add("c-c")
    def _ctrl_c(event: KeyPressEvent) -> None:
        buffer = event.current_buffer
        if buffer.text:
            buffer.reset()
        else:
            event.app.exit(result=CTRL_C)

    return bindings


class PromptInput:
    def __init__(
        self,
        completer: Completer,
        toolbar: Callable[[], AnyFormattedText],
        *,
        history_path: Path | None = None,
        input: Input | None = None,
        output: Output | None = None,
    ) -> None:
        history: History = InMemoryHistory()
        if history_path is not None:
            history_path.parent.mkdir(parents=True, exist_ok=True)
            history = FileHistory(str(history_path))
        self.session: PromptSession[str] = PromptSession(
            history=history,
            completer=completer,
            complete_while_typing=True,
            key_bindings=key_bindings(),
            bottom_toolbar=toolbar,
            erase_when_done=True,  # the submitted input is re-printed as a "you" panel
            input=input,
            output=output,
        )

    async def read(self) -> str | None:
        try:
            with patch_stdout():  # only while the prompt owns the terminal
                return await self.session.prompt_async("> ")
        except EOFError:  # Ctrl+D on an empty prompt
            return None


class PipedInput:
    """Plain lines from a non-terminal stdin (scripts, manual tests)."""

    def __init__(self, stream=None) -> None:
        self.stream = stream or sys.stdin

    async def read(self) -> str | None:
        line = await asyncio.to_thread(self.stream.readline)
        return line.rstrip("\n") if line else None
