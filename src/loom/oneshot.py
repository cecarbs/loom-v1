"""`loom -p`: a UI without a terminal (docs/09 "CLI"). The final answer goes to stdout, and
notices and errors to stderr, so the output can be piped. Nobody can answer an approval, so
those that would ask are denied; `--mode` decides what runs without asking."""

from __future__ import annotations

import sys
from typing import TextIO

from loom.app import App
from loom.core import events as ev
from loom.core.messages import Message, TextBlock


def final_text(message: Message) -> str:
    """The answer in `message`; a message that calls tools isn't one, even with text."""
    if message.tool_uses():
        return ""
    return "".join(b.text for b in message.content if isinstance(b, TextBlock)).strip()


class OneShot:
    def __init__(self, app: App, *, out: TextIO | None = None, err: TextIO | None = None) -> None:
        self.app = app
        self.out = out or sys.stdout
        self.err = err or sys.stderr
        self.answer = ""
        self.failed = False
        self.cancelled = False
        app.bus.subscribe(self)

    def _say(self, text: str) -> None:
        print(f"loom: {text}", file=self.err, flush=True)

    async def __call__(self, event: ev.AgentEvent) -> None:
        match event:
            case ev.AssistantMessageDone(message=message):
                self.answer = final_text(message)
            case ev.ApprovalRequested(request_id=request_id, tool=tool, agent_id=agent_id):
                who = f"[{agent_id}] " if agent_id else ""
                self._say(f"{who}denied {tool}: no one to approve it (see --mode)")
                self.app.bus.respond(request_id, ev.Decision("deny", by="policy"))
            case ev.Notice(text=text):
                self._say(text)
            case ev.ErrorEvent(text=text):
                self.failed = True
                self._say(text)
            case ev.TurnCancelled():
                self.cancelled = True
            case _:
                pass

    async def run(self, prompt: str) -> int:
        """Exit code: 0 with a final answer, 1 without one, 130 if interrupted."""
        for notice in self.app.notices:
            self._say(notice)
        await self.app.submit(prompt)
        if self.cancelled:
            return 130
        if self.failed or not self.answer:
            if not self.failed:
                self._say("the model gave no final answer")
            return 1
        print(self.answer, file=self.out, flush=True)
        return 0
