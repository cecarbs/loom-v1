import asyncio
import io
import os
import signal
from pathlib import Path

import pytest
from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from rich.console import Console

from loom.app import App
from loom.config import LoomConfig
from loom.core import events as ev
from loom.core.messages import Message, TextBlock
from loom.providers.fake import FakeProvider, FakeResponse, Script
from loom.tui.app import InlineUI
from loom.tui.approvals import ApprovalPrompt, DenyAll
from loom.tui.commands import CommandContext, CommandRegistry, builtin_commands
from loom.tui.input import CTRL_C, LoomCompleter, PromptInput
from loom.tui.render import Renderer, Status, status_line


def console(terminal: bool = False) -> Console:
    return Console(
        file=io.StringIO(), force_terminal=terminal, width=80, height=24, color_system=None
    )


def output(c: Console) -> str:
    file = c.file
    assert isinstance(file, io.StringIO)
    return file.getvalue()


class Lines:
    """A scripted LineSource."""

    def __init__(self, *lines: str | None) -> None:
        self.lines = list(lines)

    async def read(self) -> str | None:
        return self.lines.pop(0) if self.lines else None


class Approve:
    def __init__(self) -> None:
        self.seen: list[ev.ApprovalRequested] = []

    async def ask(self, request: ev.ApprovalRequested) -> ev.Decision:
        self.seen.append(request)
        return ev.Decision("allow_once")


def make_app(tmp_path: Path, script: list[Script], config: LoomConfig | None = None) -> App:
    workspace = tmp_path / "ws"
    workspace.mkdir(exist_ok=True)
    provider = FakeProvider(script)
    return App(
        workspace,
        config or LoomConfig(),
        provider_factory=lambda profile: provider,
        home=tmp_path / "home",
    )


# --- renderer ---------------------------------------------------------------------


async def test_renderer_prints_final_markdown_once_when_piped() -> None:
    c = console()
    r = Renderer(c, DenyAll(c), ev.EventBus())
    msg = Message("assistant", [TextBlock("# Title\n\nsome **bold** text")])
    for event in [
        ev.TurnStarted(),
        ev.ThinkingDelta("let me think"),
        ev.TextDelta("# Title\n\nsome "),
        ev.TextDelta("**bold** text"),
        ev.AssistantMessageDone(msg),
        ev.UsageUpdated(10, 5, 0, 0.18, 0.04),
        ev.TurnFinished(ev.Usage()),
    ]:
        await r(event)
    out = output(c)
    assert out.count("thinking…") == 1
    assert "let me think" not in out  # collapsed by default
    assert out.count("Title") == 1 and "bold" in out and "**" not in out
    assert out.strip().endswith("ctx 18% · $0.0400")
    assert not r.active


async def test_renderer_streams_in_live_region_on_a_terminal() -> None:
    c = console(terminal=True)
    r = Renderer(c, DenyAll(c), ev.EventBus())
    await r(ev.TextDelta("hello "))
    assert r.active  # the live region owns the terminal while text streams
    await r(ev.TextDelta("world"))
    await r(ev.AssistantMessageDone(Message("assistant", [TextBlock("hello world")])))
    assert not r.active
    await r(ev.ToolCallStarted("t1", "bash", {"command": "ls"}))
    await r(ev.ToolCallOutput("t1", "a\nb\n"))
    assert r.active
    await r(ev.ToolCallFinished("t1", "a\nb", False, 1200))
    assert not r.active
    out = output(c)
    assert "hello world" in out and "▸ bash" in out and "1.2s" in out


async def test_renderer_thinking_shown_when_enabled() -> None:
    c = console()
    r = Renderer(c, DenyAll(c), ev.EventBus())
    r.show_thinking = True
    await r(ev.ThinkingDelta("step one"))
    await r(ev.TextDelta("answer"))
    r.close()
    out = output(c)
    assert "step one" in out and "thinking…" not in out and "answer" in out


async def test_renderer_tool_results() -> None:
    c = console()
    r = Renderer(c, DenyAll(c), ev.EventBus())
    diff = "--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-a = 1\n+a = 2\n"
    await r(ev.ToolCallStarted("t1", "edit", {"path": "f.py"}))
    await r(ev.ToolCallFinished("t1", "Edited f.py", False, 10, {"diff": diff}))
    await r(ev.ToolCallStarted("t2", "read", {"path": "big"}))
    await r(ev.ToolCallFinished("t2", "\n".join(f"l{i}" for i in range(20)), False, 5))
    await r(ev.ToolCallFinished("t3", "boom", True, 5))
    await r(ev.ToolCallStarted("t4", "execute", {"code": "x = 1\nprint(x)"}))
    out = output(c)
    assert "+a = 2" in out and "-a = 1" in out
    assert "l7" in out and "l8" not in out and "12 more lines" in out
    assert "boom" in out
    assert "print(x)" in out and '"code"' not in out


async def test_renderer_cancel_and_notices() -> None:
    c = console()
    r = Renderer(c, DenyAll(c), ev.EventBus())
    await r(ev.TextDelta("partial"))
    await r(ev.Notice("retrying"))
    await r(ev.ErrorEvent("bad"))
    await r(ev.TurnCancelled())
    out = output(c)
    assert out.index("partial") < out.index("retrying") < out.index("bad")
    assert "[cancelled]" in out


def test_status_line() -> None:
    assert status_line(Status(0.18, 0.04), "sonnet", "none", "ask") == (
        "sonnet · ctx 18% · $0.0400 · none · ask"
    )
    assert status_line(Status(0.0, None), "local") == "local · ctx 0% · —"
    assert status_line(Status(0.5, 12.345)) == "ctx 50% · $12.35"


# --- approvals --------------------------------------------------------------------

REQUEST = ev.ApprovalRequested("r1", "splunk.create_saved_search", 'name = "x"', "write")


async def _ask(*keys: str, request: ev.ApprovalRequested = REQUEST) -> tuple[ev.Decision, str]:
    c = console()
    with create_pipe_input() as pipe:
        prompt = ApprovalPrompt(c, input=pipe, output=DummyOutput())
        task = asyncio.create_task(prompt.ask(request))
        for key in keys:
            await asyncio.sleep(0.05)  # let each prompt start before its keys arrive
            pipe.send_text(key)
        decision = await asyncio.wait_for(task, 5)
    return decision, output(c)


@pytest.mark.parametrize(
    ("key", "action"),
    [("y", "allow_once"), ("a", "allow_session"), ("n", "deny"), ("\x03", "deny")],
)
async def test_approval_keys(key: str, action: str) -> None:
    decision, out = await _ask(key)
    assert decision.action == action and decision.by == "user"
    assert "splunk.create_saved_search" in out and "[write]" in out and 'name = "x"' in out


async def test_approval_ignores_other_keys() -> None:
    decision, _ = await _ask("x", "y")
    assert decision.action == "allow_once"


async def test_approval_deny_with_message() -> None:
    decision, out = await _ask("m", "too risky\r")
    assert decision == ev.Decision("deny", "too risky")
    assert "denied: too risky" in out


async def test_approval_shows_subagent_id() -> None:
    request = ev.ApprovalRequested("r2", "bash", "rm -rf build", "write", agent_id="sub-1")
    _, out = await _ask("n", request=request)
    assert "[sub-1]" in out


async def test_approval_round_trip_through_bus() -> None:
    bus = ev.EventBus()
    c = console()
    approver = Approve()
    bus.subscribe(Renderer(c, approver, bus))
    decision = await bus.request_approval(REQUEST)
    assert decision.action == "allow_once" and approver.seen == [REQUEST]


async def test_unanswered_approval_never_resolves() -> None:
    bus = ev.EventBus()  # no subscriber answers
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(bus.request_approval(REQUEST), 0.05)


async def test_deny_all_when_piped() -> None:
    c = console()
    decision = await DenyAll(c).ask(REQUEST)
    assert decision == ev.Decision("deny", by="policy")
    assert "denied" in output(c)


# --- input ------------------------------------------------------------------------


async def _prompt(text: str) -> str | None:
    with create_pipe_input() as pipe:
        source = PromptInput(
            LoomCompleter(Path.cwd(), lambda: []),
            lambda: "status",
            input=pipe,
            output=DummyOutput(),
        )
        pipe.send_text(text)
        return await asyncio.wait_for(source.read(), 5)


async def test_prompt_enter_submits_and_ctrl_j_inserts_newline() -> None:
    assert await _prompt("one\x0atwo\r") == "one\ntwo"
    assert await _prompt("one\x1b\rtwo\r") == "one\ntwo"  # Alt+Enter


async def test_prompt_ctrl_c_and_ctrl_d() -> None:
    assert await _prompt("\x03") == CTRL_C
    assert await _prompt("typed\x03again\r") == "again"  # Ctrl+C clears non-empty input
    assert await _prompt("\x04") is None


def test_completer_commands_and_paths(tmp_path: Path) -> None:
    (tmp_path / "src" / "pkg").mkdir(parents=True)
    (tmp_path / "src" / "main.py").write_text("")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".env").write_text("")
    completer = LoomCompleter(tmp_path, builtin_commands().commands)

    def complete(text: str) -> list[str]:
        doc = Document(text, len(text))
        return [c.text for c in completer.get_completions(doc, CompleteEvent())]

    assert complete("/mo") == ["model"]
    assert "quit" in complete("/q")
    assert complete("/model x") == []
    assert complete("look at @s") == ["src/"]
    assert complete("look at @src/") == ["src/main.py", "src/pkg/"]
    assert complete("@src/m") == ["src/main.py"]
    assert complete("@") == ["src/"]  # hidden entries need a leading dot
    assert complete("@.") == [".env"]  # .git is skipped
    assert complete("@../") == []  # nothing outside the workspace
    assert complete("email me@s") == []


# --- commands ---------------------------------------------------------------------


def test_is_command() -> None:
    assert CommandRegistry.is_command("/help")
    assert CommandRegistry.is_command("  /model sonnet")
    assert not CommandRegistry.is_command("/usr/bin/env is broken")
    assert not CommandRegistry.is_command("/")
    assert not CommandRegistry.is_command("what does /help do")


async def _command(app: App, line: str) -> tuple[CommandContext, str]:
    c = console()
    ui = InlineUI(app, console=c, source=Lines(), approver=Approve())
    ctx = CommandContext(app, c, ui.renderer, ui.commands)
    await ui.commands.run(ctx, line)
    return ctx, output(c)


async def test_help_model_cost_thinking_exit(tmp_path: Path) -> None:
    config = LoomConfig.model_validate(
        {"models": {"local": {"provider": "openai_compat", "model": "qwen3:8b"}}}
    )
    app = make_app(tmp_path, [], config)

    _, out = await _command(app, "/help")
    assert all(f"/{n}" in out for n in ("help", "model", "cost", "clear", "thinking", "exit"))

    _, out = await _command(app, "/model")
    assert "sonnet" in out and "local" in out

    _, out = await _command(app, "/model local")
    assert "Switched to local" in out and app.profile.alias == "local"
    assert any(r["kind"] == "model_switch" for r in app.agent.log.read())

    _, out = await _command(app, "/model nope")
    assert "Unknown model 'nope'" in out and app.profile.alias == "local"

    _, out = await _command(app, "/cost")
    assert "Input tokens" in out and "Session cost" in out

    ctx, out = await _command(app, "/thinking on")
    assert ctx.renderer.show_thinking and "shown" in out

    ctx, _ = await _command(app, "/quit")
    assert ctx.exit_requested

    _, out = await _command(app, "/bogus")
    assert "Unknown command /bogus" in out


async def test_clear_starts_new_session(tmp_path: Path) -> None:
    app = make_app(tmp_path, [[TextBlock("hi")]])
    await app.submit("hello")
    old_log = app.agent.log.path
    _, out = await _command(app, "/clear")
    assert app.agent.history == [] and app.agent.log.path != old_log
    assert app.session_id in out


# --- the main loop ------------------------------------------------------------------


async def test_loop_runs_turns_commands_and_exits(tmp_path: Path) -> None:
    app = make_app(tmp_path, [[TextBlock("first answer")], [TextBlock("second answer")]])
    c = console()
    ui = InlineUI(app, console=c, source=Lines("hi", "", "/cost", "again", "/exit", "never"))
    await ui.run()
    out = output(c)
    assert "first answer" in out and "second answer" in out and "Input tokens" in out
    assert out.count("╭─ you") == 2
    assert "never" not in out


async def test_two_ctrl_c_on_empty_prompt_quits(tmp_path: Path) -> None:
    app = make_app(tmp_path, [])
    c = console()
    source = Lines(CTRL_C, CTRL_C, "unreached")
    await InlineUI(app, console=c, source=source).run()
    assert source.lines == ["unreached"] and "Ctrl+C again" in output(c)


async def test_ctrl_c_cancels_running_turn_and_returns_to_prompt(tmp_path: Path) -> None:
    app = make_app(
        tmp_path,
        [
            FakeResponse([TextBlock("thinking out loud")], hang_after_text=True),
            [TextBlock("after cancel")],
        ],
    )
    c = console(terminal=True)
    source = Lines("long task", "next", None)
    ui = InlineUI(app, console=c, source=source, approver=Approve())
    loop = asyncio.get_running_loop()
    sent = False

    async def interrupt_once(event: ev.AgentEvent) -> None:
        nonlocal sent
        if isinstance(event, ev.TextDelta) and not sent:
            sent = True
            # A real SIGINT, as the terminal sends on Ctrl+C.
            loop.call_later(0.05, os.kill, os.getpid(), signal.SIGINT)

    app.bus.subscribe(interrupt_once)
    await asyncio.wait_for(ui.run(), 5)

    out = output(c)
    assert "[cancelled]" in out and "after cancel" in out
    first_reply = app.agent.history[1]
    assert first_reply.meta.get("interrupted") is True
    assert not ui.renderer.active
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler


async def test_approval_suspends_live_region_and_resumes_with_a_fresh_one() -> None:
    c = console(terminal=True)
    bus = ev.EventBus()
    states: list[bool] = []

    class Check:
        async def ask(self, request: ev.ApprovalRequested) -> ev.Decision:
            states.append(r.active)  # the prompt owns the terminal: no live region
            return ev.Decision("allow_once")

    r = Renderer(c, Check(), bus)
    bus.subscribe(r)
    await r(ev.ToolCallOutput("t1", "line\n"))
    before = r._live
    assert await bus.request_approval(REQUEST) == ev.Decision("allow_once")
    # A restarted Live would redraw over the prompt's lines, so a new one is used.
    assert states == [False] and r.active and r._live is not before
    r.close()


def test_no_echo_during_turn_restores_terminal() -> None:
    import pty
    import termios

    from loom.tui.app import no_echo

    master, slave = pty.openpty()
    try:
        with os.fdopen(slave, "r", closefd=False) as stream:
            before = termios.tcgetattr(slave)
            assert before[3] & termios.ECHO
            with no_echo(stream):
                lflag = termios.tcgetattr(slave)[3]
                assert not lflag & termios.ECHO and not lflag & termios.ECHOCTL
                assert lflag & termios.ISIG  # Ctrl+C still sends SIGINT
            assert termios.tcgetattr(slave) == before
        with no_echo(io.StringIO()):  # not a terminal: no-op
            pass
    finally:
        os.close(master)
        os.close(slave)


@pytest.mark.parametrize(
    "seq",
    ["\x1b[27;2;13~", "\x1b[13;2u", "\x1b[27;5;13~", "\x1b[13;3u"],  # Shift/Ctrl/Alt+Enter
)
async def test_modified_enter_inserts_newline(seq: str) -> None:
    assert await _prompt(f"one{seq}two\r") == "one\ntwo"


async def test_modified_escape_is_not_inserted_as_text() -> None:
    # Ghostty sent this for Shift+Esc; it used to appear in the prompt as "[27;2;27~".
    assert await _prompt("a\x1b[27;2;27~b\x1b[27;5u\r") == "ab"
