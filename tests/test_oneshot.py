"""`loom -p` (docs/09 "CLI") and the first-run tutorial (docs/06, M11)."""

import io
import re
from pathlib import Path

import pytest
from loom_tools_demo import package as demo_package
from rich.console import Console

from loom import cli
from loom.app import App
from loom.config import LoomConfig
from loom.core.messages import TextBlock, ToolResultBlock, ToolUseBlock
from loom.extensions.api import ExtensionAPI, ExtensionEnv, Extensions
from loom.extensions.loader import load_extensions
from loom.oneshot import OneShot
from loom.providers.base import ProviderError
from loom.providers.fake import FakeProvider, Script
from loom.tui.app import InlineUI
from loom.tui.commands import TUTORIAL, builtin_commands


def make_app(tmp_path: Path, script: list[Script], mode: str = "auto-write", **kw) -> App:
    ws = tmp_path / "ws"
    ws.mkdir(exist_ok=True)
    provider = FakeProvider(script)
    config = LoomConfig.model_validate({"approvals": {"mode": mode}, **kw})
    return App(
        ws,
        config,
        provider_factory=lambda p: provider,
        home=tmp_path / "home",
        packages=[demo_package],
        extensions=Extensions(),
    )


async def run(app: App, prompt: str = "go") -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    await app.start()
    try:
        code = await OneShot(app, out=out, err=err).run(prompt)
    finally:
        await app.aclose()
    return code, out.getvalue(), err.getvalue()


def execute(code: str) -> list[ToolUseBlock]:
    return [ToolUseBlock("e1", "execute", {"code": code})]


async def test_final_answer_only_on_stdout(tmp_path: Path) -> None:
    """Acceptance: a final answer and exit code 0; text before tool calls isn't the answer."""
    script: list[Script] = [
        [TextBlock("Let me look."), *execute("1 + 1")],
        [TextBlock("The answer is 2.")],
    ]
    assert await run(make_app(tmp_path, script)) == (0, "The answer is 2.\n", "")


async def test_approvals_are_denied_by_default(tmp_path: Path) -> None:
    """Acceptance: what would ask is denied; the model is told and the turn still ends."""
    code = (
        "try:\n    demo.delete_ticket('T-101')\n"
        "except PermissionDenied as e:\n    print('denied:', e)\n"
        "len(demo.search_tickets())"
    )
    app = make_app(tmp_path, [execute(code), [TextBlock("Could not delete it.")]])
    exit_code, out, err = await run(app)
    assert (exit_code, out) == (0, "Could not delete it.\n")
    assert "denied demo.delete_ticket" in err
    result = next(b for m in app.agent.history for b in m.content if isinstance(b, ToolResultBlock))
    assert "denied: demo.delete_ticket (destructive) is not allowed" in str(result.content)
    assert str(result.content).rstrip().endswith("3")  # nothing was deleted
    [entry] = [r for r in app.agent.log.read() if r["kind"] == "approval"]
    assert (entry["action"], entry["by"]) == ("deny", "policy")


async def test_mode_can_allow_more(tmp_path: Path) -> None:
    code = "print(demo.delete_ticket('T-101'))"
    app = make_app(tmp_path, [execute(code), [TextBlock("Deleted.")]], mode="yolo")
    exit_code, out, err = await run(app)
    assert (exit_code, out, err) == (0, "Deleted.\n", "")


async def test_no_final_answer_exits_1(tmp_path: Path) -> None:
    empty = make_app(tmp_path, [[]])
    exit_code, out, err = await run(empty)
    assert (exit_code, out) == (1, "")
    assert "empty response" in err and "no final answer" in err


async def test_model_error_exits_1(tmp_path: Path) -> None:
    error = ProviderError("auth", "bad key")
    exit_code, out, err = await run(make_app(tmp_path, [error]))
    assert (exit_code, out) == (1, "")
    assert "bad key" in err


async def test_step_limit_is_not_an_answer(tmp_path: Path) -> None:
    script: list[Script] = [[TextBlock("Working on it."), *execute("1")]]
    exit_code, out, err = await run(make_app(tmp_path, script, agent={"max_steps": 1}))
    assert (exit_code, out) == (1, "")
    assert "Step limit reached" in err


@pytest.fixture
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    ws = tmp_path / "ws"
    ws.mkdir()
    monkeypatch.chdir(ws)
    return ws


def test_cli_prints_and_exits_0(
    isolated: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        "loom.app.create_provider", lambda *a, **kw: FakeProvider([[TextBlock("hello")]])
    )
    assert cli.main(["-p", "say hello"]) == 0
    captured = capsys.readouterr()
    assert captured.out == "hello\n"
    assert "First time" not in captured.out + captured.err


def test_cli_rejects_an_empty_prompt(isolated: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["-p", "  "]) == 2
    assert "needs a prompt" in capsys.readouterr().err


def test_cli_never_asks_for_trust(isolated: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A script has nobody to answer: untrusted project settings just stay off."""
    extensions = isolated / ".loom" / "extensions"
    extensions.mkdir(parents=True)
    (extensions / "x.py").write_text("def register(api):\n    raise SystemExit('loaded!')\n")
    monkeypatch.setattr("sys.stdin", type("Tty", (io.StringIO,), {"isatty": lambda self: True})())
    monkeypatch.setattr("builtins.input", lambda *_: pytest.fail("asked"))
    monkeypatch.setattr(
        "loom.app.create_provider", lambda *a, **kw: FakeProvider([[TextBlock("hi")]])
    )
    assert cli.main(["-p", "x"]) == 0


# --- tutorial -------------------------------------------------------------------------


class NoInput:
    async def read(self) -> str | None:
        return None


class Lines:
    def __init__(self, *lines: str) -> None:
        self.lines = list(lines)

    async def read(self) -> str | None:
        return self.lines.pop(0) if self.lines else None


async def ui_output(app: App, source) -> str:
    console = Console(file=io.StringIO(), width=120, highlight=False)
    await app.start()
    try:
        await InlineUI(app, console=console, source=source).run()
    finally:
        await app.aclose()
    assert isinstance(console.file, io.StringIO)
    return console.file.getvalue()


async def test_first_run_hint_only_the_first_time(tmp_path: Path) -> None:
    assert "/tutorial" in await ui_output(make_app(tmp_path, []), NoInput())
    assert "/tutorial" not in await ui_output(make_app(tmp_path, []), NoInput())


async def test_tutorial_prints_the_steps(tmp_path: Path) -> None:
    output = await ui_output(make_app(tmp_path, []), Lines("/tutorial"))
    assert "Using the demo tools" in output


async def test_tutorial_without_the_demo_package(tmp_path: Path) -> None:
    app = make_app(tmp_path, [])
    app.broker.packages.pop("demo")
    output = await ui_output(app, Lines("/tutorial"))
    assert "loom-tools-demo" in output and "Using the demo tools" not in output


def test_tutorial_only_names_real_commands() -> None:
    commands = builtin_commands()
    for name in set(re.findall(r"(?<![\w/])/(\w+)", TUTORIAL)) - {"checkout"}:  # a URL path
        assert commands.get(name) is not None, name


# --- extension API ---------------------------------------------------------------------


def test_extension_env_and_notices(tmp_path: Path) -> None:
    source = (
        "def register(api):\n"
        "    api.notice(api.package_config('demo')['greeting'])\n"
        "    api.notice(str(api.secret('nope')))\n"
    )
    path = tmp_path / "hello.py"
    path.write_text(source)
    env = ExtensionEnv(packages={"demo": {"greeting": "hi"}})
    extensions, notices = load_extensions([path], env)
    assert notices == []
    assert extensions.notices == ["Extension hello: hi", "Extension hello: None"]


def test_installed_extension_failures_are_notices(monkeypatch: pytest.MonkeyPatch) -> None:
    class Entry:
        name = "broken"

        def load(self):
            raise ImportError("no such thing")

    monkeypatch.setattr("loom.extensions.loader.entry_points", lambda group: [Entry()])
    extensions, notices = load_extensions([], installed=True)
    assert extensions.names == []
    assert notices == ["Extension broken failed to load: ImportError: no such thing"]


def test_a_file_cant_shadow_an_installed_extension(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Entry:
        name = "mine"

        def load(self):
            return lambda api: None

    monkeypatch.setattr("loom.extensions.loader.entry_points", lambda group: [Entry()])
    path = tmp_path / "mine.py"
    path.write_text("def register(api): pass\n")
    extensions, notices = load_extensions([path], installed=True)
    assert extensions.names == ["mine"]
    assert "an extension named 'mine' is loaded" in notices[0]


async def test_shutdown_hook_runs_once_on_close(tmp_path: Path) -> None:
    calls: list[object] = []
    api = ExtensionAPI("x")
    api.on("shutdown", calls.append)
    extensions = Extensions()
    extensions.add(api)
    ws = tmp_path / "ws"
    ws.mkdir()
    app = App(
        ws,
        LoomConfig(),
        provider_factory=lambda p: FakeProvider([]),
        home=tmp_path / "home",
        packages=[],
        extensions=extensions,
    )
    await app.start()
    await app.new_session()  # switching sessions keeps process-wide resources
    assert calls == []
    await app.aclose()
    assert calls == [None]
