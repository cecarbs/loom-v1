"""M6: resume, compaction, checkpoints and /undo."""

import io
import os
import shutil
from pathlib import Path

import pytest
from rich.console import Console

from loom.app import App
from loom.cli import build_parser
from loom.config import LoomConfig
from loom.core import compaction
from loom.core import events as ev
from loom.core.agent import RESUMED_NOTE
from loom.core.events import Usage
from loom.core.messages import (
    Message,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    message_to_dict,
)
from loom.kernel.protocol import VarInfo
from loom.providers.base import ProviderError
from loom.providers.fake import FakeProvider, FakeResponse, Script
from loom.session.checkpoints import MAX_FILE_BYTES
from loom.session.store import SessionError, find_session, list_sessions
from loom.tui.approvals import DenyAll
from loom.tui.commands import CommandContext, builtin_commands
from loom.tui.render import Renderer

needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")


def yolo() -> LoomConfig:
    return LoomConfig.model_validate({"approvals": {"mode": "yolo"}})


def make_app(
    tmp_path: Path,
    script: list[Script],
    config: LoomConfig | None = None,
    **kwargs,
) -> tuple[App, FakeProvider, list]:
    workspace = tmp_path / "ws"
    workspace.mkdir(exist_ok=True)
    provider = FakeProvider(script)
    app = App(
        workspace,
        config or yolo(),
        provider_factory=lambda p: provider,
        home=tmp_path / "home",
        packages=[],
        **kwargs,
    )
    events: list = []

    async def record(e: ev.AgentEvent) -> None:
        events.append(e)

    app.bus.subscribe(record)
    return app, provider, events


def dicts(messages: list[Message]) -> list[dict]:
    return [message_to_dict(m) for m in messages]


def big(tokens: int) -> Usage:
    return Usage(input_tokens=tokens, output_tokens=20)


# --- resume ---------------------------------------------------------------------------


async def test_resume_rebuilds_identical_history(tmp_path: Path) -> None:
    app, _, _ = make_app(
        tmp_path,
        [
            [TextBlock("writing"), ToolUseBlock("t1", "write", {"path": "a.txt", "content": "hi"})],
            [TextBlock("wrote it")],
            [TextBlock("second answer")],
        ],
    )
    await app.submit("create a.txt")
    await app.submit("anything else?")
    history = dicts(app.agent.history)
    turns, usage, session_id = app.agent.turns, app.agent.usage, app.session_id
    await app.aclose()

    resumed, provider, _ = make_app(tmp_path, [[TextBlock("third")]], resume_latest=True)
    assert resumed.resumed and resumed.session_id == session_id
    assert dicts(resumed.agent.history) == history
    assert resumed.agent.turns == turns == 2
    assert resumed.agent.usage == usage
    await resumed.submit("and now?")
    request = provider.requests[0].messages
    assert dicts(request[: len(history)]) == history
    assert request[-1].content == [TextBlock("and now?")]  # no kernel was used: no note
    await resumed.aclose()

    log = (resumed.session_paths.log).read_text()
    assert '"kind": "session_resume"' in log


async def test_resume_tells_the_model_the_kernel_is_fresh(tmp_path: Path) -> None:
    app, _, _ = make_app(
        tmp_path,
        [[ToolUseBlock("t1", "execute", {"code": "x = 41"})], [TextBlock("set")]],
    )
    await app.submit("set x")
    await app.aclose()

    resumed, provider, _ = make_app(tmp_path, [[TextBlock("gone")]], resume=app.session_id[:13])
    await resumed.submit("what is x?")
    assert provider.requests[0].messages[-1].content[-1] == TextBlock(RESUMED_NOTE)
    await resumed.aclose()


async def test_resume_rebuilds_history_after_compaction(tmp_path: Path) -> None:
    script: list[Script] = [[TextBlock(f"answer {i}")] for i in range(4)]
    script += [[TextBlock("SUMMARY")], [TextBlock("answer 4")]]
    config = LoomConfig.model_validate(
        {"approvals": {"mode": "yolo"}, "context": {"keep_recent_turns": 1}}
    )
    app, _, _ = make_app(tmp_path, script, config)
    for i in range(4):
        await app.submit(f"q{i}")
    assert await app.compact()
    await app.submit("q4")
    live = dicts(app.agent.history)
    assert live[0]["meta"] == {"compaction": True}
    await app.aclose()

    resumed, _, _ = make_app(tmp_path, [], config, resume_latest=True)
    assert dicts(resumed.agent.history) == live
    assert resumed.agent.turns == 5
    await resumed.aclose()


async def test_session_lock_and_lookup(tmp_path: Path) -> None:
    app, _, _ = make_app(tmp_path, [[TextBlock("ok")]])
    await app.submit("first prompt\nsecond line")
    workspace, home = app.workspace, tmp_path / "home"

    [summary] = list_sessions(workspace, home)
    assert summary.id == app.session_id and summary.title == "first prompt"
    assert find_session(workspace, app.session_id[:10], home).id == app.session_id
    with pytest.raises(SessionError, match="no session"):
        find_session(workspace, "1999", home)
    lock = app.session_paths.lock
    assert lock.read_text() == str(os.getpid())
    await app.aclose()  # releases the lock
    assert not lock.exists()

    lock.write_text(str(os.getppid()))  # another live loom holds it
    with pytest.raises(SessionError, match="in use"):
        make_app(tmp_path, [], resume=app.session_id)
    lock.write_text("999999")  # a dead one: taken over
    other, _, _ = make_app(tmp_path, [], resume=app.session_id)
    await other.aclose()


async def test_resume_command_switches_session(tmp_path: Path) -> None:
    app, _, _ = make_app(tmp_path, [[TextBlock("one")], [TextBlock("two")]])
    await app.submit("in the first session")
    first = app.session_id
    await app.new_session()
    await app.submit("in the second session")

    out = io.StringIO()
    console = Console(file=out, width=100, color_system=None)
    renderer = Renderer(console, DenyAll(console), app.bus)
    ctx = CommandContext(app, console, renderer, builtin_commands())
    await ctx.commands.run(ctx, "/sessions")
    assert first in out.getvalue() and "in the first session" in out.getvalue()
    await ctx.commands.run(ctx, f"/resume {first}")
    assert app.session_id == first
    assert app.agent.history[0].text() == "in the first session"
    assert "Resumed session" in out.getvalue() and "one" in out.getvalue()
    await app.aclose()


def test_cli_resume_flags() -> None:
    parser = build_parser()
    assert parser.parse_args(["-c"]).continue_
    assert parser.parse_args(["--resume", "2026"]).resume == "2026"
    assert parser.parse_args(["sessions"]).command == "sessions"
    with pytest.raises(SystemExit):
        parser.parse_args(["-c", "--resume", "x"])


# --- compaction -----------------------------------------------------------------------


def turn(prompt: str, *tool_ids: str) -> list[Message]:
    msgs = [Message("user", [TextBlock(prompt)])]
    for tid in tool_ids:
        msgs.append(Message("assistant", [ToolUseBlock(tid, "read", {"path": "x"})]))
        msgs.append(Message("user", [ToolResultBlock(tid, "ok")]))
    msgs.append(Message("assistant", [TextBlock("done")]))
    return msgs


def test_cut_point_keeps_recent_turns_and_tool_pairs() -> None:
    history = turn("a", "t1") + turn("b") + turn("c", "t2", "t3")
    assert compaction.cut_point(history, 2, mid_turn=False) == 4  # start of turn "b"
    assert compaction.cut_point(history, 3, mid_turn=False) is None
    # Inside one long turn: cut before the latest assistant message, never between a
    # tool use and its result.
    long_turn = turn("only", "t1", "t2", "t3")[:-1]  # still running: ends with results
    cut = compaction.cut_point(long_turn, 3, mid_turn=True)
    assert cut == 5 and long_turn[cut].role == "assistant"
    assert isinstance(long_turn[cut + 1].content[0], ToolResultBlock)
    # Only an earlier summary before the cut: nothing to do.
    summary = compaction.summary_message("s", [])
    assert compaction.cut_point([summary, *turn("x")], 1, mid_turn=False) is None
    assert compaction.cut_point([summary, *turn("x")], 0, mid_turn=False) == 3


def test_summary_message_includes_inventory() -> None:
    msg = compaction.summary_message(
        "goal: x", [VarInfo("rows", "list", "len=48211"), VarInfo("n", "int", "")]
    )
    text = msg.text()
    assert text.startswith(compaction.SUMMARY_PREFACE)
    inventory = "Variables available in the kernel: rows (list, len=48211), n (int)"
    assert f"<summary>\ngoal: x\n\n{inventory}\n</summary>" in text


async def test_auto_compaction_at_threshold(tmp_path: Path) -> None:
    window = 1_000_000  # sonnet's; compact_at 0.75 → 750k
    script: list[Script] = [
        [ToolUseBlock("t1", "execute", {"code": "rows = list(range(48211))"})],
        [TextBlock("loaded rows")],
        [TextBlock("answer 2")],
        [TextBlock("answer 3")],
        FakeResponse([TextBlock("answer 4")], usage=big(int(window * 0.8))),
        FakeResponse([TextBlock("THE SUMMARY")], usage=big(1000)),
        [TextBlock("answer 5")],
    ]
    app, provider, events = make_app(tmp_path, script)
    for prompt in ("load rows", "q2", "q3", "q4"):
        await app.submit(prompt)
    before = dicts(app.agent.history)
    assert not any(isinstance(e, ev.Compacted) for e in events)

    await app.submit("q5")
    summary_req, next_req = provider.requests[-2], provider.requests[-1]
    # The summary request is the old history prefix plus the summary prompt.
    assert summary_req.messages[-1].text().startswith("Summarize the conversation")
    head = next_req.messages[0].text()
    assert "<summary>\nTHE SUMMARY" in head
    assert "Variables available in the kernel: rows (list, len=48211)" in head
    # keep_recent_turns = 3: turns q3, q4, q5 follow unchanged.
    kept = [m.text() for m in next_req.messages[1:] if m.role == "user"]
    assert kept == ["q3", "q4", "q5"]
    assert dicts(next_req.messages[1:5]) == before[-4:]
    compacted = [e for e in events if isinstance(e, ev.Compacted)]
    assert len(compacted) == 1 and compacted[0].after_tokens < compacted[0].before_tokens
    log = app.agent.log.read()
    entry = next(r for r in log if r["kind"] == "compaction")
    assert entry["first_kept"] == len(before) - 4 and entry["summary"] == "THE SUMMARY"
    assert any(r.get("purpose") == "compaction" for r in log if r["kind"] == "model_request")
    await app.aclose()


async def test_context_too_long_compacts_then_retries(tmp_path: Path) -> None:
    script: list[Script] = [[TextBlock(f"a{i}")] for i in range(4)]
    script += [
        ProviderError("context_too_long", "prompt is too long"),
        [TextBlock("SUMMARY")],
        [TextBlock("after retry")],
    ]
    app, _, events = make_app(tmp_path, script)
    for i in range(5):
        await app.submit(f"q{i}")
    assert app.agent.history[-1].text() == "after retry"
    assert app.agent.history[0].meta.get("compaction")
    assert not any(isinstance(e, ev.ErrorEvent) for e in events)
    await app.aclose()


async def test_manual_compact_with_instructions(tmp_path: Path) -> None:
    script: list[Script] = [[TextBlock("a0")], [TextBlock("a1")], [TextBlock("S")]]
    config = LoomConfig.model_validate(
        {"approvals": {"mode": "yolo"}, "context": {"keep_recent_turns": 1}}
    )
    app, provider, _ = make_app(tmp_path, script, config)
    assert not await app.compact()  # nothing yet
    for i in range(2):
        await app.submit(f"q{i}")
    assert await app.compact("focus on the parser")
    assert "focus on the parser" in provider.requests[-1].messages[-1].text()
    assert [m.role for m in app.agent.history] == ["user", "user", "assistant"]
    await app.aclose()


# --- checkpoints and undo -------------------------------------------------------------


@needs_git
async def test_undo_restores_edited_file_and_removes_created_one(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "a.txt").write_text("one\n")
    (workspace / ".gitignore").write_text("ignored.log\n")
    (workspace / "ignored.log").write_text("keep me")
    app, _, _ = make_app(
        tmp_path,
        [
            [
                ToolUseBlock("t1", "read", {"path": "a.txt"}),
            ],
            [
                ToolUseBlock(
                    "t2", "edit", {"path": "a.txt", "old_string": "one", "new_string": "two"}
                ),
                ToolUseBlock("t3", "write", {"path": "sub/new.txt", "content": "new"}),
                ToolUseBlock("t4", "bash", {"command": "echo more >> ignored.log"}),
            ],
            [TextBlock("done")],
        ],
    )
    await app.submit("change things")
    assert (workspace / "a.txt").read_text() == "two\n"
    assert (workspace / "sub" / "new.txt").exists()
    log = app.agent.log.read()
    assert len([r for r in log if r["kind"] == "checkpoint"]) == 1  # once per turn

    [entry] = await app.undo_entries()
    assert entry.checkpoint.turn == 1 and entry.checkpoint.prompt == "change things"
    assert {(c.status, c.path) for c in entry.changes} == {("M", "a.txt"), ("A", "sub/new.txt")}
    result = await app.undo(entry)
    assert (workspace / "a.txt").read_text() == "one\n"
    assert not (workspace / "sub").exists()
    assert (workspace / "ignored.log").read_text() == "keep memore\n"  # ignored: not covered
    assert result.restored == ["a.txt"] and result.removed == ["sub/new.txt"]
    assert "ran /undo" in app.agent.pending_notes[0]
    assert app.agent.log.read()[-1]["kind"] == "note"

    # The undo itself can be undone.
    entries = await app.undo_entries()
    assert entries[0].checkpoint.prompt == "before /undo"
    await app.undo(entries[0])
    assert (workspace / "a.txt").read_text() == "two\n"
    assert (workspace / "sub" / "new.txt").read_text() == "new"
    await app.aclose()


@needs_git
async def test_checkpoints_skip_big_files_and_survive_resume(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    app, _, _ = make_app(
        tmp_path,
        [
            [ToolUseBlock("t1", "bash", {"command": "echo x > small.txt"})],
            [TextBlock("ok")],
        ],
    )
    (workspace / "huge.bin").write_bytes(b"\0" * (MAX_FILE_BYTES + 1))
    await app.submit("make a file")
    await app.aclose()

    resumed, _, _ = make_app(tmp_path, [], resume_latest=True)
    [entry] = await resumed.undo_entries()
    assert [c.path for c in entry.changes] == ["small.txt"]
    await resumed.undo(entry)
    assert not (workspace / "small.txt").exists()
    await resumed.aclose()
    # The undo note survives a restart and goes out with the next prompt, once.
    script: list[Script] = [[TextBlock("ok")], [TextBlock("ok")]]
    again, provider, _ = make_app(tmp_path, script, resume_latest=True)
    await again.submit("next")
    await again.submit("and next")
    notes = [b for m in provider.requests[1].messages for b in m.content if "ran /undo" in str(b)]
    assert len(notes) == 1
    resumed = again
    assert (workspace / "huge.bin").stat().st_size == MAX_FILE_BYTES + 1  # untouched
    # The shadow repo lives under loom's home, not in the workspace.
    assert not (workspace / ".git").exists()
    assert list((tmp_path / "home" / "checkpoints").glob("*.git"))
    await resumed.aclose()


@needs_git
async def test_undo_command_asks_which_checkpoint(tmp_path: Path) -> None:
    app, _, _ = make_app(
        tmp_path,
        [
            [ToolUseBlock("t1", "write", {"path": "f.txt", "content": "1"})],
            [TextBlock("ok")],
        ],
    )
    await app.submit("create f")
    out = io.StringIO()
    console = Console(file=out, width=120, color_system=None)
    asked: list[str] = []

    async def ask(prompt: str) -> str | None:
        asked.append(prompt)
        return ""  # Enter = the latest

    ctx = CommandContext(
        app,
        console,
        Renderer(console, DenyAll(console), app.bus),
        builtin_commands(),
        ask=ask,
    )
    await ctx.commands.run(ctx, "/undo")
    assert asked and "f.txt (new)" in out.getvalue()
    assert not (app.workspace / "f.txt").exists()
    assert "not kernel variables" in out.getvalue()
    await app.aclose()


def test_log_listeners_get_redacted_records_and_cannot_break_logging(tmp_path) -> None:
    from loom.broker.redact import Redactor
    from loom.session.store import SessionLog

    redactor = Redactor()
    redactor.add("hunter2")
    log = SessionLog(tmp_path / "session.jsonl", redactor)
    seen: list[dict] = []

    def broken(record: dict) -> None:
        raise RuntimeError("exporter bug")

    log.listeners += [broken, seen.append]
    log.append("note", text="password hunter2")
    assert [r["text"] for r in seen] == [log.read()[0]["text"]]
    assert "hunter2" not in seen[0]["text"]
