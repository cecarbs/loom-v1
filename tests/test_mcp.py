"""The loom-mcp extension: MCP servers as kernel tools (docs/08 "MCP adapter", M11)."""

import asyncio
import inspect
import socket
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from loom_mcp.client import McpManager
from loom_mcp.config import ServerConfig
from loom_mcp.tools import McpToolError, effect_of, identifier, package_for, result_value
from pydantic import ValidationError

from loom.app import App
from loom.broker.redact import Redactor
from loom.broker.secrets import SecretStore
from loom.config import LoomConfig, load_config
from loom.core.messages import TextBlock, ToolResultBlock, ToolUseBlock
from loom.extensions.loader import load_extensions
from loom.providers.fake import FakeProvider, Script

SERVER = Path(__file__).parent / "fixtures" / "mcp_server.py"
STDIO = {"command": sys.executable, "args": [str(SERVER)]}


def config(servers: dict[str, Any], mode: str = "yolo") -> LoomConfig:
    return LoomConfig.model_validate(
        {"packages": {"mcp": {"servers": servers}}, "approvals": {"mode": mode}}
    )


def make_app(
    tmp_path: Path,
    script: list[Script],
    servers: dict[str, Any],
    *,
    mode: str = "yolo",
    secrets: SecretStore | None = None,
) -> App:
    ws = tmp_path / "ws"
    ws.mkdir(exist_ok=True)
    provider = FakeProvider(script)
    return App(
        ws,
        config(servers, mode),
        provider_factory=lambda p: provider,
        home=tmp_path / "home",
        secrets=secrets or SecretStore({}, redactor=Redactor()),
    )


def result_texts(app: App) -> list[str]:
    return [
        str(block.content)
        for message in app.agent.history
        for block in message.content
        if isinstance(block, ToolResultBlock)
    ]


async def run_code(
    tmp_path: Path, code: str, servers: dict[str, Any], **kw: Any
) -> tuple[App, str]:
    app = make_app(
        tmp_path,
        [[ToolUseBlock("e1", "execute", {"code": code})], [TextBlock("ok")]],
        servers,
        **kw,
    )
    await app.start()
    try:
        await app.submit("go")
    finally:
        await app.aclose()
    return app, result_texts(app)[0]


# --- acceptance ----------------------------------------------------------------------


async def test_stdio_tools_are_searchable_and_callable_from_the_kernel(tmp_path: Path) -> None:
    """Acceptance: a stdio server's tools appear in search_tools and work from the kernel."""
    script: list[Script] = [
        [ToolUseBlock("s1", "search_tools", {"query": "add"})],
        [
            ToolUseBlock(
                "e1",
                "execute",
                {
                    "code": "print(mcp_t.add(2, b=3))\n"
                    "data = mcp_t.rows()\n"
                    "print(data[0]['host'])\n"
                    "print(mcp_t.make_note(text='hi'))\n"
                    "print(mcp_t.add(5))"
                },
            )
        ],
        [TextBlock("done")],
    ]
    app = make_app(tmp_path, script, {"t": STDIO})
    await app.start()
    try:
        assert app.extensions.names == ["mcp"]
        assert app.notices == []
        await app.submit("go")
    finally:
        await app.aclose()
    found, out = result_texts(app)
    assert "mcp_t.add(a: int, b: int = 1)" in found and "[read]" in found
    assert "Add two numbers." in found and "Second paragraph" not in found
    assert out.split() == ["5", "web-3", "noted:", "hi", "6"]
    calls = [r for r in app.agent.log.read() if r["kind"] == "broker_call"]
    assert [c["tool"] for c in calls][:2] == ["mcp_t.add", "mcp_t.rows"]
    assert all(c["ok"] for c in calls)


async def test_annotations_drive_approvals(tmp_path: Path) -> None:
    """In read-only mode, read tools run and the others are denied by the policy."""
    app = make_app(tmp_path, [], {"t": STDIO}, mode="read-only")
    await app.start()
    try:
        assert (await app.broker.call("mcp_t.add", {"a": 1})).value == "2"
        note = await app.broker.call("mcp_t.make_note", {"text": "x"})
        wipe = await app.broker.call("mcp_t.wipe", {})
    finally:
        await app.aclose()
    assert note.error_type == wipe.error_type == "PermissionDenied"
    assert "(write) is not allowed" in note.message
    assert "(destructive) is not allowed" in wipe.message


async def test_server_errors_raise_in_the_kernel(tmp_path: Path) -> None:
    code = "try:\n    mcp_t.boom()\nexcept ToolError as e:\n    print('caught', e)"
    _, out = await run_code(tmp_path, code, {"t": STDIO})
    assert "caught mcp_t.boom: McpToolError: Error executing tool boom" in out


async def test_a_dead_server_is_an_error_not_a_hang(tmp_path: Path) -> None:
    code = (
        "for f in (mcp_t.crash, lambda: mcp_t.add(1)):\n"
        "    try:\n"
        "        print(f())\n"
        "    except ToolError as e:\n"
        "        print('caught', e)"
    )
    start = time.monotonic()
    _, out = await run_code(tmp_path, code, {"t": STDIO})
    assert out.count("caught") == 2
    assert time.monotonic() - start < 30


async def test_broker_timeout_cancels_the_call_on_the_other_loop(tmp_path: Path) -> None:
    manager = McpManager({"t": ServerConfig(**STDIO)}, lambda n: None, tmp_path)
    packages, notices = await asyncio.to_thread(manager.start)
    assert notices == []
    try:
        task = asyncio.create_task(manager.call("t", "slow", {}))
        await asyncio.sleep(0.2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert await manager.call("t", "add", {"a": 1}) == "2"  # the connection still works
    finally:
        await manager.aclose()
    assert packages[0].name == "mcp_t"


async def test_secrets_reach_the_server_and_nothing_else_of_ours_does(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LOOM_SECRET_OTHER", "other-value")
    monkeypatch.setenv("SOME_HOST_VAR", "host-value")
    secrets = SecretStore(
        {"LOOM_SECRET_TOKEN": "s3cret-value", "LOOM_SECRET_OTHER": "other-value"},
        redactor=Redactor(),
    )
    server = {**STDIO, "env": {"TOKEN": "secret:token"}}
    code = (
        "print(mcp_t.env_var('TOKEN'))\n"
        "print(mcp_t.env_var('LOOM_SECRET_OTHER'))\n"
        "print(mcp_t.env_var('SOME_HOST_VAR'))"
    )
    _, out = await run_code(tmp_path, code, {"t": server}, secrets=secrets)
    assert "s3cret-value" not in out  # the broker's redactor knows it
    assert out.count("<unset>") == 2


def test_a_failing_server_is_a_notice_and_the_others_still_load(tmp_path: Path) -> None:
    app = make_app(
        tmp_path,
        [],
        {
            "good": STDIO,
            "nope": {"command": str(tmp_path / "does-not-exist")},
            "exits": {"command": sys.executable, "args": ["-c", "pass"], "startup_timeout_s": 5},
            "nosecret": {**STDIO, "env": {"X": "secret:missing"}},
            "typo": {"comand": "x"},
        },
    )
    text = "\n".join(app.notices)
    assert "'nope' failed to connect" in text
    assert "'exits'" in text
    assert "secret 'missing' is not set" in text
    assert "'typo' skipped" in text
    assert "mcp_good" in app.broker.packages
    assert not {"mcp_nope", "mcp_exits", "mcp_nosecret", "mcp_typo"} & set(app.broker.packages)


def test_startup_timeout(tmp_path: Path) -> None:
    hang = {
        "command": sys.executable,
        "args": ["-c", "import time; time.sleep(60)"],
        "startup_timeout_s": 0.5,
    }
    start = time.monotonic()
    app = make_app(tmp_path, [], {"h": hang})
    assert time.monotonic() - start < 10
    assert any("didn't connect in 0.5s" in n for n in app.notices)


async def test_server_stderr_goes_to_a_log_file(tmp_path: Path) -> None:
    await run_code(tmp_path, "1", {"my-server": STDIO})
    log = tmp_path / "home" / "logs" / "mcp-my_server.log"
    assert "fixture server stderr line" in log.read_text()


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def test_streamable_http(tmp_path: Path) -> None:
    port = free_port()
    process = subprocess.Popen(
        [sys.executable, str(SERVER), "http", str(port)],
        stderr=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
    )
    try:
        for _ in range(100):
            with socket.socket() as s:
                if s.connect_ex(("127.0.0.1", port)) == 0:
                    break
            await asyncio.sleep(0.1)
        server = {"url": f"http://127.0.0.1:{port}/mcp", "headers": {"X-Token": "secret:token"}}
        secrets = SecretStore({"LOOM_SECRET_TOKEN": "abc"}, redactor=Redactor())
        _, out = await run_code(tmp_path, "print(mcp_h.add(2, 2))", {"h": server}, secrets=secrets)
        assert "4" in out
    finally:
        process.terminate()
        process.wait()


def test_untrusted_project_config_cant_start_servers(tmp_path: Path) -> None:
    """Invariant 9: the config says what runs, so a cloned repo can't set it."""
    project = tmp_path / ".loom" / "config.toml"
    project.parent.mkdir()
    project.write_text('[packages.mcp.servers.evil]\ncommand = "/bin/evil"\n')
    user = tmp_path / "user.toml"
    untrusted = load_config(tmp_path, user_path=user)
    assert untrusted.config.packages == {}
    assert any("packages" in n for n in untrusted.notices)
    trusted = load_config(tmp_path, user_path=user, trusted=True)
    assert "evil" in trusted.config.packages["mcp"]["servers"]


def test_no_servers_means_no_connections() -> None:
    extensions, notices = load_extensions([], installed=True)
    assert extensions.names == ["mcp"] and notices == [] and extensions.packages == []
    assert extensions.hooks == []


# --- mapping --------------------------------------------------------------------------


def tool(name: str = "t", schema: dict | None = None, annotations: Any = None) -> Any:
    return SimpleNamespace(
        name=name,
        title=None,
        description="Does a thing.",
        input_schema=schema or {"type": "object"},
        annotations=annotations,
    )


def hints(read: bool | None = None, destructive: bool | None = None) -> Any:
    return SimpleNamespace(read_only_hint=read, destructive_hint=destructive)


def test_effect_mapping() -> None:
    assert effect_of(None) == "write"
    assert effect_of(hints()) == "write"
    assert effect_of(hints(read=True)) == "read"
    assert effect_of(hints(destructive=True)) == "destructive"
    assert effect_of(hints(read=False, destructive=False)) == "write"
    assert effect_of(hints(read=True, destructive=True)) == "destructive"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("get-issue", "get_issue"),
        ("a.b/c", "a_b_c"),
        ("_private", "private"),
        ("2fa", "x2fa"),
        ("class", "class_"),
        ("---", "x"),
    ],
)
def test_identifier(raw: str, expected: str) -> None:
    assert identifier(raw) == expected


async def test_schema_becomes_a_signature_and_args_map_back() -> None:
    seen: list[tuple[str, dict]] = []

    async def call(name: str, args: dict[str, Any]) -> str:
        seen.append((name, args))
        return "ok"

    schema = {
        "type": "object",
        "required": ["repo-name", "class"],
        "properties": {
            "limit": {"type": "integer", "default": 10, "description": "Max rows."},
            "state": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "repo-name": {"type": "string"},
            "class": {"type": "string"},
            "json": {"type": "object"},
            "ctx": {"type": ["string", "null"]},
            "any": {},
        },
    }
    package, notices = package_for("gh", "", [tool("list-issues", schema)], call)
    assert notices == []
    spec = package.tools["list_issues"]
    assert list(spec.signature.parameters)[:2] == ["repo_name", "class_"]
    assert spec.signature.parameters["limit"].default == 10
    assert "limit: Max rows." in spec.doc
    assert spec.display().startswith("mcp_gh.list_issues(repo_name: str, class_: str, limit: int")
    package.catalog_entry()  # must be JSON for the kernel
    model = spec.args_model.model_validate({"repo_name": "r", "class_": "c", "json_": {"a": 1}})
    kwargs = {k: getattr(model, k) for k in type(model).model_fields}
    assert await spec.func(object(), **kwargs) == "ok"
    assert seen == [
        ("list-issues", {"repo-name": "r", "class": "c", "limit": 10, "json": {"a": 1}})
    ]
    assert inspect.iscoroutinefunction(spec.func)


def test_name_clashes_are_notices() -> None:
    async def call(name: str, args: dict[str, Any]) -> None: ...

    package, notices = package_for("s", "", [tool("a-b"), tool("a_b")], call)
    assert list(package.tools) == ["a_b"]
    assert len(notices) == 1 and "'a_b'" in notices[0]


def block(kind: str, **fields: Any) -> Any:
    return SimpleNamespace(type=kind, **fields)


def result(*content: Any, structured: Any = None, error: bool = False) -> Any:
    return SimpleNamespace(content=list(content), structured_content=structured, is_error=error)


def test_result_values() -> None:
    assert result_value(result()) is None
    assert result_value(result(block("text", text="hello"))) == "hello"
    assert result_value(result(block("text", text='{"a": 1}'))) == {"a": 1}
    assert result_value(result(block("text", text="[not json"))) == "[not json"
    assert result_value(result(block("text", text="x"), structured={"a": 2})) == {"a": 2}
    two = result(block("text", text="a"), block("image", mime_type="image/png", data="QQ=="))
    assert result_value(two) == ["a", {"type": "image", "mime_type": "image/png", "data": "QQ=="}]
    with pytest.raises(McpToolError, match="nope"):
        result_value(result(block("text", text="nope"), error=True))


@pytest.mark.parametrize(
    "raw",
    [
        {},
        {"command": "x", "url": "http://y"},
        {"command": "x", "headers": {"a": "b"}},
        {"url": "http://y", "args": ["z"]},
        {"command": "x", "unknown": 1},
    ],
)
def test_server_config_errors(raw: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        ServerConfig.model_validate(raw)


async def test_shutdown_closes_the_servers(tmp_path: Path) -> None:
    manager = McpManager({"t": ServerConfig(**STDIO)}, lambda n: None, tmp_path)
    await asyncio.to_thread(manager.start)
    await manager.aclose()
    assert manager._clients == {}
