"""Kernel tool calls through the broker (docs/04): stubs, pipeline, approvals, secrets."""

import asyncio
import json
from collections.abc import AsyncIterator, Sequence
from pathlib import Path

import pytest
from loom_tools_demo import package as demo_package

from loom.app import build_agent
from loom.broker import PackageError, Secret, ToolPackage
from loom.broker.redact import REDACTED, Redactor
from loom.broker.secrets import SecretStore
from loom.config import LoomConfig
from loom.core import events as ev
from loom.core.events import EventBus
from loom.core.messages import TextBlock, ToolResultBlock, ToolUseBlock
from loom.providers.fake import FakeProvider, Script

SECRET = "s3cr3t-value-0042"


def make_secret_package() -> ToolPackage:
    pkg = ToolPackage(name="vault", description="Test package with a secret.", secrets={
        "token": Secret("vault_token"),
    })  # fmt: skip

    @pkg.tool(effect="read")
    def leak(ctx) -> dict:
        """Returns its own credential (a badly behaved package)."""
        return {"token": ctx.secret("token"), "nested": [f"Bearer {ctx.secret('token')}"]}

    @pkg.tool(effect="read")
    def needs_missing(ctx) -> str:
        return ctx.secret("token")

    @pkg.tool(effect="read")
    async def slow(ctx, seconds: float) -> str:
        await asyncio.sleep(seconds)
        return "done"

    @pkg.tool(effect="read")
    def echo(ctx, data: bytes, when: str = "x") -> dict:
        import datetime

        return {"data": data, "at": datetime.date(2026, 1, 15), "items": {1, 2}}

    @pkg.tool(effect="read")
    def boom(ctx, n: int) -> int:
        raise ValueError(f"bad n={n} with {ctx.secret('token')}")

    return pkg


class Harness:
    def __init__(
        self,
        tmp_path: Path,
        script: list[Script],
        mode: str = "ask",
        packages: Sequence[ToolPackage] = (),
        environ: dict[str, str] | None = None,
        **config,
    ) -> None:
        self.workspace = tmp_path / "ws"
        self.workspace.mkdir(exist_ok=True)
        self.bus = EventBus()
        self.events: list[ev.AgentEvent] = []
        self.bus.subscribe(self._record)
        self.provider = FakeProvider(script)
        self.secrets = SecretStore(environ or {}, redactor=Redactor())
        # Backend `none` asks before every execute; allow it so the tests see only the
        # broker's approvals.
        approvals = config.pop("approvals", {"mode": mode})
        approvals.setdefault("overrides", {}).setdefault("execute", "allow")
        cfg = LoomConfig.model_validate({"approvals": approvals, **config})
        self.agent = build_agent(
            self.workspace,
            cfg,
            self.bus,
            provider=self.provider,
            home=tmp_path / "home",
            secrets=self.secrets,
            packages=list(packages) or [demo_package],
        )
        self.answers: list[ev.Decision] = []
        self.asked: list[ev.ApprovalRequested] = []

    async def _record(self, event: ev.AgentEvent) -> None:
        self.events.append(event)
        if isinstance(event, ev.ApprovalRequested) and self.answers:
            self.asked.append(event)
            self.bus.respond(event.request_id, self.answers.pop(0))

    def result(self, i: int = -1) -> ToolResultBlock:
        return [
            b
            for m in self.agent.history
            if m.role == "user"
            for b in m.content
            if isinstance(b, ToolResultBlock)
        ][i]

    def text(self, i: int = -1) -> str:
        content = self.result(i).content
        assert isinstance(content, str)
        return content

    def log(self, kind: str) -> list[dict]:
        return [r for r in self.agent.log.read() if r["kind"] == kind]

    async def close(self) -> None:
        assert self.agent.tool_ctx.kernel is not None
        await self.agent.tool_ctx.kernel.shutdown()


def execute(id_: str, code: str, **kw) -> ToolUseBlock:
    return ToolUseBlock(id_, "execute", {"code": code, **kw})


@pytest.fixture
async def make(tmp_path: Path) -> AsyncIterator:
    made: list[Harness] = []

    def factory(script: list[Script], **kw) -> Harness:
        h = Harness(tmp_path, script, **kw)
        made.append(h)
        return h

    yield factory
    for h in made:
        await h.close()


# --- acceptance ---------------------------------------------------------------------


async def test_ten_thousand_rows_stay_in_the_kernel(make) -> None:
    h = make(
        [
            [
                execute(
                    "e1",
                    "rows = demo.search_logs(limit=10_000)\nprint(len(rows))\nrows",
                )
            ],
            [execute("e2", "from collections import Counter\nCounter(r['service'] for r in rows)")],
            [TextBlock("ok")],
        ]
    )
    await h.agent.run_turn("go")
    first = h.text(0)
    assert first.startswith("10000\n[{")
    assert len(first) < 12_000  # repr limited by the runner; nothing spilled
    assert h.result(0).is_error is False
    assert "Counter({" in h.text(1)
    [call] = h.log("broker_call")
    assert call["tool"] == "demo.search_logs" and call["ok"] and call["result_chars"] > 1_000_000
    assert call["decision"] == {"action": "allow_once", "by": "policy"}


async def test_write_tool_asks_and_denial_is_permission_denied(make) -> None:
    code = (
        "try:\n"
        "    demo.create_ticket(title='Checkout 500s on web-3', priority='P1')\n"
        "except PermissionDenied as e:\n"
        "    print('denied:', e)\n"
        "len(demo.search_tickets())"
    )
    h = make([[execute("e1", code)], [TextBlock("ok")]])
    h.answers = [ev.Decision("deny", "not now")]
    await h.agent.run_turn("go")
    [request] = h.asked
    assert request.tool == "demo.create_ticket" and request.effect == "write"
    assert 'title = "Checkout 500s on web-3"' in request.args_preview
    assert 'priority = "P1"' in request.args_preview
    assert h.text() == "denied: User denied demo.create_ticket: not now\n3"
    call = next(c for c in h.log("broker_call") if c["tool"] == "demo.create_ticket")
    assert call["error"]["type"] == "PermissionDenied"
    assert call["decision"] == {"action": "deny", "by": "user"}


async def test_secret_returned_by_package_is_redacted_and_never_logged(make) -> None:
    h = make(
        [
            [execute("e1", "r = vault.leak()\nprint(r)\nr['token'] == '[REDACTED]'")],
            [execute("e2", "vault.boom(3)")],
            [TextBlock(f"the token is {SECRET}")],  # even the model echoing it gets scrubbed
        ],
        packages=[make_secret_package()],
        environ={"LOOM_SECRET_VAULT_TOKEN": SECRET},
    )
    await h.agent.run_turn("go")
    first = h.text(0)
    assert SECRET not in first and first.endswith("True")
    assert f"Bearer {REDACTED}" in first
    assert "ValueError: bad n=3 with [REDACTED]" in h.text(1)
    raw = h.agent.log.path.read_text()
    assert SECRET not in raw
    assert "[REDACTED]" in raw


# --- stubs in the kernel ------------------------------------------------------------


async def test_stubs_have_signatures_docs_and_modules(make) -> None:
    code = (
        "import inspect, demo as d\n"
        "print(inspect.signature(demo.search_logs))\n"
        "print(demo.search_logs.__doc__.splitlines()[0])\n"
        "print(tools.demo is demo, d is demo, 'tools' in dir(), demo.__doc__)\n"
        "demo.get_ticket('T-101')['title']"
    )
    h = make([[execute("e1", code)], [execute("e2", "len(_out)")], [TextBlock("ok")]])
    await h.agent.run_turn("go")
    lines = h.text(0).splitlines()
    assert lines[0] == (
        "(service: str | None = None, level: str | None = None, contains: str | None = None, "
        "min_status: int | None = None, limit: int = 10000) -> list[dict]"
    )
    assert lines[1].startswith("Return log rows")
    assert lines[2].startswith("True True True Synthetic web logs")
    assert lines[3] == "'Checkout failing for some users'"
    kernel = h.agent.tool_ctx.kernel
    assert kernel is not None
    names = {v.name for v in await kernel.inventory()}
    assert not names & {"demo", "tools", "PermissionDenied", "ToolError"}


async def test_argument_errors(make) -> None:
    code = (
        "for call in (lambda: demo.get_ticket(), lambda: demo.search_logs(limit='many'),\n"
        "             lambda: demo.get_ticket('T-999'), lambda: demo.search_logs(rows=object())):\n"
        "    try:\n"
        "        call()\n"
        "    except Exception as e:\n"
        "        print(type(e).__name__, '|', e)\n"
    )
    h = make([[execute("e1", code)], [TextBlock("ok")]])
    await h.agent.run_turn("go")
    lines = h.text().splitlines()
    assert lines[0].startswith("TypeError | missing a required argument: 'ticket_id'")
    assert lines[1].startswith("InvalidArguments | demo.search_logs: limit:")
    assert lines[2] == "ToolError | demo.get_ticket: KeyError: 'no ticket T-999'"
    assert lines[3].startswith("TypeError | got an unexpected keyword argument 'rows'")


async def test_values_round_trip_bytes_and_dates(make) -> None:
    h = make(
        [[execute("e1", "vault.echo(b'\\x00\\xff', when='now')")], [TextBlock("ok")]],
        packages=[make_secret_package()],
    )
    await h.agent.run_turn("go")
    assert h.text() == "{'data': b'\\x00\\xff', 'at': '2026-01-15', 'items': [1, 2]}"
    assert "runner.py" not in h.text()


async def test_missing_secret_explains_how_to_set_it(make) -> None:
    h = make(
        [[execute("e1", "vault.needs_missing()")], [TextBlock("ok")]],
        packages=[make_secret_package()],
    )
    await h.agent.run_turn("go")
    assert "MissingSecret: Secret 'vault_token' is not set" in h.text()
    assert "loom secrets set vault_token" in h.text()


async def test_broker_call_timeout(make) -> None:
    h = make(
        [[execute("e1", "vault.slow(5)")], [TextBlock("ok")]],
        packages=[make_secret_package()],
        broker={"call_timeout_s": 0.2},
    )
    await h.agent.run_turn("go")
    assert "ToolTimeout: vault.slow timed out after 0.2s" in h.text()


# --- approvals ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("mode", "tool_call", "asks", "allowed"),
    [
        ("read-only", "demo.create_ticket(title='x')", False, False),
        ("ask", "demo.create_ticket(title='x')", True, True),
        ("auto-write", "demo.create_ticket(title='x')", False, True),
        ("auto-write", "demo.delete_ticket('T-101')", True, True),
        ("yolo", "demo.delete_ticket('T-101')", False, True),
        ("read-only", "demo.list_services()", False, True),
    ],
)
async def test_broker_approval_modes(make, mode, tool_call, asks, allowed) -> None:
    code = f"try:\n    {tool_call}\n    print('ran')\nexcept PermissionDenied as e:\n    print(e)"
    h = make([[execute("e1", code)], [TextBlock("ok")]], mode=mode)
    h.answers = [ev.Decision("allow_once")]
    await h.agent.run_turn("go")
    assert bool(h.asked) == asks
    assert (h.text() == "ran\n") == allowed, h.text()
    if not allowed:
        assert f"is not allowed in {mode} mode" in h.text()


async def test_allow_session_and_overrides(make) -> None:
    code = (
        "demo.create_ticket(title='a'); demo.create_ticket(title='b'); demo.delete_ticket('T-101')"
    )
    h = make(
        [[execute("e1", code)], [TextBlock("ok")]],
        approvals={"mode": "ask", "overrides": {"demo.delete_ticket": "allow"}},
    )
    h.answers = [ev.Decision("allow_session")]
    await h.agent.run_turn("go")
    assert len(h.asked) == 1 and h.result().is_error is False
    decisions = [(c["tool"], c["decision"]["by"]) for c in h.log("broker_call")]
    assert decisions == [
        ("demo.create_ticket", "user"),
        ("demo.create_ticket", "session"),
        ("demo.delete_ticket", "override"),
    ]


async def test_execute_clock_pauses_while_approval_is_pending(make) -> None:
    h = make(
        [[execute("e1", "demo.create_ticket(title='x')['id']", timeout_s=1)], [TextBlock("ok")]]
    )

    async def answer_late(event: ev.AgentEvent) -> None:
        if isinstance(event, ev.ApprovalRequested):
            await asyncio.sleep(1.5)  # longer than the execute timeout
            h.bus.respond(event.request_id, ev.Decision("allow_once"))

    h.bus.subscribe(answer_late)
    await h.agent.run_turn("go")
    assert h.text() == "'T-104'"


async def test_interrupt_during_approval_abandons_the_call(make) -> None:
    h = make(
        [
            [execute("e1", "kept = 1\ndemo.create_ticket(title='x')")],
            [execute("e2", "kept")],
            [TextBlock("ok")],
        ]
    )
    asked = asyncio.Event()

    async def hold(event: ev.AgentEvent) -> None:
        if isinstance(event, ev.ApprovalRequested):
            asked.set()  # never answered

    h.bus.subscribe(hold)
    task = asyncio.create_task(h.agent.run_turn("go"))
    await asked.wait()
    task.cancel()
    await task
    approval = h.log("approval")[-1]
    assert approval["by"] == "cancel" and approval["tool"] == "demo.create_ticket"
    kernel = h.agent.tool_ctx.kernel
    assert kernel is not None and kernel.state == "idle"
    await h.agent.run_turn("again")
    assert h.text() == "1"  # same kernel


async def test_tool_call_outside_an_execution_is_refused(make) -> None:
    h = make([])
    kernel = h.agent.tool_ctx.kernel
    assert kernel is not None
    code = (
        "import threading, time\n"
        "box = {}\n"
        "def later():\n"
        "    time.sleep(0.2)\n"
        "    try:\n"
        "        demo.list_services()\n"
        "    except Exception as e:\n"
        "        box['error'] = f'{type(e).__name__}: {e}'\n"
        "threading.Thread(target=later).start()"
    )
    await kernel.execute(code, timeout_s=10)
    await asyncio.sleep(0.6)  # the thread calls while no execution runs
    result = await kernel.execute("box['error']", timeout_s=10)
    assert result.value_repr == ("'ToolError: tool calls only work while an execute call runs'")


# --- search_tools -------------------------------------------------------------------


async def test_search_tools(make) -> None:
    h = make(
        [
            [ToolUseBlock("s1", "search_tools", {"query": ""})],
            [ToolUseBlock("s2", "search_tools", {"query": "ticket create"})],
            [ToolUseBlock("s3", "search_tools", {"query": "kubernetes"})],
            [TextBlock("ok")],
        ]
    )
    await h.agent.run_turn("go")
    assert "demo — Synthetic web logs" in h.text(0) and "(7 tools)" in h.text(0)
    best = h.text(1).splitlines()[0]
    assert best.startswith("demo.create_ticket(title: str") and best.endswith("[write]")
    assert h.text(1).splitlines()[1].strip() == "Open a new ticket and return it."
    assert "No tools match 'kubernetes'" in h.text(2)


# --- registry -----------------------------------------------------------------------


def test_registry_rejects_bad_definitions() -> None:
    pkg = ToolPackage(name="bad", description="")
    with pytest.raises(PackageError, match=r"\*args"):
        pkg.tool(effect="read")(lambda ctx, *rows: None)
    with pytest.raises(PackageError, match="context"):

        @pkg.tool(effect="read")
        def no_ctx(*, x: int) -> None: ...

    with pytest.raises(PackageError, match="JSON-serializable"):

        @pkg.tool(effect="read")
        def bad_default(ctx, x: object = object()) -> None: ...

    with pytest.raises(PackageError, match="effect"):
        pkg.tool(effect="maybe")  # type: ignore[arg-type]

    @pkg.tool(effect="read")
    def twice(ctx) -> None: ...

    with pytest.raises(PackageError, match="twice"):

        @pkg.tool(effect="read")
        def twice(ctx) -> None: ...

    for name in ("tools", "class", "_private", "has-dash"):
        with pytest.raises(PackageError):
            ToolPackage(name=name, description="")


async def test_unknown_tool_suggests_names(tmp_path: Path) -> None:
    h = Harness(tmp_path, [])
    broker = h.agent.tool_ctx.broker
    assert broker is not None
    reply = await broker.call("demo.search_log", {})
    assert reply.error_type == "ToolNotFound" and "demo.search_logs" in reply.message
    bad = await broker.call("demo.list_services", ["not", "a", "dict"])
    assert bad.error_type == "InvalidArguments"


def test_catalog_is_json() -> None:
    entry = json.loads(json.dumps(demo_package.catalog_entry()))
    search = next(t for t in entry["tools"] if t["name"] == "search_logs")
    assert search["params"][-1] == {
        "name": "limit",
        "kind": "POSITIONAL_OR_KEYWORD",
        "annotation": "int",
        "default": 10000,
    }
    assert search["returns"] == "list[dict]" and search["effect"] == "read"
