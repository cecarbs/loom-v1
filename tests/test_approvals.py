"""Built-in tool approvals (docs/03 "Approvals for built-in tools")."""

from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from test_bash_execute import Harness, call

from loom.broker.approval import builtin_rule
from loom.config import LoomConfig
from loom.core import events as ev
from loom.core.messages import TextBlock, ToolUseBlock


def config(mode: str, **overrides: str) -> LoomConfig:
    return LoomConfig.model_validate({"approvals": {"mode": mode, "overrides": overrides}})


class Answer:
    """Answers approval requests on the bus like the TUI would."""

    def __init__(self, h: Harness, *decisions: ev.Decision) -> None:
        self.decisions = list(decisions)
        self.seen: list[ev.ApprovalRequested] = []
        self.bus = h.bus
        h.bus.subscribe(self)

    async def __call__(self, event: ev.AgentEvent) -> None:
        if isinstance(event, ev.ApprovalRequested):
            self.seen.append(event)
            self.bus.respond(event.request_id, self.decisions.pop(0))


@pytest.fixture
async def make(tmp_path: Path) -> AsyncIterator:
    made: list[Harness] = []

    def factory(script, cfg: LoomConfig) -> Harness:
        h = Harness(tmp_path, script, cfg)
        made.append(h)
        return h

    yield factory
    for h in made:
        await h.close()


MODES = ("read-only", "ask", "auto-write", "yolo")


@pytest.mark.parametrize(
    ("kind", "sandboxed", "expected"),
    [
        ("read", False, ("allow", "allow", "allow", "allow")),
        ("outside_read", False, ("ask", "ask", "ask", "allow")),
        ("write", False, ("deny", "allow", "allow", "allow")),
        ("exec", True, ("allow", "allow", "allow", "allow")),
        ("exec", False, ("ask", "ask", "allow", "allow")),
    ],
)
def test_policy_table(kind, sandboxed: bool, expected: tuple[str, ...]) -> None:
    for mode, rule in zip(MODES, expected, strict=True):
        assert builtin_rule(kind, mode, sandboxed=sandboxed) == rule, (kind, mode)  # type: ignore[arg-type]


async def test_bash_asks_with_backend_none_and_denial_is_an_error(make) -> None:
    h = make([[call("b1", "bash", command="touch hacked")], [TextBlock("ok")]], config("ask"))
    answer = Answer(h, ev.Decision("deny", "use read instead"))
    await h.agent.run_turn("go")

    request = answer.seen[0]
    assert (
        request.tool == "bash"
        and request.effect == "exec"
        and request.args_preview == ("touch hacked")
    )
    result = h.results()[-1]
    assert result.is_error
    assert result.content == "The user denied this bash call: use read instead"
    assert not (h.workspace / "hacked").exists()
    [entry] = h.log("approval")
    assert entry["tool"] == "bash" and entry["action"] == "deny" and entry["by"] == "user"
    assert entry["message"] == "use read instead" and entry["call_id"] == "b1"


async def test_execute_asks_and_allow_session_is_remembered(make) -> None:
    h = make(
        [
            [call("e1", "execute", code="x = 1")],
            [call("e2", "execute", code="x + 1")],
            [TextBlock("ok")],
        ],
        config("ask"),
    )
    answer = Answer(h, ev.Decision("allow_session"))
    await h.agent.run_turn("go")
    assert len(answer.seen) == 1  # the second call didn't ask
    assert h.result_text() == "2"
    assert [(e["action"], e["by"]) for e in h.log("approval")] == [
        ("allow_session", "user"),
        ("allow_once", "session"),
    ]


async def test_read_outside_workspace_asks_per_path(make, tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("A")
    (tmp_path / "b.txt").write_text("B")
    h = make(
        [
            [call("r1", "read", path="../a.txt")],
            [call("r2", "read", path="../a.txt")],
            [call("r3", "read", path="../b.txt")],
            [call("r4", "read", path="inside.txt")],
            [TextBlock("ok")],
        ],
        config("ask"),
    )
    (h.workspace / "inside.txt").write_text("I")
    answer = Answer(h, ev.Decision("allow_session"), ev.Decision("deny"))
    await h.agent.run_turn("go")
    assert [r.is_error for r in h.results()] == [False, False, True, False]
    assert len(answer.seen) == 2  # a.txt once (remembered), b.txt once; inside never
    assert "outside the workspace" in answer.seen[0].args_preview
    assert answer.seen[0].effect == "outside read"


async def test_read_only_mode_denies_writes_without_asking(make) -> None:
    h = make(
        [[call("w1", "write", path="new.txt", content="x")], [TextBlock("ok")]],
        config("read-only"),
    )
    answer = Answer(h)
    await h.agent.run_turn("go")
    assert not answer.seen and not (h.workspace / "new.txt").exists()
    assert h.results()[-1].content == "write is not allowed in read-only mode."
    assert h.log("approval")[0]["by"] == "policy"


async def test_overrides(make) -> None:
    h = make(
        [
            [call("b1", "bash", command="echo allowed")],
            [call("e1", "execute", code="1")],
            [TextBlock("ok")],
        ],
        config("ask", bash="allow", execute="deny"),
    )
    answer = Answer(h)
    await h.agent.run_turn("go")
    assert not answer.seen
    assert "allowed" in h.result_text(0)
    assert h.result_text(1) == "execute is denied by the user's configuration."
    assert [(e["tool"], e["by"]) for e in h.log("approval")] == [
        ("bash", "override"),
        ("execute", "override"),
    ]


async def test_cancel_while_asking_logs_cancel_and_keeps_pairing(make) -> None:
    import asyncio

    h = make([[call("b1", "bash", command="echo hi")], [TextBlock("x")]], config("ask"))
    asked = asyncio.Event()

    async def hold(event: ev.AgentEvent) -> None:
        if isinstance(event, ev.ApprovalRequested):
            asked.set()  # never answered

    h.bus.subscribe(hold)
    task = asyncio.create_task(h.agent.run_turn("go"))
    await asked.wait()
    task.cancel()
    await task
    [entry] = h.log("approval")
    assert entry["by"] == "cancel" and entry["action"] == "deny"
    assert isinstance(h.agent.history[-1].content[0], type(h.results()[-1]))
    assert isinstance(h.agent.history[-2].content[-1], ToolUseBlock)
