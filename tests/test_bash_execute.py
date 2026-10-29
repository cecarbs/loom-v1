"""bash and execute through the agent loop: streaming, budget, timeouts, cancellation."""

import asyncio
import os
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from loom.app import build_agent
from loom.config import LoomConfig
from loom.core import events as ev
from loom.core.agent import AgentLoop
from loom.core.events import EventBus
from loom.core.messages import INTERRUPTED, Message, TextBlock, ToolResultBlock, ToolUseBlock
from loom.providers.fake import FakeProvider, Script
from loom.tools.builtin.execute import RESTARTED

YOLO = LoomConfig.model_validate({"approvals": {"mode": "yolo"}})


class Harness:
    def __init__(self, tmp_path: Path, script: list[Script], config: LoomConfig = YOLO) -> None:
        self.workspace = tmp_path / "ws"
        self.workspace.mkdir(exist_ok=True)
        self.bus = EventBus()
        self.events: list[ev.AgentEvent] = []

        async def record(e: ev.AgentEvent) -> None:
            self.events.append(e)

        self.bus.subscribe(record)
        self.provider = FakeProvider(script)
        self.agent: AgentLoop = build_agent(
            self.workspace, config, self.bus, provider=self.provider, home=tmp_path / "home"
        )

    def results(self) -> list[ToolResultBlock]:
        return [
            b
            for m in self.agent.history
            if m.role == "user"
            for b in m.content
            if isinstance(b, ToolResultBlock)
        ]

    def result_text(self, i: int = -1) -> str:
        content = self.results()[i].content
        assert isinstance(content, str)
        return content

    def of(self, kind: type) -> list:
        return [e for e in self.events if isinstance(e, kind)]

    def log(self, kind: str) -> list[dict]:
        return [r for r in self.agent.log.read() if r["kind"] == kind]

    async def close(self) -> None:
        assert self.agent.tool_ctx.kernel is not None
        await self.agent.tool_ctx.kernel.shutdown()


def call(id_: str, name: str, **input_) -> ToolUseBlock:
    return ToolUseBlock(id_, name, input_)


@pytest.fixture
async def make(tmp_path: Path) -> AsyncIterator:
    made: list[Harness] = []

    def factory(script: list[Script], config: LoomConfig = YOLO) -> Harness:
        h = Harness(tmp_path, script, config)
        made.append(h)
        return h

    yield factory
    for h in made:
        await h.close()


# --- bash ---------------------------------------------------------------------------


async def test_bash_runs_in_workspace_and_streams(make) -> None:
    command = "pwd; echo out; echo err >&2; exit 3"
    h = make([[call("b1", "bash", command=command)], [TextBlock("ok")]])
    await h.agent.run_turn("go")
    text = h.result_text()
    assert str(h.workspace.resolve()) in text or str(h.workspace) in text
    assert "out\n" in text and "err\n" in text and text.endswith("exit code: 3")
    assert h.results()[-1].is_error
    streamed = "".join(e.chunk for e in h.of(ev.ToolCallOutput) if e.id == "b1")
    assert "out\n" in streamed and "err\n" in streamed
    finished = h.of(ev.ToolCallFinished)[-1]
    assert finished.meta["exit_code"] == 3


async def test_bash_env_is_allowlisted(make, monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-not-leak")
    h = make([[call("b1", "bash", command="env; echo HOME=$HOME")], [TextBlock("ok")]])
    await h.agent.run_turn("go")
    text = h.result_text()
    assert "sk-should-not-leak" not in text and "ANTHROPIC" not in text
    assert "/sandbox/home" in text


async def test_bash_children_run_in_their_own_process_group(make) -> None:
    h = make([[call("b1", "bash", command="ps -o pgid= -p $$; echo pid $$")], [TextBlock("ok")]])
    await h.agent.run_turn("go")
    lines = h.result_text().split()
    pgid, pid = int(lines[0]), int(lines[2])
    assert pgid == pid and pgid != os.getpgid(0)


async def test_bash_timeout_kills_process_group(make, tmp_path: Path) -> None:
    marker = tmp_path / "survived"
    command = f"(sleep 3; touch {marker}) & echo started; sleep 30"
    h = make([[call("b1", "bash", command=command, timeout_s=1)], [TextBlock("ok")]])
    await asyncio.wait_for(h.agent.run_turn("go"), 10)
    text = h.result_text()
    assert "started" in text and "timed out after 1s" in text
    await asyncio.sleep(3)
    assert not marker.exists()  # the background child was killed with the group


async def test_bash_returns_when_background_child_holds_the_pipe(make) -> None:
    h = make([[call("b1", "bash", command="sleep 30 & echo bg")], [TextBlock("ok")]])
    await asyncio.wait_for(h.agent.run_turn("go"), 5)
    assert h.result_text().endswith("bg\nexit code: 0")


async def test_cancel_kills_running_bash(make, tmp_path: Path) -> None:
    marker = tmp_path / "survived"
    command = f"echo started; sleep 2; touch {marker}"
    h = make([[call("b1", "bash", command=command)], [TextBlock("unused")]])
    task = asyncio.create_task(h.agent.run_turn("go"))
    while not h.of(ev.ToolCallOutput):
        await asyncio.sleep(0.01)
    task.cancel()
    await task
    assert h.of(ev.TurnCancelled)
    assert h.results()[-1].content == INTERRUPTED and h.results()[-1].is_error
    await asyncio.sleep(2.5)
    assert not marker.exists()


# --- execute ------------------------------------------------------------------------


async def test_execute_variables_persist_across_calls(make) -> None:
    h = make(
        [
            [call("e1", "execute", code="rows = list(range(50_000))\nlen(rows)")],
            [call("e2", "execute", code="sum(rows)")],
            [TextBlock("done")],
        ]
    )
    await h.agent.run_turn("go")
    assert h.result_text(0) == "50000"
    assert h.result_text(1) == str(sum(range(50_000)))
    assert [r["status"] for r in h.log("kernel_execute")] == ["ok", "ok"]
    assert h.of(ev.KernelStatus)[-1] == ev.KernelStatus("idle", 1)


async def test_execute_error_and_streaming(make) -> None:
    h = make([[call("e1", "execute", code="print('before')\n1/0")], [TextBlock("ok")]])
    await h.agent.run_turn("go")
    result = h.results()[-1]
    assert result.is_error
    assert isinstance(result.content, str)
    assert result.content.startswith("before\nTraceback")
    assert result.content.endswith("ZeroDivisionError: division by zero")
    assert [e.chunk for e in h.of(ev.ToolCallOutput)] == ["before\n"]


async def test_execute_big_output_is_budgeted_with_kernel_ref(make) -> None:
    h = make(
        [
            [call("e1", "execute", code="print('x' * 30_000)")],
            [call("e2", "execute", code="len(_out[1])")],
            [TextBlock("ok")],
        ]
    )
    await h.agent.run_turn("go")
    text = h.result_text(0)
    assert len(text) < 12_500 and "In kernel: _out[1]" in text
    assert h.result_text(1) == "30001"


async def test_execute_timeout_then_restart_prefix(make) -> None:
    h = make(
        [
            [call("e1", "execute", code="x = 1")],
            [call("e2", "execute", code="import time; time.sleep(30)", timeout_s=1)],
            [call("e3", "execute", code="'x' in dir()")],
            [TextBlock("ok")],
        ]
    )
    await asyncio.wait_for(h.agent.run_turn("go"), 15)
    assert "Timed out after 1s; the kernel was killed" in h.result_text(1)
    assert h.results()[1].is_error
    assert h.result_text(2) == f"{RESTARTED}\nFalse"


async def test_ctrl_c_interrupts_infinite_loop_and_kernel_survives(make) -> None:
    h = make(
        [
            [call("e1", "execute", code="kept = 42")],
            [call("e2", "execute", code="while True: pass")],
            [call("e3", "execute", code="kept")],
            [TextBlock("ok")],
        ]
    )
    task = asyncio.create_task(h.agent.run_turn("go"))
    while not (h.of(ev.ToolCallStarted) and h.of(ev.ToolCallStarted)[-1].id == "e2"):
        await asyncio.sleep(0.01)
    await asyncio.sleep(0.3)
    task.cancel()
    await task
    kernel = h.agent.tool_ctx.kernel
    assert kernel is not None and kernel.state == "idle"
    assert h.results()[-1].content == INTERRUPTED
    assert any(r["status"] == "interrupted" for r in h.log("kernel_execute"))

    await h.agent.run_turn("continue")
    assert h.result_text() == "42"  # no restart prefix: same kernel


async def test_history_stays_valid_after_cancelled_execute(make) -> None:
    h = make([[call("e1", "execute", code="import time; time.sleep(30)")], [TextBlock("x")]])
    task = asyncio.create_task(h.agent.run_turn("go"))
    while not h.of(ev.KernelStatus) or h.of(ev.KernelStatus)[-1].state != "busy":
        await asyncio.sleep(0.01)
    task.cancel()
    await task
    history: list[Message] = h.agent.history
    assert [m.role for m in history] == ["user", "assistant", "user"]
    assert isinstance(history[-1].content[0], ToolResultBlock)
