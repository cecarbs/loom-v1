import asyncio
from pathlib import Path

import pytest
from pydantic import BaseModel

from loom.app import build_agent
from loom.config import LoomConfig
from loom.core import events as ev
from loom.core.agent import TRUNCATED_TOOL_INPUT, AgentLoop
from loom.core.events import EventBus, Usage
from loom.core.messages import (
    INTERRUPTED,
    Message,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from loom.providers.base import ProviderError
from loom.providers.fake import FakeProvider, FakeResponse, Script
from loom.tools.base import ToolContext, ToolResult


def assert_valid_pairing(history: list[Message]) -> None:
    """Every assistant tool_use is answered by a tool_result in the very next message."""
    for i, msg in enumerate(history):
        if msg.role != "assistant" or not msg.tool_uses():
            continue
        assert i + 1 < len(history), "tool uses at end of history without results"
        nxt = history[i + 1]
        assert nxt.role == "user"
        answered = {b.tool_use_id for b in nxt.content if isinstance(b, ToolResultBlock)}
        assert {t.id for t in msg.tool_uses()} == answered


class Harness:
    def __init__(self, tmp_path: Path, script: list[Script], config: LoomConfig | None = None):
        self.workspace = tmp_path / "ws"
        self.workspace.mkdir()
        self.bus = EventBus()
        self.events: list[ev.AgentEvent] = []

        async def record(e: ev.AgentEvent) -> None:
            self.events.append(e)

        self.bus.subscribe(record)
        self.provider = FakeProvider(script)
        self.agent: AgentLoop = build_agent(
            self.workspace,
            config or LoomConfig(),
            self.bus,
            provider=self.provider,
            home=tmp_path / "home",
        )

    def log_kinds(self) -> list[str]:
        return [r["kind"] for r in self.agent.log.read()]

    def of(self, kind: type) -> list:
        return [e for e in self.events if isinstance(e, kind)]


def tool(id_: str, name: str, **input_) -> ToolUseBlock:
    return ToolUseBlock(id_, name, input_)


async def test_read_edit_final_answer(tmp_path: Path) -> None:
    h = Harness(
        tmp_path,
        [
            [TextBlock("Let me look."), tool("t1", "read", path="hello.py")],
            [tool("t2", "edit", path="hello.py", old_string="world", new_string="loom")],
            [TextBlock("Done: greeting now says loom.")],
        ],
    )
    (h.workspace / "hello.py").write_text('print("hello world")\n')

    await h.agent.run_turn("change the greeting")

    assert (h.workspace / "hello.py").read_text() == 'print("hello loom")\n'
    history = h.agent.history
    assert [m.role for m in history] == [
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert_valid_pairing(history)
    assert history[-1].text() == "Done: greeting now says loom."
    read_result = history[2].content[0]
    assert isinstance(read_result, ToolResultBlock) and "hello world" in str(read_result.content)

    # The model saw the system prompt and the tool schemas.
    request = h.provider.requests[0]
    assert {t.name for t in request.tools} == {
        "read",
        "write",
        "edit",
        "bash",
        "execute",
        "search_tools",
    }
    assert "`edit`" in request.system and "`search_tools`" in request.system

    kinds = h.log_kinds()
    assert kinds[0] == "session_start"
    assert kinds.count("model_request") == 3
    first = next(r for r in h.agent.log.read() if r["kind"] == "model_request")
    assert first["settings"]["max_tokens"] == 64_000 and first["settings"]["tools"] == 6
    assert kinds.count("tool_call") == kinds.count("tool_result") == 2
    assert kinds.count("message") == 6
    assert [type(e) for e in h.events][-1] is ev.TurnFinished
    assert "".join(e.text for e in h.of(ev.TextDelta)).startswith("Let me look.")


async def test_usage_and_cost_accounting(tmp_path: Path) -> None:
    usage = Usage(input_tokens=1_000, output_tokens=500, cache_read_tokens=10_000)
    h = Harness(tmp_path, [FakeResponse([TextBlock("hi")], usage=usage)])
    await h.agent.run_turn("hello")
    updated = h.of(ev.UsageUpdated)[-1]
    # sonnet: $2 in, $10 out, $0.20 cache read per MTok
    assert updated.cost_usd == pytest.approx((1_000 * 2 + 500 * 10 + 10_000 * 0.2) / 1e6)
    assert updated.context_pct == pytest.approx(11_000 / 1_000_000)


async def test_cancel_mid_stream_keeps_partial_text(tmp_path: Path) -> None:
    h = Harness(
        tmp_path,
        [
            FakeResponse(
                [TextBlock("I am thinking about it"), tool("t1", "read", path="x")],
                hang_after_text=True,
            )
        ],
    )
    task = asyncio.create_task(h.agent.run_turn("go"))
    while not h.of(ev.TextDelta):
        await asyncio.sleep(0)
    await asyncio.sleep(0.01)
    task.cancel()
    await task  # run_turn absorbs the cancellation

    assert not task.cancelled()
    history = h.agent.history
    assert history[-1].role == "assistant"
    assert history[-1].meta["interrupted"] is True
    assert history[-1].text() == "I am thinking about it"
    assert not history[-1].tool_uses()  # the tool use never completed
    assert_valid_pairing(history)
    assert h.of(ev.TurnCancelled) and not h.of(ev.TurnFinished)
    assert "turn_cancelled" in h.log_kinds()


async def test_cancel_mid_tool_batch(tmp_path: Path) -> None:
    h = Harness(
        tmp_path,
        [
            [
                tool("t1", "write", path="a.txt", content="a"),
                tool("t2", "slow"),
                tool("t3", "read", path="a.txt"),
            ]
        ],
    )
    started = asyncio.Event()

    class NoArgs(BaseModel):
        pass

    class SlowTool:
        name = "slow"
        description = "sleeps"
        input_model = NoArgs

        async def run(self, args: BaseModel, ctx: ToolContext) -> ToolResult:
            started.set()
            await asyncio.sleep(60)
            return ToolResult("unreachable")

    h.agent.tools._tools["slow"] = SlowTool()  # type: ignore[assignment]
    task = asyncio.create_task(h.agent.run_turn("go"))
    await started.wait()
    task.cancel()
    await task

    history = h.agent.history
    assert_valid_pairing(history)
    results = {b.tool_use_id: b for b in history[-1].content if isinstance(b, ToolResultBlock)}
    assert results["t1"].is_error is False  # completed before the cancel: real result kept
    assert results["t2"].content == INTERRUPTED and results["t2"].is_error
    assert results["t3"].content == INTERRUPTED
    assert (h.workspace / "a.txt").read_text() == "a"


async def test_cancelled_turn_then_next_turn_is_valid(tmp_path: Path) -> None:
    h = Harness(
        tmp_path,
        [
            FakeResponse([TextBlock("partial answer")], hang_after_text=True),
            [TextBlock("second answer")],
        ],
    )
    task = asyncio.create_task(h.agent.run_turn("first"))
    while not h.of(ev.TextDelta):
        await asyncio.sleep(0)
    task.cancel()
    await task
    await h.agent.run_turn("second")
    assert h.agent.history[-1].text() == "second answer"
    assert h.provider.requests[1].messages[-1].text() == "second"


async def test_max_tokens_tool_use_is_not_run(tmp_path: Path) -> None:
    h = Harness(
        tmp_path,
        [
            FakeResponse(
                [tool("t1", "write", path="big.txt", content="trunc")], stop_reason="max_tokens"
            ),
            [TextBlock("ok")],
        ],
    )
    await h.agent.run_turn("write a big file")
    assert not (h.workspace / "big.txt").exists()
    result = h.agent.history[2].content[0]
    assert isinstance(result, ToolResultBlock)
    assert result.is_error and result.content == TRUNCATED_TOOL_INPUT
    assert_valid_pairing(h.agent.history)


async def test_retries_rate_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        if seconds:  # FakeProvider's zero chunk delay goes through the same patched sleep
            sleeps.append(seconds)

    monkeypatch.setattr("loom.core.agent.asyncio.sleep", fake_sleep)
    h = Harness(
        tmp_path,
        [
            ProviderError("rate_limit", "slow down", retry_after=3),
            ProviderError("overloaded", "busy"),
            [TextBlock("finally")],
        ],
    )
    await h.agent.run_turn("hi")
    assert sleeps == [3, 2.0]
    assert h.agent.history[-1].text() == "finally"
    assert len(h.of(ev.Notice)) == 2


async def test_non_retryable_error_ends_turn(tmp_path: Path) -> None:
    h = Harness(tmp_path, [ProviderError("auth", "bad key")])
    await h.agent.run_turn("hi")
    assert "bad key" in h.of(ev.ErrorEvent)[0].text
    assert h.of(ev.TurnFinished)
    assert [m.role for m in h.agent.history] == ["user"]


async def test_step_limit(tmp_path: Path) -> None:
    config = LoomConfig.model_validate({"agent": {"max_steps": 2}})
    h = Harness(tmp_path, [[tool(f"t{i}", "read", path="f")] for i in range(2)], config)
    (h.workspace / "f").write_text("x")
    await h.agent.run_turn("loop")
    assert any("Step limit" in n.text for n in h.of(ev.Notice))
    assert_valid_pairing(h.agent.history)


async def test_bad_tool_calls_return_errors(tmp_path: Path) -> None:
    h = Harness(
        tmp_path,
        [
            [
                tool("t1", "nope"),
                tool("t2", "read"),  # missing path
                ToolUseBlock("t3", "read", {"__invalid_json__": "{bad"}),
            ],
            [TextBlock("sorry")],
        ],
    )
    await h.agent.run_turn("x")
    results = [b for b in h.agent.history[2].content if isinstance(b, ToolResultBlock)]
    assert all(r.is_error for r in results)
    assert "Unknown tool 'nope'" in str(results[0].content)
    assert "path" in str(results[1].content)
    assert "not valid JSON" in str(results[2].content)


async def test_large_tool_result_is_budgeted(tmp_path: Path) -> None:
    h = Harness(
        tmp_path,
        [[tool("t1", "read", path="big.txt", limit=100_000)], [TextBlock("ok")]],
    )
    (h.workspace / "big.txt").write_text("\n".join("x" * 50 for _ in range(10_000)))
    await h.agent.run_turn("read it")
    result = h.agent.history[2].content[0]
    assert isinstance(result, ToolResultBlock) and isinstance(result.content, str)
    assert len(result.content) < 12_500
    assert "chars omitted" in result.content
    spilled = h.agent.log.read()
    spill = next(r for r in spilled if r["kind"] == "tool_result")["spill_path"]
    assert Path(spill).read_text().count("\n") == 9_999


async def test_empty_response_is_not_appended(tmp_path: Path) -> None:
    h = Harness(tmp_path, [[], [TextBlock("second try")]])
    await h.agent.run_turn("hi")
    assert [m.role for m in h.agent.history] == ["user"]
    assert any("empty response" in n.text for n in h.of(ev.Notice))
    assert h.of(ev.TurnFinished)
    await h.agent.run_turn("again")
    assert h.agent.history[-1].text() == "second try"
