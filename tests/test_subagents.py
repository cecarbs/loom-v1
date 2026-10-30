"""Sub-agents (docs/08): `agents.run` / `agents.map`, approvals, limits, transcripts."""

import asyncio
import io
import json
from collections.abc import AsyncIterator, Mapping, Sequence
from pathlib import Path

import pytest
from loom_tools_demo import package as demo_package
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from rich.console import Console

from loom.app import build_agent
from loom.broker import ToolPackage
from loom.broker.redact import Redactor
from loom.broker.secrets import SecretStore
from loom.config import LoomConfig
from loom.core import events as ev
from loom.core.agent import AgentLoop
from loom.core.events import EventBus, Usage
from loom.core.messages import TextBlock, ToolResultBlock, ToolUseBlock
from loom.extensions.api import Block, Extensions, Hook
from loom.providers.base import ModelRequest, ProviderError, ProviderEvent
from loom.providers.fake import FakeProvider, FakeResponse, Script
from loom.session.store import SessionLog, find_session, rebuild, release_lock
from loom.tui.approvals import ApprovalPrompt
from loom.tui.render import Renderer


class Routed(FakeProvider):
    """Plays a separate script per first prompt, so parallel children can't mix them up, and
    records how many streams ran at once."""

    def __init__(self, script: Sequence[Script], tasks: Mapping[str, Sequence[Script]]) -> None:
        super().__init__(script)
        self.tasks = {task: FakeProvider(s) for task, s in tasks.items()}
        self.active = self.peak = 0

    async def stream(self, req: ModelRequest) -> AsyncIterator[ProviderEvent]:
        fake = self.tasks.get(req.messages[0].text())
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            async for event in (super() if fake is None else fake).stream(req):
                yield event
        finally:
            self.active -= 1

    def child_requests(self, task: str) -> list[ModelRequest]:
        return self.tasks[task].requests


def ops_package() -> ToolPackage:
    pkg = ToolPackage(name="ops", description="Test operations.")

    @pkg.tool(effect="write")
    def deploy(ctx, name: str) -> str:
        """Deploy a service."""
        return f"deployed {name}"

    return pkg


class Harness:
    def __init__(
        self,
        tmp_path: Path,
        script: Sequence[Script],
        tasks: Mapping[str, Sequence[Script]] | None = None,
        *,
        mode: str = "ask",
        extensions: Extensions | None = None,
        models: Mapping[str, Sequence[Script]] | None = None,
        resume: bool = False,
        **config,
    ) -> None:
        self.tmp_path = tmp_path
        self.workspace = tmp_path / "ws"
        self.workspace.mkdir(exist_ok=True)
        self.bus = EventBus()
        self.events: list[ev.AgentEvent] = []
        self.bus.subscribe(self._record)
        self.provider = Routed(script, tasks or {})
        self.answers: list[ev.Decision] = []
        self.config = LoomConfig.model_validate(
            {"approvals": {"mode": mode, "overrides": {"execute": "allow"}}, **config}
        )
        self.other = FakeProvider(list((models or {}).get("other", [])))
        self.agent = build_agent(
            self.workspace,
            self.config,
            self.bus,
            provider=self.provider,
            provider_factory=lambda profile: self.other,
            home=tmp_path / "home",
            secrets=SecretStore({}, redactor=Redactor()),
            packages=[demo_package, ops_package()],
            extensions=extensions,
            resume=find_session(self.workspace, None, tmp_path / "home") if resume else None,
        )
        subagents = self.agent.tool_ctx.subagents
        assert subagents is not None
        self.subagents = subagents
        self.loops: list[AgentLoop] = []
        make_loop = subagents.make_loop

        def recording(spec):
            loop = make_loop(spec)
            self.loops.append(loop)
            return loop

        subagents.make_loop = recording

    async def _record(self, event: ev.AgentEvent) -> None:
        self.events.append(event)
        if isinstance(event, ev.ApprovalRequested) and self.answers:
            self.bus.respond(event.request_id, self.answers.pop(0))

    def text(self, i: int = -1) -> str:
        results = [
            b
            for m in self.agent.history
            if m.role == "user"
            for b in m.content
            if isinstance(b, ToolResultBlock)
        ]
        content = results[i].content
        assert isinstance(content, str)
        return content

    def log(self, kind: str) -> list[dict]:
        return [r for r in self.agent.log.read() if r["kind"] == kind]

    def transcript(self, agent_id: str = "sa1") -> list[dict]:
        return SessionLog(self.agent.log.path.parent / "agents" / f"{agent_id}.jsonl").read()

    def notices(self) -> list[str]:
        return [e.text for e in self.events if isinstance(e, ev.Notice)]

    async def close(self) -> None:
        await self.subagents.aclose()
        assert self.agent.tool_ctx.kernel is not None
        await self.agent.tool_ctx.kernel.shutdown()
        release_lock(self.agent_paths())

    def agent_paths(self):
        return find_session(self.workspace, self.agent.log.path.parent.name, self.tmp_path / "home")


def execute(code: str, id_: str = "e1", **kw) -> ToolUseBlock:
    return ToolUseBlock(id_, "execute", {"code": code, **kw})


def parent_script(code: str, **kw) -> list[Script]:
    return [[execute(code, **kw)], [TextBlock("all done")]]


@pytest.fixture
async def make(tmp_path: Path) -> AsyncIterator:
    made: list[Harness] = []

    def factory(*args, **kw) -> Harness:
        h = Harness(tmp_path, *args, **kw)
        made.append(h)
        return h

    yield factory
    for h in made:
        await h.close()


def slow_answer(text: str, delay: float = 0.02) -> FakeResponse:
    # One chunk per word, so `delay` per word keeps the stream open.
    return FakeResponse([TextBlock(" ".join([text] * 5))], chunk_delay=delay)


# --- acceptance ---------------------------------------------------------------------


async def test_map_runs_concurrently_within_the_limit_and_keeps_order(make) -> None:
    tasks = {f"t{i}": [slow_answer(f"answer{i}")] for i in range(4)}
    h = make(
        parent_script("out = agents.map(['t0', 't1', 't2', 't3'])\nprint(type(out), out)"),
        tasks,
        agents={"max_concurrency": 2},
    )
    await h.agent.run_turn("go")
    assert h.provider.peak == 2
    assert "<class 'list'>" in h.text()
    positions = [h.text().index(f"answer{i}") for i in range(4)]
    assert positions == sorted(positions)
    assert [r["status"] for r in h.log("subagent")] == ["done"] * 4


async def test_map_concurrency_can_only_lower_the_limit(make) -> None:
    tasks = {f"t{i}": [slow_answer(f"a{i}")] for i in range(3)}
    h = make(parent_script("agents.map(['t0', 't1', 't2'], concurrency=1)"), tasks)
    await h.agent.run_turn("go")
    assert h.provider.peak == 1

    tasks = {f"u{i}": [slow_answer(f"b{i}")] for i in range(3)}
    h = make(
        parent_script("agents.map(['u0', 'u1', 'u2'], concurrency=50)"),
        tasks,
        agents={"max_concurrency": 2},
    )
    await h.agent.run_turn("go")
    assert h.provider.peak == 2


async def test_write_approval_reaches_the_prompt_labelled_with_the_agent_id(make) -> None:
    h = make(
        parent_script("print(agents.run('deploy it'))"),
        {
            "deploy it": [
                [execute("print(ops.deploy('web'))")],
                [TextBlock("deployed")],
            ]
        },
    )
    file = io.StringIO()
    console = Console(file=file, width=80, color_system=None)
    with create_pipe_input() as pipe:
        prompt = ApprovalPrompt(console, input=pipe, output=DummyOutput())
        h.bus.subscribe(Renderer(console, prompt, h.bus))
        turn = asyncio.create_task(h.agent.run_turn("go"))
        while "ops.deploy" not in file.getvalue():
            await asyncio.sleep(0.02)
        await asyncio.sleep(0.05)
        pipe.send_text("y")
        await asyncio.wait_for(turn, 10)
    assert "[sa1] ops.deploy  [write]" in file.getvalue()
    assert "deployed" in h.text()
    # The decision is in the sub-agent's transcript, as every approval is in a log.
    assert [r["tool"] for r in h.transcript() if r["kind"] == "approval"][-1] == "ops.deploy"


async def test_sub_agent_cannot_spawn_a_sub_agent(make) -> None:
    h = make(
        parent_script("print(agents.run('nest'))"),
        {
            "nest": [
                [execute("agents.run('deeper')")],
                [ToolUseBlock("s1", "search_tools", {"query": ""})],
                [TextBlock("could not")],
            ]
        },
    )
    await h.agent.run_turn("go")
    results = [r["result"] for r in h.transcript() if r["kind"] == "tool_result"]
    assert "NameError" in results[0] and "agents" in results[0]
    assert "agents" not in results[1].split("demo")[0]
    assert "could not" in h.text()
    assert len(h.log("subagent")) == 1
    # Not even with a hand-written protocol message: the child's broker has no such tool.
    child = h.loops[0].tool_ctx.broker
    reply = await child.call("agents.run", {"task": "x"})
    assert reply.error_type == "ToolNotFound"


async def test_sub_agent_outlasts_call_timeout_and_execute_timeout(make) -> None:
    h = make(
        parent_script("print(agents.run('slow'))", timeout_s=1),
        {"slow": [FakeResponse([TextBlock("a b c d e f g h i j k l")], chunk_delay=0.15)]},
        broker={"call_timeout_s": 0.2},
    )
    await h.agent.run_turn("go")
    assert "a b c d e f g h i j k l" in h.text()
    record = h.log("broker_call")[0]
    assert record["tool"] == "agents.run" and record["ok"] and record["duration_ms"] > 1200


# --- what a sub-agent is ------------------------------------------------------------


async def test_own_kernel_history_and_system_prompt(make) -> None:
    h = make(
        [
            [execute("secret_var = 42")],
            [execute("print(agents.run('look'))", "e2")],
            [TextBlock("done")],
        ],
        {"look": [[execute("print('secret_var' in globals())")], [TextBlock("isolated")]]},
    )
    await h.agent.run_turn("go")
    result = next(r["result"] for r in h.transcript() if r["kind"] == "tool_result")
    assert "False" in result
    child = h.provider.child_requests("look")[0]
    assert "# Sub-agent" in child.system and "# Sub-agent" not in h.provider.requests[0].system
    assert "agents.run(task)" in h.provider.requests[0].system
    assert "agents.run(task)" not in child.system
    assert len(child.messages) == 1  # nothing of the parent's conversation


async def test_read_only_mode_is_inherited(make) -> None:
    h = make(
        parent_script("print(agents.run('deploy it'))"),
        {
            "deploy it": [
                [execute("ops.deploy('web')")],
                [ToolUseBlock("w1", "write", {"path": "x.txt", "content": "x"})],
                [TextBlock("blocked")],
            ]
        },
        mode="read-only",
    )
    await h.agent.run_turn("go")
    results = [r["result"] for r in h.transcript() if r["kind"] == "tool_result"]
    assert "PermissionDenied" in results[0]
    assert "not allowed in read-only mode" in results[1]
    assert not (h.workspace / "x.txt").exists()


async def test_allow_for_session_is_shared_and_ids_do_not_collide(make) -> None:
    deploy = [[execute("ops.deploy('web')")], [TextBlock("ok")]]
    h = make(
        parent_script("print(agents.map(['a', 'b'], concurrency=1))"),
        {"a": deploy, "b": deploy},
    )
    h.answers = [ev.Decision("allow_session")]
    await h.agent.run_turn("go")
    asked = [e for e in h.events if isinstance(e, ev.ApprovalRequested)]
    assert [e.agent_id for e in asked] == ["sa1"]  # sa2 used the grant


async def test_concurrent_approvals_get_distinct_request_ids(make) -> None:
    deploy = [[execute("ops.deploy('web')")], [TextBlock("ok")]]
    h = make(parent_script("print(agents.map(['a', 'b']))"), {"a": deploy, "b": deploy})
    h.answers = [ev.Decision("allow_once"), ev.Decision("allow_once")]
    await h.agent.run_turn("go")
    asked = [e for e in h.events if isinstance(e, ev.ApprovalRequested)]
    assert {e.agent_id for e in asked} == {"sa1", "sa2"}
    assert len({e.request_id for e in asked}) == 2


async def test_broker_hooks_bind_sub_agents(make) -> None:
    def guard(call):
        return Block("no deploys") if call.tool == "ops.deploy" else None

    h = make(
        parent_script("print(agents.run('deploy it'))"),
        {"deploy it": [[execute("ops.deploy('web')")], [TextBlock("blocked")]]},
        extensions=Extensions(hooks=[Hook("guard", "before_broker_call", guard)]),
    )
    await h.agent.run_turn("go")
    result = next(r["result"] for r in h.transcript() if r["kind"] == "tool_result")
    assert "PermissionDenied" in result and "no deploys" in result


async def test_model_override_uses_that_model(make) -> None:
    h = make(
        parent_script("print(agents.run('look', model='haiku'))"),
        models={"other": [[TextBlock("from haiku")]]},
    )
    await h.agent.run_turn("go")
    assert "from haiku" in h.text()
    assert h.log("subagent")[0]["model"] == "haiku"


# --- failures -----------------------------------------------------------------------


async def test_run_raises_when_the_model_fails(make) -> None:
    h = make(
        parent_script("try:\n    agents.run('t')\nexcept ToolError as e:\n    print('caught', e)"),
        {"t": [ProviderError("auth", "bad key")]},
    )
    await h.agent.run_turn("go")
    assert "caught" in h.text() and "bad key" in h.text() and "sa1.jsonl" in h.text()
    assert h.log("subagent")[0]["status"] == "failed"


async def test_max_turns_without_an_answer_is_a_failure(make) -> None:
    h = make(
        parent_script("agents.run('t', max_turns=2)"),
        {"t": [[execute("1", "c1")], [execute("2", "c2")]]},
    )
    await h.agent.run_turn("go")
    assert "ToolError" in h.text() and "max_turns=2" in h.text()


async def test_map_keeps_the_answers_of_the_others(make) -> None:
    h = make(
        parent_script("for r in agents.map(['good', 'bad']):\n    print(r)"),
        {"good": [[TextBlock("fine")]], "bad": [ProviderError("auth", "bad key")]},
    )
    await h.agent.run_turn("go")
    assert "fine" in h.text() and "[sub-agent sa2 failed: Model error (auth)" in h.text()


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("agents.run('t', model='nope')", "unknown model alias 'nope'"),
        ("agents.map(['t'], model='nope')", "unknown model alias 'nope'"),
        ("agents.run('t', max_turns=0)", "max_turns must be at least 1"),
        ("agents.map(['t'], concurrency=0)", "concurrency must be at least 1"),
    ],
)
async def test_bad_arguments_fail_before_anything_runs(make, code: str, expected: str) -> None:
    h = make(parent_script(code))
    await h.agent.run_turn("go")
    assert expected in h.text()
    assert h.log("subagent") == [] and h.loops == []


async def test_map_of_nothing(make) -> None:
    h = make(parent_script("print(agents.map([]))"))
    await h.agent.run_turn("go")
    assert "[]" in h.text()


async def test_failed_start_is_reported_and_the_slot_released(make) -> None:
    h = make(parent_script("print(agents.map(['a', 'b'], concurrency=1))"))

    def broken(spec):
        raise RuntimeError("no kernel for you")

    h.subagents.make_loop = broken
    await h.agent.run_turn("go")
    assert h.text().count("no kernel for you") == 2


# --- cancellation -------------------------------------------------------------------


async def test_cancel_stops_children_and_keeps_history_valid(make) -> None:
    hang = [
        [execute("x = 1")],
        FakeResponse([TextBlock("thinking about it")], hang_after_text=True),
    ]
    h = make(
        [[execute("agents.map(['a', 'b'])")]],
        {"a": hang, "b": [FakeResponse([TextBlock("hmm")], hang_after_text=True)]},
    )
    turn = asyncio.create_task(h.agent.run_turn("go"))
    while not (len(h.loops) == 2 and h.provider.active == 2):
        await asyncio.sleep(0.02)
    kernels = [loop.tool_ctx.kernel for loop in h.loops]
    turn.cancel()
    await asyncio.wait_for(turn, 15)

    assert any(isinstance(e, ev.TurnCancelled) for e in h.events)
    assert h.provider.active == 0 and not h.subagents._active
    assert not any(k is not None and k.alive for k in kernels)
    assert [r["status"] for r in h.log("subagent")] == ["cancelled", "cancelled"]
    last = h.agent.history[-1]
    assert isinstance(last.content[0], ToolResultBlock)


async def test_aclose_cleans_up_what_cancellation_left(make) -> None:
    h = make(
        parent_script("agents.run('hang')"),
        {"hang": [FakeResponse([TextBlock("hmm")], hang_after_text=True)]},
    )
    turn = asyncio.create_task(h.agent.run_turn("go"))
    while h.provider.active == 0 or not h.loops:
        await asyncio.sleep(0.02)
    await asyncio.wait_for(h.subagents.aclose(), 5)
    assert not h.subagents._active
    turn.cancel()
    await asyncio.wait_for(turn, 15)


# --- transcripts and accounting -----------------------------------------------------


async def test_transcript_is_written_and_referenced(make) -> None:
    h = make(
        parent_script("print(agents.run('look'))"),
        {"look": [[execute("6 * 7")], [TextBlock("forty-two")]]},
    )
    await h.agent.run_turn("go")
    record = h.log("subagent")[0]
    assert record["id"] == "sa1" and record["status"] == "done" and record["steps"] == 2
    assert Path(record["transcript"]) == h.agent.log.path.parent / "agents" / "sa1.jsonl"
    kinds = [r["kind"] for r in h.transcript()]
    assert kinds[0] == "session_start"
    for kind in ("message", "model_request", "tool_call", "kernel_execute", "turn_end"):
        assert kind in kinds
    assert h.transcript()[0]["session_id"].endswith("/sa1")
    # The parent's log stays the parent's: sub-agent records are in the transcript.
    assert not any(r.get("code") == "6 * 7" for r in h.log("kernel_execute"))


async def test_usage_counts_toward_the_session_but_not_its_context(make) -> None:
    usage = Usage(input_tokens=1000, output_tokens=50)
    h = make(
        parent_script("print(agents.run('look'))"),
        {"look": [FakeResponse([TextBlock("ok")], usage=usage)]},
    )
    await h.agent.run_turn("go")
    # 2 parent requests at 100 in / 20 out, plus the child's.
    assert h.agent.usage.input_tokens == 1200 and h.agent.usage.output_tokens == 90
    assert h.agent.cost_usd is not None and h.agent.cost_usd > 0
    assert h.agent.context_tokens() < 1000
    state = rebuild(h.agent.log.read())
    assert state.usage == h.agent.usage
    assert state.cost_usd == pytest.approx(h.agent.cost_usd)
    assert state.last_prompt_tokens == 100


async def test_resumed_session_continues_the_transcript_numbers(make) -> None:
    h = make(parent_script("print(agents.run('one'))"), {"one": [[TextBlock("1")]]})
    await h.agent.run_turn("go")
    await h.close()

    h2 = make(
        parent_script("print(agents.run('two'))"),
        {"two": [[TextBlock("2")]]},
        resume=True,
    )
    await h2.agent.run_turn("again")
    assert h2.log("subagent")[-1]["id"] == "sa2"
    assert json.loads(
        (h2.agent.log.path.parent / "agents" / "sa1.jsonl").read_text().splitlines()[0]
    )["kind"] == "session_start"  # fmt: skip


async def test_transcripts_are_readable_with_the_read_tool(make) -> None:
    h = make(
        [
            [execute("print(agents.run('look'))")],
            [ToolUseBlock("r1", "read", {"path": "PLACEHOLDER"})],
            [TextBlock("done")],
        ],
        {"look": [[TextBlock("ok")]]},
    )
    path = h.agent.log.path.parent / "agents" / "sa1.jsonl"
    h.provider._script[1] = [ToolUseBlock("r1", "read", {"path": str(path)})]
    await h.agent.run_turn("go")
    assert "session_start" in h.text()
    assert not any(isinstance(e, ev.ApprovalRequested) for e in h.events)


async def test_trace_shows_the_sub_agent(make) -> None:
    from loom.tracing.log import build_trace, render_trace

    h = make(parent_script("print(agents.run('look'))"), {"look": [[TextBlock("ok")]]})
    await h.agent.run_turn("go")
    text = "\n".join(render_trace(build_trace(h.agent.log.read())))
    assert "subagent sa1 done" in text and "sa1.jsonl" in text
    assert "agents.run" in text
