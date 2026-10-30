"""M5 acceptance: the real sandboxes, from both `bash` and `execute` (docs/05, docs/10)."""

import asyncio
import subprocess
import sys
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path

import pytest
from loom_tools_demo import package as demo_package

from loom.app import build_agent
from loom.config import LoomConfig
from loom.core.events import EventBus
from loom.core.messages import TextBlock, ToolResultBlock, ToolUseBlock
from loom.providers.fake import FakeProvider
from loom.sandbox.base import SandboxBackend, SessionInfo
from loom.sandbox.seatbelt import SeatbeltBackend
from loom.session.store import SessionLog

IMAGE = "loom-sandbox:latest"


def _image_exists() -> bool:
    try:
        done = subprocess.run(["docker", "image", "inspect", IMAGE], capture_output=True)
    except OSError:
        return False
    return done.returncode == 0


@dataclass
class World:
    """A fake home with things the sandbox must not reach, and a workspace."""

    root: Path
    home: Path
    loom_home: Path
    workspace: Path
    ssh_key: Path
    other_log: Path


@pytest.fixture
def world(tmp_path: Path) -> World:
    root = tmp_path.resolve()
    home = root / "home"
    (home / ".ssh").mkdir(parents=True)
    (home / ".ssh" / "id_rsa").write_text("PRIVATE KEY")
    other = home / ".loom" / "sessions" / "proj" / "other-session"
    other.mkdir(parents=True)
    (other / "session.jsonl").write_text('{"kind": "secret stuff"}\n')
    workspace = root / "ws"
    workspace.mkdir()
    return World(
        root, home, home / ".loom", workspace, home / ".ssh" / "id_rsa", other / "session.jsonl"
    )


def seatbelt(world: World) -> SandboxBackend:
    # The project venv stands in for ~/.loom/sandbox-venv (it lives outside the denylist).
    return SeatbeltBackend(loom_home=world.loom_home, home=world.home, python=sys.executable)


def docker(world: World) -> SandboxBackend:
    if not _image_exists():
        pytest.skip(f"{IMAGE} missing; run `loom sandbox setup --docker`")
    from loom.config import SandboxConfig
    from loom.sandbox.docker import DockerBackend

    return DockerBackend(SandboxConfig(), loom_home=world.loom_home)


BACKENDS = [
    pytest.param(seatbelt, marks=pytest.mark.macos, id="seatbelt"),
    pytest.param(docker, marks=pytest.mark.docker, id="docker"),
]


class Sandboxed:
    """Runs bash / execute calls through a real agent on a real backend."""

    def __init__(self, world: World, backend: SandboxBackend, mode: str = "yolo") -> None:
        self.world = world
        self.backend = backend
        self.provider = FakeProvider([])
        self.agent = build_agent(
            world.workspace,
            LoomConfig.model_validate({"approvals": {"mode": mode}}),
            EventBus(),
            provider=self.provider,
            home=world.loom_home,
            backend=backend,
            packages=[demo_package],
        )
        assert self.agent.tool_ctx.kernel is not None
        self.kernel = self.agent.tool_ctx.kernel
        self.session: SessionInfo = self.kernel.session

    async def start(self) -> None:
        await self.backend.start(self.session)

    async def close(self) -> None:
        await self.kernel.shutdown()
        await self.backend.stop()

    async def call(self, name: str, **args) -> ToolResultBlock:
        self.provider._script.extend([[ToolUseBlock("t", name, args)], [TextBlock("ok")]])
        await self.agent.run_turn("go")
        result = self.agent.history[-2].content[0]
        assert isinstance(result, ToolResultBlock)
        return result

    async def bash(self, command: str) -> str:
        result = await self.call("bash", command=command)
        assert isinstance(result.content, str)
        return result.content

    async def py(self, code: str) -> str:
        result = await self.call("execute", code=code)
        assert isinstance(result.content, str)
        return result.content


@pytest.fixture(params=BACKENDS)
async def box(request: pytest.FixtureRequest, world: World) -> AsyncIterator[Sandboxed]:
    box = Sandboxed(world, request.param(world))
    await box.start()
    yield box
    await box.close()


async def test_network_is_blocked(box: Sandboxed) -> None:
    out = await box.bash(
        "curl -sS -m 5 https://example.com >/dev/null && echo OPEN || echo blocked"
    )
    assert "blocked" in out and "OPEN" not in out
    out = await box.py(
        "import socket\n"
        "try:\n"
        "    socket.create_connection(('1.1.1.1', 53), timeout=3)\n"
        "    'OPEN'\n"
        "except OSError as e:\n"
        "    print('blocked', type(e).__name__)"
    )
    assert "blocked" in out and "OPEN" not in out


async def test_credentials_and_other_sessions_are_unreadable(box: Sandboxed) -> None:
    w = box.world
    for target in (w.ssh_key, w.other_log):
        out = await box.bash(f"cat {target} && echo READ || echo denied")
        assert "denied" in out and "PRIVATE" not in out and "secret stuff" not in out
        out = await box.py(f"open({str(target)!r}).read()")
        assert "Error" in out and "PRIVATE" not in out and "secret stuff" not in out


async def test_writes(box: Sandboxed) -> None:
    w, s = box.world, box.session
    assert "exit code: 0" in await box.bash("echo hi > inside.txt")
    assert (w.workspace / "inside.txt").read_text() == "hi\n"
    await box.py("open('from_kernel.txt', 'w').write('k')")
    assert (w.workspace / "from_kernel.txt").read_text() == "k"
    # This session's sandbox dir is writable; the log and anything outside are not.
    assert "exit code: 0" in await box.bash(f"echo t > {s.sandbox_dir / 'tmp' / 'scratch'}")
    log = s.session_dir / "session.jsonl"
    before = log.read_text()
    for target in (log, w.root / "outside.txt", w.home / "planted"):
        out = await box.bash(f"echo x >> {target} && echo WROTE || echo denied")
        assert "denied" in out, target
        out = await box.py(f"open({str(target)!r}, 'a').write('x')")
        assert "Error" in out, target
    assert log.read_text().startswith(before)  # only loom's own appends since
    assert not (w.root / "outside.txt").exists() and not (w.home / "planted").exists()


@pytest.mark.parametrize("make_backend", BACKENDS)
async def test_read_only_mode_blocks_workspace_writes(world: World, make_backend) -> None:
    box = Sandboxed(world, make_backend(world), mode="read-only")
    await box.start()
    try:
        # read-only mode with a sandbox: bash/execute still run (docs/03 table).
        out = await box.bash("echo x > nope.txt && echo WROTE || echo denied")
        assert "denied" in out
        out = await box.py("open('nope2.txt', 'w')")
        assert "Error" in out
        assert "exit code: 0" in await box.bash("ls")  # reading still works
        assert not (world.workspace / "nope.txt").exists()
    finally:
        await box.close()


async def test_broker_tools_and_inventory_survive_across_calls(box: Sandboxed) -> None:
    out = await box.py("rows = demo.search_logs(limit=5000)\nlen(rows)")
    assert out == "5000"
    out = await box.py("import collections\ncollections.Counter(r['level'] for r in rows)")
    assert out.startswith("Counter(")
    names = {v.name for v in await box.kernel.inventory()}
    assert "rows" in names


async def test_sub_agent_kernel_is_sandboxed_and_its_transcript_is_out_of_reach(
    box: Sandboxed,
) -> None:
    w, s = box.world, box.session
    transcript = s.session_dir / "agents" / "sa1.jsonl"
    probe = (
        "open('from_child.txt', 'w').write('c')\n"
        f"try:\n    open({str(transcript)!r}, 'a').write('x')\n    print('WROTE')\n"
        "except OSError:\n    print('denied')\n"
    )
    box.provider._script.extend(
        [
            [ToolUseBlock("p1", "execute", {"code": "print(agents.run('probe'))"})],
            [ToolUseBlock("c1", "execute", {"code": probe})],
            [TextBlock("child finished")],
            [TextBlock("ok")],
        ]
    )
    await box.agent.run_turn("go")
    result = box.agent.history[-2].content[0]
    assert isinstance(result, ToolResultBlock) and "child finished" in str(result.content)
    assert (w.workspace / "from_child.txt").read_text() == "c"
    records = SessionLog(transcript).read()
    result = next(r["result"] for r in records if r["kind"] == "tool_result")
    assert "denied" in result and "WROTE" not in result


async def test_cancel_stops_a_busy_sub_agent_kernel(box: Sandboxed) -> None:
    box.provider._script.extend(
        [
            [ToolUseBlock("p1", "execute", {"code": "agents.run('sleep')"})],
            [ToolUseBlock("c1", "execute", {"code": "import time\ntime.sleep(300)"})],
        ]
    )
    subagents = box.agent.tool_ctx.subagents
    assert subagents is not None
    turn = asyncio.create_task(box.agent.run_turn("go"))
    kernel = None
    async with asyncio.timeout(30):
        while kernel is None or kernel.state != "busy":
            await asyncio.sleep(0.05)
            for loop, _ in subagents._active.values():
                kernel = loop.tool_ctx.kernel
    await asyncio.sleep(0.5)  # inside time.sleep
    turn.cancel()
    await asyncio.wait_for(turn, 20)
    assert kernel is not None and not kernel.alive
    assert not subagents._active
    assert [r["status"] for r in box.agent.log.read() if r["kind"] == "subagent"] == ["cancelled"]
    # The sandbox still works for the parent.
    assert "still here" in await box.py("print('still here')")


async def test_children_run_in_their_own_process_group(box: Sandboxed) -> None:
    # perl is in macOS and in Debian slim; /bin/ps is setuid, which seatbelt won't exec.
    # `; true` makes bash fork perl instead of exec'ing it.
    out = await box.bash('perl -e \'print getppid(), " ", getpgrp(), "\\n"\'; true')
    parent, group = out.split()[:2]
    assert parent == group  # bash leads the group its children run in
    assert await box.py("import os\nos.getpgid(0) == os.getpid()") == "True"


# --- process control reaches into the sandbox ----------------------------------------


async def test_ctrl_c_interrupts_the_kernel_inside_the_sandbox(box: Sandboxed) -> None:
    import asyncio

    await box.py("kept = 'still here'")
    task = asyncio.create_task(box.kernel.execute("while True: pass", timeout_s=60))
    while box.kernel.state != "busy":
        await asyncio.sleep(0.01)
    await asyncio.sleep(0.5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert box.kernel.state == "idle"
    assert await box.py("kept") == "'still here'"


async def test_bash_timeout_and_cancel_kill_inside_the_sandbox(box: Sandboxed) -> None:
    import asyncio

    marker = box.world.workspace / "survived"
    box.provider._script.extend(
        [
            [ToolUseBlock("t", "bash", {"command": f"sleep 4; touch {marker}", "timeout_s": 1})],
            [TextBlock("ok")],
        ]
    )
    await box.agent.run_turn("go")
    result = box.agent.history[-2].content[0]
    assert isinstance(result, ToolResultBlock) and "timed out after 1s" in str(result.content)

    box.provider._script.extend(
        [[ToolUseBlock("t2", "bash", {"command": f"sleep 4; touch {marker}"})], [TextBlock("x")]]
    )
    task = asyncio.create_task(box.agent.run_turn("go"))
    await asyncio.sleep(1.5)
    task.cancel()
    await task
    await asyncio.sleep(4)
    assert not marker.exists()  # neither sleep survived (in the container, too)


@pytest.mark.docker
async def test_docker_read_only_switch_recreates_the_container(world: World) -> None:
    box = Sandboxed(world, docker(world))
    await box.start()
    try:
        assert "exit code: 0" in await box.bash("echo x > a.txt")
        backend = box.backend
        first = backend.container  # type: ignore[attr-defined]
        box.agent.tool_ctx.workspace_writable = False
        box.kernel.workspace_writable = False
        out = await box.bash("echo y > b.txt && echo WROTE || echo denied")
        assert "denied" in out and backend.container == first  # type: ignore[attr-defined]
        # Same name, new container: check it really is mounted read-only now.
        # Check the workspace mount by path: docker doesn't list mounts in a fixed order.
        inspect = subprocess.run(
            ["docker", "inspect", "-f", "{{range .Mounts}}{{.Destination}}={{.RW}} {{end}}", first],
            capture_output=True, text=True,
        )  # fmt: skip
        rw = dict(m.rsplit("=", 1) for m in inspect.stdout.split())
        assert rw[str(world.workspace.resolve())] == "false"
    finally:
        await box.close()
    gone = subprocess.run(["docker", "inspect", first], capture_output=True)
    assert gone.returncode != 0  # stop() removed it


@pytest.mark.docker
async def test_docker_stale_containers_are_removed(world: World) -> None:
    backend = docker(world)
    dead_pid = 999_999
    name = "loom-stale-test"
    subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    subprocess.run(
        ["docker", "run", "-d", "--name", name, "--label", "loom.sandbox=1",
         "--label", f"loom.host_pid={dead_pid}", IMAGE, "sleep", "60"],
        check=True, capture_output=True,
    )  # fmt: skip
    box = Sandboxed(world, backend)
    await box.start()
    await box.close()
    assert subprocess.run(["docker", "inspect", name], capture_output=True).returncode != 0


async def test_docker_startup_errors_are_clear(world: World, monkeypatch) -> None:
    from loom.config import SandboxConfig
    from loom.sandbox.base import SandboxUnavailable
    from loom.sandbox.docker import DockerBackend

    session = SessionInfo("s", world.workspace, world.root / "session")
    monkeypatch.setenv("PATH", str(world.root))  # no docker CLI at all
    with pytest.raises(SandboxUnavailable, match="--sandbox seatbelt"):
        await DockerBackend(SandboxConfig(), loom_home=world.loom_home).start(session)
    monkeypatch.undo()
    if not _image_exists():
        pytest.skip("needs Docker")
    cfg = SandboxConfig.model_validate({"docker": {"image": "loom-sandbox:does-not-exist"}})
    with pytest.raises(SandboxUnavailable, match="loom sandbox setup --docker"):
        await DockerBackend(cfg, loom_home=world.loom_home).start(session)


# --- seatbelt profile and setup (no sandbox needed) -----------------------------------


def test_seatbelt_profile_rendering(world: World) -> None:
    from loom.sandbox.seatbelt import render_profile, sbpl_string

    session = SessionInfo("s", world.workspace, world.loom_home / "sessions" / "p" / "s")
    profile = render_profile(home=world.home, loom_home=world.loom_home, session=session,
                             extra_deny=[Path('/tmp/odd "dir"')])  # fmt: skip
    assert f"(subpath {sbpl_string(world.home / '.ssh')})" in profile
    assert '(subpath "/private/tmp/odd \\"dir\\"")' in profile  # resolved and escaped
    assert f"(subpath {sbpl_string(world.workspace)})" in profile.split("(allow file-write*")[1]
    assert "(allow file-read-metadata\n  (literal" in profile
    assert "(deny network*)" in profile

    read_only = render_profile(home=world.home, loom_home=world.loom_home, session=session,
                               workspace_writable=False)  # fmt: skip
    assert str(world.workspace) not in read_only.split("(allow file-write*")[1]

    # Only ancestors inside ~/.loom are listed; none for paths outside it. (The rule is
    # never rendered empty: `(allow file-read-metadata)` alone would allow everything.)
    from loom.sandbox.seatbelt import _ancestors

    assert _ancestors(world.root / "elsewhere" / "s", stop_above=world.loom_home) == set()
    assert _ancestors(world.loom_home / "a" / "b", stop_above=world.loom_home) == {
        world.loom_home / "a",
        world.loom_home,
    }


async def test_seatbelt_missing_venv_says_how_to_fix(world: World) -> None:
    from loom.sandbox.base import SandboxUnavailable

    backend = SeatbeltBackend(loom_home=world.loom_home, home=world.home)
    with pytest.raises(SandboxUnavailable, match="loom sandbox setup"):
        await backend.start(SessionInfo("s", world.workspace, world.root / "s"))


def test_sandbox_setup_commands(world: World, monkeypatch) -> None:
    from loom.config import SandboxConfig
    from loom.sandbox import setup

    ran: list[list[str]] = []
    monkeypatch.setattr(setup, "_run", lambda argv: ran.append(argv) or 0)
    monkeypatch.setattr(setup.shutil, "which", lambda name: f"/bin/{name}")
    cfg = SandboxConfig.model_validate({"python_packages": ["pandas"], "docker": {"image": "x:1"}})
    assert setup.setup_seatbelt(cfg, world.loom_home) == 0
    venv = str(world.loom_home / "sandbox-venv")
    assert ran[0][:4] == ["/bin/uv", "venv", "--python", "3.12"] and ran[0][-1] == venv
    assert ran[1][-1] == "pandas" and ran[1][ran[1].index("--python") + 1] == venv + "/bin/python"

    ran.clear()
    monkeypatch.setattr(setup.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0))
    assert setup.setup_docker(cfg) == 0
    assert ran[0][:4] == ["docker", "build", "-t", "x:1"]
    assert "PYTHON_PACKAGES=pandas" in ran[0]
