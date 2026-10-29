import ast
import asyncio
import os
import shutil
import subprocess
import sys
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from loom.core import events as ev
from loom.core.events import EventBus
from loom.kernel import runner
from loom.kernel.client import KernelClient
from loom.sandbox.base import SessionInfo, sandbox_env
from loom.sandbox.none import NoneBackend
from loom.session.store import SessionLog

RUNNER = Path(runner.__file__)


class Recorder:
    def __init__(self, bus: EventBus) -> None:
        self.events: list[ev.AgentEvent] = []
        bus.subscribe(self)

    async def __call__(self, event: ev.AgentEvent) -> None:
        self.events.append(event)

    def states(self) -> list[str]:
        return [e.state for e in self.events if isinstance(e, ev.KernelStatus)]


def make_kernel(tmp_path: Path, **settings: int) -> tuple[KernelClient, Recorder, SessionLog]:
    workspace = tmp_path / "ws"
    workspace.mkdir(exist_ok=True)
    session = SessionInfo("s1", workspace, tmp_path / "session")
    session.sandbox_dir.mkdir(parents=True, exist_ok=True)
    bus = EventBus()
    log = SessionLog(session.session_dir / "session.jsonl")
    kernel = KernelClient(
        NoneBackend(),
        session,
        env=sandbox_env(session),
        bus=bus,
        session_log=log,
        settings={"repr_max": 2000, "out_max_chars": 12000, **settings},
    )
    return kernel, Recorder(bus), log


@pytest.fixture
async def kernel(tmp_path: Path) -> AsyncIterator[KernelClient]:
    k, _, _ = make_kernel(tmp_path)
    yield k
    await k.shutdown()


# --- runner (static) ----------------------------------------------------------------


def test_runner_imports_only_stdlib() -> None:
    tree = ast.parse(RUNNER.read_text())
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "relative import in runner.py"
            modules.add((node.module or "").split(".")[0])
    modules.discard("__future__")
    assert "loom" not in modules
    assert modules <= sys.stdlib_module_names, modules - sys.stdlib_module_names


def test_compose_text() -> None:
    assert runner.compose_text("", None, None) == ""
    assert runner.compose_text("out", "42", None) == "out\n42"
    assert runner.compose_text("out\n", None, "Traceback") == "out\nTraceback"


@pytest.mark.skipif(shutil.which("python3.11") is None, reason="needs python3.11")
def test_runner_runs_on_python_311(tmp_path: Path) -> None:
    lines = (
        '{"type": "init", "settings": {}}\n'
        '{"type": "execute", "id": "e1", "code": "x = [1, 2]\\nprint(\'hi\')\\nx"}\n'
        '{"type": "shutdown"}\n'
    )
    python = shutil.which("python3.11")
    assert python is not None
    done = subprocess.run(
        [python, "-I", str(RUNNER)], input=lines, capture_output=True, text=True, timeout=30
    )
    assert '"type": "ready"' in done.stdout
    assert '"text": "hi\\n"' in done.stdout and '"value_repr": "[1, 2]"' in done.stdout


# --- execution ----------------------------------------------------------------------


async def test_variables_persist_across_executions(kernel: KernelClient) -> None:
    first = await kernel.execute("rows = [{'n': i} for i in range(10_000)]", timeout_s=10)
    assert first.status == "ok" and first.value_repr is None
    second = await kernel.execute("print(len(rows))\nsum(r['n'] for r in rows)", timeout_s=10)
    assert second.status == "ok"
    assert second.output == "10000\n" and second.value_repr == "49995000"
    assert kernel.vars == 1


async def test_output_streams_before_result(kernel: KernelClient) -> None:
    chunks: list[str] = []

    async def on_output(text: str) -> None:
        chunks.append(text)

    code = "import sys\nfor i in range(3): print(i)\nsys.stderr.write('err\\n')"
    result = await kernel.execute(code, timeout_s=10, on_output=on_output)
    assert "".join(chunks) == result.output
    assert "0\n1\n2\n" in result.output and "err\n" in result.output


async def test_subprocess_writing_to_fd1_does_not_corrupt_protocol(kernel: KernelClient) -> None:
    code = (
        "import os, subprocess\n"
        "subprocess.run(['echo', 'from child'])\n"
        'os.write(1, b\'{"type": "execute_result", "id": "e99"}\\n\')\n'
        "os.write(2, b'raw stderr\\n')\n"
        "'still fine'"
    )
    result = await kernel.execute(code, timeout_s=10)
    assert result.status == "ok" and result.value_repr == "'still fine'"
    assert "from child\n" in result.output
    assert '{"type": "execute_result", "id": "e99"}' in result.output
    assert "raw stderr" in result.output
    # The protocol still works afterwards.
    assert (await kernel.execute("1 + 1", timeout_s=10)).value_repr == "2"


async def test_stdin_is_not_the_protocol(kernel: KernelClient) -> None:
    result = await kernel.execute("import sys\nsys.stdin.read()", timeout_s=10)
    assert result.value_repr == "''"


async def test_errors_have_cell_tracebacks(kernel: KernelClient) -> None:
    result = await kernel.execute("def f():\n    return {}['host']\nf()", timeout_s=10)
    assert result.status == "error" and result.error is not None
    assert result.error.ename == "KeyError"
    tb = result.error.traceback
    assert '<cell e1>", line 3' in tb and "return {}['host']" in tb
    assert "runner.py" not in tb

    syntax = await kernel.execute("def (", timeout_s=10)
    assert syntax.error is not None and syntax.error.ename == "SyntaxError"

    exiting = await kernel.execute("import sys; sys.exit(3)", timeout_s=10)
    assert exiting.error is not None and exiting.error.ename == "SystemExit"
    assert kernel.state == "idle"  # sys.exit doesn't end the kernel


async def test_top_level_await(kernel: KernelClient) -> None:
    result = await kernel.execute(
        "import asyncio\nawait asyncio.sleep(0.01)\nawait asyncio.gather(*(asyncio.sleep(0, i) "
        "for i in range(3)))",
        timeout_s=10,
    )
    assert result.status == "ok" and result.value_repr == "[0, 1, 2]"


async def test_repr_is_limited(kernel: KernelClient) -> None:
    result = await kernel.execute("list(range(100_000))", timeout_s=10)
    assert result.value_repr is not None and result.value_repr.endswith("...]")
    assert len(result.value_repr) < 400
    long_str = await kernel.execute("'x' * 50_000", timeout_s=10)
    assert long_str.value_repr is not None and len(long_str.value_repr) <= 2100


async def test_large_output_kept_in_kernel_as_out(tmp_path: Path) -> None:
    kernel, _, _ = make_kernel(tmp_path, out_max_chars=100)
    try:
        small = await kernel.execute("print('short')", timeout_s=10)
        assert small.out_ref is None
        big = await kernel.execute("print('y' * 500)", timeout_s=10)
        assert big.out_ref == 2
        check = await kernel.execute("len(_out[2]), sorted(_out)", timeout_s=10)
        assert check.value_repr == "(501, [2])"
    finally:
        await kernel.shutdown()


async def test_inventory(kernel: KernelClient) -> None:
    await kernel.execute(
        "import json\nrows = [1, 2, 3]\nname = 'x'\n_hidden = 1\ndef helper(): pass\n"
        "class Shape: shape = (4, 7)\ngrid = Shape()",
        timeout_s=10,
    )
    variables = {v.name: v for v in await kernel.inventory()}
    assert set(variables) == {"rows", "name", "helper", "Shape", "grid"}
    assert variables["rows"].summary == "len=3" and variables["rows"].type == "list"
    assert variables["name"].summary == "len=1"
    assert variables["grid"].summary == "4x7"


# --- lifecycle ----------------------------------------------------------------------


async def test_lazy_start_and_status_events(tmp_path: Path) -> None:
    kernel, rec, log = make_kernel(tmp_path)
    try:
        assert kernel.state == "off" and kernel.pid is None
        await kernel.execute("x = 1", timeout_s=10)
        assert rec.states() == ["starting", "idle", "busy", "idle"]
        idle = [e for e in rec.events if isinstance(e, ev.KernelStatus)][-1]
        assert idle.vars == 1
        assert (kernel.session.sandbox_dir / "runner.py").read_text() == RUNNER.read_text()
    finally:
        await kernel.shutdown()
    kinds = [r["kind"] for r in log.read()]
    assert kinds == ["kernel_start", "kernel_execute", "kernel_exit"]
    assert rec.states()[-1] == "dead"
    # A clean shutdown is not reported to the user as a crash.
    assert not [e for e in rec.events if isinstance(e, ev.Notice)]


async def test_kernel_runs_in_its_own_process_group(kernel: KernelClient) -> None:
    result = await kernel.execute("import os\nos.getpgid(0) == os.getpid()", timeout_s=10)
    assert result.value_repr == "True"
    assert kernel.pid is not None and os.getpgid(kernel.pid) != os.getpgid(0)


async def test_kernel_env_is_allowlisted(kernel: KernelClient, monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-not-leak")
    kernel.env = sandbox_env(kernel.session)
    result = await kernel.execute(
        "import os\nsorted(k for k in os.environ if k not in ('__CF_USER_TEXT_ENCODING',))",
        timeout_s=10,
    )
    assert result.value_repr is not None and "ANTHROPIC" not in result.value_repr
    home = await kernel.execute("os.environ['HOME']", timeout_s=10)
    assert home.value_repr == repr(str(kernel.session.sandbox_dir / "home"))


async def test_interrupt_keeps_kernel_alive(kernel: KernelClient) -> None:
    await kernel.execute("kept = 'yes'", timeout_s=10)
    task = asyncio.create_task(kernel.execute("while True: pass", timeout_s=30))
    while kernel.state != "busy":
        await asyncio.sleep(0.01)
    await asyncio.sleep(0.2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert kernel.state == "idle"
    after = await kernel.execute("kept", timeout_s=10)
    assert after.value_repr == "'yes'" and not after.restarted


async def test_timeout_kills_and_next_execute_restarts(tmp_path: Path) -> None:
    kernel, _, log = make_kernel(tmp_path)
    try:
        await kernel.execute("x = 1", timeout_s=10)
        old_pid = kernel.pid
        result = await kernel.execute("import time\nprint('started')\ntime.sleep(30)", timeout_s=1)
        assert result.status == "timeout" and "started" in result.output
        assert kernel.state == "dead"
        assert old_pid is not None
        with pytest.raises(ProcessLookupError):
            os.kill(old_pid, 0)

        restarted = await kernel.execute("'x' in dir()", timeout_s=10)
        assert restarted.restarted and restarted.value_repr == "False"
        assert kernel.pid != old_pid
        again = await kernel.execute("1", timeout_s=10)
        assert not again.restarted  # the notice is given once
    finally:
        await kernel.shutdown()
    kill = next(r for r in log.read() if r["kind"] == "kernel_kill")
    assert "timed out" in kill["reason"]


async def test_crash_is_reported_and_restarts(kernel: KernelClient) -> None:
    rec = Recorder(kernel.bus)
    await kernel.execute("x = 1", timeout_s=10)
    died = await kernel.execute("import os\nos._exit(7)", timeout_s=10)
    assert died.status == "died" and "code 7" in died.output
    notices = [e.text for e in rec.events if isinstance(e, ev.Notice)]
    assert notices and "exited unexpectedly" in notices[0]
    nxt = await kernel.execute("1", timeout_s=10)
    assert nxt.status == "ok" and nxt.restarted


async def test_reset(kernel: KernelClient) -> None:
    await kernel.execute("x = 1", timeout_s=10)
    await kernel.reset()
    assert kernel.state == "idle" and kernel.vars == 0
    result = await kernel.execute("'x' in dir()", timeout_s=10)
    assert result.restarted and result.value_repr == "False"


async def test_startup_failure_is_a_result_not_a_crash(tmp_path: Path) -> None:
    kernel, _, _ = make_kernel(tmp_path)
    kernel.backend.python = str(tmp_path / "no-such-python")  # pyright: ignore[reportAttributeAccessIssue]
    result = await kernel.execute("1", timeout_s=10)
    assert result.status == "died" and "failed to start" in result.output
    assert kernel.state == "dead"
