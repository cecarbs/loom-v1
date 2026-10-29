"""KernelClient: starts the runner through the sandbox backend and speaks its protocol
(docs/04 "Kernel lifecycle", "Protocol"). One kernel per session, started lazily."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import shutil
import signal
import time
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from typing import Any

from loom.core import events as ev
from loom.core.events import EventBus
from loom.kernel.protocol import ExecutionResult, KernelState, VarInfo, encode
from loom.sandbox.base import SandboxBackend, SandboxProcess, SessionInfo
from loom.session.store import SessionLog

log = logging.getLogger(__name__)

RUNNER_PATH = Path(__file__).with_name("runner.py")
STARTUP_TIMEOUT_S = 30.0
INTERRUPT_GRACE_S = 3.0  # after SIGINT, wait this long for a reply before killing
SHUTDOWN_GRACE_S = 2.0
INVENTORY_TIMEOUT_S = 5.0
STDERR_TAIL_CHARS = 4000

OutputCallback = Callable[[str], Awaitable[None]]


class KernelDied(Exception):
    pass


class KernelClient:
    def __init__(
        self,
        backend: SandboxBackend,
        session: SessionInfo,
        *,
        env: Mapping[str, str],
        bus: EventBus,
        session_log: SessionLog,
        settings: Mapping[str, Any],
        workspace_writable: bool = True,
    ) -> None:
        self.backend = backend
        self.session = session
        self.env = dict(env)
        self.bus = bus
        self.log = session_log
        self.settings = dict(settings)
        self.workspace_writable = workspace_writable

        self.state: KernelState = "off"
        self.vars: int | None = None
        self.pid: int | None = None
        self._proc: SandboxProcess | None = None
        self._reader: asyncio.Task[None] | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._stderr_tail = ""
        self._ready: asyncio.Future[dict[str, Any]] | None = None
        self._pending: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._outputs: dict[str, OutputCallback] = {}
        self._lock = asyncio.Lock()  # one execution at a time
        self._counter = 0
        self._was_ready = False  # the current process got as far as `ready`
        self._expected_exit = False
        self._restart_pending = False  # a kernel with state died; tell the model next time

    # --- lifecycle -------------------------------------------------------------------

    @property
    def alive(self) -> bool:
        return self.state in ("starting", "idle", "busy")

    async def start(self) -> None:
        """Start a kernel if none is running (no-op otherwise)."""
        if self.alive:
            return
        await self._set_state("starting")
        sandbox_dir = self.session.sandbox_dir
        sandbox_dir.mkdir(parents=True, exist_ok=True)
        runner = sandbox_dir / "runner.py"
        shutil.copyfile(RUNNER_PATH, runner)

        self._was_ready = False
        self._expected_exit = False
        self._stderr_tail = ""
        loop = asyncio.get_running_loop()
        self._ready = loop.create_future()
        try:
            proc = await self.backend.spawn(
                [self.backend.python, "-I", str(runner)],
                cwd=self.session.workspace,
                env=self.env,
                stdin=True,
                purpose="kernel",
                workspace_writable=self.workspace_writable,
            )
        except OSError as e:
            await self._set_state("dead")
            raise KernelDied(f"kernel failed to start: {e}") from e
        self._proc = proc
        self._reader = asyncio.create_task(self._read(proc), name="kernel-reader")
        self._stderr_task = asyncio.create_task(self._drain_stderr(proc), name="kernel-stderr")
        try:
            await self._send({"type": "init", "catalog": [], "settings": self.settings})
            ready = await asyncio.wait_for(self._ready, STARTUP_TIMEOUT_S)
        except (TimeoutError, KernelDied, ConnectionError) as e:
            await self._kill("startup failed")
            detail = self._stderr_tail.strip() or str(e) or type(e).__name__
            raise KernelDied(f"kernel failed to start: {detail}") from e
        self._was_ready = True
        self.pid = int(ready.get("pid", proc.pid))
        self.log.append(
            "kernel_start", pid=self.pid, python=ready.get("python"), backend=self.backend.name
        )
        self.vars = 0
        await self._set_state("idle")

    async def reset(self) -> None:
        """Restart with an empty namespace (`/kernel reset`)."""
        async with self._lock:
            await self.shutdown()
            await self.start()

    async def shutdown(self) -> None:
        proc = self._proc
        if proc is None or not self.alive:
            return
        self._expected_exit = True
        with contextlib.suppress(ConnectionError, KernelDied):
            await self._send({"type": "shutdown"})
        try:
            await asyncio.wait_for(asyncio.shield(self._wait_reader()), SHUTDOWN_GRACE_S)
        except TimeoutError:
            await self._kill("shutdown")

    async def _kill(self, reason: str) -> None:
        proc = self._proc
        if proc is None:
            return
        self._expected_exit = True
        self.log.append("kernel_kill", pid=self.pid, reason=reason)
        proc.kill()
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(asyncio.shield(self._wait_reader()), SHUTDOWN_GRACE_S)
        if self.state != "dead":  # the reader didn't notice; mark it ourselves
            await self._exited(proc, proc.returncode)

    async def _wait_reader(self) -> None:
        if self._reader is not None:
            await self._reader

    # --- execution -------------------------------------------------------------------

    async def execute(
        self, code: str, *, timeout_s: float, on_output: OutputCallback | None = None
    ) -> ExecutionResult:
        async with self._lock:
            if not self.alive:
                try:
                    await self.start()
                except KernelDied as e:
                    return ExecutionResult("died", str(e), 0)
            restarted, self._restart_pending = self._restart_pending, False
            self._counter += 1
            exec_id = f"e{self._counter}"
            chunks: list[str] = []

            async def collect(text: str) -> None:
                chunks.append(text)
                if on_output is not None:
                    await on_output(text)

            future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
            self._pending[exec_id] = future
            self._outputs[exec_id] = collect
            start = time.monotonic()
            await self._set_state("busy")
            try:
                await self._send({"type": "execute", "id": exec_id, "code": code})
                msg = await asyncio.wait_for(asyncio.shield(future), timeout_s)
            except TimeoutError:
                await self._kill(f"execute {exec_id} timed out after {timeout_s:g}s")
                result = ExecutionResult(
                    "timeout", "".join(chunks), _ms(start), restarted=restarted
                )
            except (KernelDied, ConnectionError) as e:
                output = "".join(chunks)
                result = ExecutionResult(
                    "died",
                    output + ("\n" if output else "") + str(e),
                    _ms(start),
                    restarted=restarted,
                )
            except asyncio.CancelledError:
                await self._interrupt(future)
                self.log.append("kernel_execute", id=exec_id, code=code, status="interrupted")
                raise
            else:
                result = ExecutionResult.from_message(msg, "".join(chunks), restarted)
                self.vars = msg.get("vars", self.vars)
            finally:
                self._pending.pop(exec_id, None)
                self._outputs.pop(exec_id, None)
                if self.state == "busy":
                    await self._set_state("idle")
            self.log.append(
                "kernel_execute",
                id=exec_id,
                code=code,
                status=result.status,
                duration_ms=result.duration_ms,
                ename=result.error.ename if result.error else None,
                out_ref=result.out_ref,
                restarted=restarted,
            )
            return result

    async def _interrupt(self, future: asyncio.Future[dict[str, Any]]) -> None:
        """SIGINT the runner and wait for its reply; kill it if none comes in time."""
        if self._proc is None or not self.alive:
            return
        self.log.append("kernel_interrupt", pid=self.pid)
        self._proc.send_signal(signal.SIGINT)
        try:
            await asyncio.wait_for(asyncio.shield(future), INTERRUPT_GRACE_S)
        except (TimeoutError, KernelDied, ConnectionError):
            await self._kill("no reply to interrupt")
        except asyncio.CancelledError:  # cancelled again while waiting: don't wait
            await self._kill("interrupt cancelled")
            raise

    async def inventory(self) -> list[VarInfo]:
        if not self.alive or self.state == "starting":
            return []
        self._counter += 1
        req_id = f"i{self._counter}"
        future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._pending[req_id] = future
        try:
            await self._send({"type": "inventory", "id": req_id})
            msg = await asyncio.wait_for(future, INVENTORY_TIMEOUT_S)
        except (TimeoutError, KernelDied, ConnectionError):
            return []
        finally:
            self._pending.pop(req_id, None)
        out = [
            VarInfo(str(v.get("name")), str(v.get("type")), str(v.get("summary", "")))
            for v in msg.get("vars", [])
            if isinstance(v, dict)
        ]
        self.vars = len(out)
        return out

    # --- transport -------------------------------------------------------------------

    async def _send(self, msg: dict[str, Any]) -> None:
        proc = self._proc
        if proc is None or proc.stdin is None or proc.stdin.is_closing():
            raise KernelDied("kernel is not running")
        proc.stdin.write(encode(msg))
        await proc.stdin.drain()

    async def _read(self, proc: SandboxProcess) -> None:
        try:
            while line := await proc.stdout.readline():
                try:
                    msg = json.loads(line)
                except ValueError:
                    log.warning("kernel sent a non-JSON line: %r", line[:200])
                    continue
                if isinstance(msg, dict):
                    await self._dispatch(msg)
        except (ValueError, ConnectionError) as e:  # line over the limit, pipe errors
            log.warning("kernel stream failed: %s", e)
            proc.kill()
        returncode = await proc.wait()
        if self._stderr_task is not None:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(asyncio.shield(self._stderr_task), 1.0)
        await self._exited(proc, returncode)

    async def _dispatch(self, msg: dict[str, Any]) -> None:
        match msg.get("type"):
            case "ready":
                if self._ready is not None and not self._ready.done():
                    self._ready.set_result(msg)
            case "stream":
                callback = self._outputs.get(str(msg.get("id")))
                text = msg.get("text")
                if callback is not None and isinstance(text, str):
                    await callback(text)
            case "execute_result" | "inventory_result":
                future = self._pending.get(str(msg.get("id")))
                if future is not None and not future.done():
                    future.set_result(msg)
            case other:
                log.warning("unexpected kernel message type %r", other)

    async def _drain_stderr(self, proc: SandboxProcess) -> None:
        """The runner's own fd 2 before redirection (startup errors, runner crashes)."""
        while chunk := await proc.stderr.read(4096):
            text = self._stderr_tail + chunk.decode("utf-8", "replace")
            self._stderr_tail = text[-STDERR_TAIL_CHARS:]

    async def _exited(self, proc: SandboxProcess, returncode: int | None) -> None:
        if proc is not self._proc or self.state == "dead":
            return
        expected = self._expected_exit
        if self._was_ready:
            self._restart_pending = True
        detail = self._stderr_tail.strip()
        reason = f"kernel exited (code {returncode})" + (f":\n{detail}" if detail else "")
        for future in [self._ready, *self._pending.values()]:
            if future is not None and not future.done():
                future.set_exception(KernelDied(reason))
                future.exception()  # mark retrieved: nobody may be awaiting it
        self.log.append("kernel_exit", pid=self.pid, returncode=returncode, expected=expected)
        self.pid = None
        self.vars = None
        await self._set_state("dead")
        if not expected and self._was_ready:
            await self.bus.emit(
                ev.Notice(
                    f"The kernel exited unexpectedly (code {returncode}); its variables "
                    "are gone. It restarts on the next execute."
                )
            )

    async def _set_state(self, state: KernelState) -> None:
        self.state = state
        await self.bus.emit(ev.KernelStatus(state, self.vars))


def _ms(start: float) -> int:
    return int((time.monotonic() - start) * 1000)
