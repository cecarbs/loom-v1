"""The `none` backend: no isolation, for developing loom itself (docs/05). Processes still
get the allowlisted environment and their own process group."""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from loom.sandbox.base import STREAM_LIMIT, Purpose, SessionInfo

EXIT_POLL_S = 0.02


class LocalProcess:
    """A host subprocess that leads its own process group."""

    def __init__(self, proc: asyncio.subprocess.Process) -> None:
        assert proc.stdout is not None and proc.stderr is not None
        self._proc = proc
        self.pid = proc.pid
        self.stdin = proc.stdin
        self.stdout = proc.stdout
        self.stderr = proc.stderr

    async def wait(self) -> int:
        """Wait for the leader to exit. asyncio's own `wait()` also waits for the pipes to
        close, which never happens while a background child still holds them."""
        while (code := self._proc.returncode) is None:
            await asyncio.sleep(EXIT_POLL_S)
        return code

    @property
    def returncode(self) -> int | None:
        return self._proc.returncode

    def send_signal(self, sig: int) -> None:
        with contextlib.suppress(ProcessLookupError):
            os.kill(self.pid, sig)

    def kill(self) -> None:
        # The group outlives its leader while children remain, so this also works after
        # the leader exited (e.g. `sleep 100 &` in bash).
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(self.pid, signal.SIGKILL)


class NoneBackend:
    name = "none"
    python = sys.executable

    async def start(self, session: SessionInfo) -> None:
        pass

    async def spawn(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
        stdin: bool = False,
        purpose: Purpose,
        workspace_writable: bool = True,
    ) -> LocalProcess:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            cwd=cwd,
            env=dict(env),
            stdin=asyncio.subprocess.PIPE if stdin else asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,  # terminal Ctrl+C reaches only loom
            limit=STREAM_LIMIT,
        )
        return LocalProcess(proc)

    async def stop(self) -> None:
        pass

    def describe(self) -> str:
        return "none"
