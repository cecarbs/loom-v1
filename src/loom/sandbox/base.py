"""The sandbox backend interface (docs/05). Every model-driven process (bash, the kernel
runner) is started through a backend, in its own session / process group."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol

if TYPE_CHECKING:
    from loom.config import SandboxConfig

Purpose = Literal["bash", "kernel"]
# Receives each connection attempt seen by the allowlist proxy (logged as `network`).
NetworkCallback = Callable[[dict[str, Any]], None]

# Host variables a sandboxed process may see (plus `[sandbox] env_passthrough`). HOME and
# TMPDIR are replaced with the session's sandbox dirs.
ENV_ALLOWLIST = ("PATH", "LANG", "TERM")

# asyncio's default line limit (64 KiB) is too small for kernel protocol lines.
STREAM_LIMIT = 64 * 1024 * 1024


@dataclass(frozen=True)
class SessionInfo:
    id: str
    workspace: Path
    session_dir: Path

    @property
    def sandbox_dir(self) -> Path:
        return self.session_dir / "sandbox"

    @property
    def out_dir(self) -> Path:
        return self.session_dir / "out"


class SandboxProcess(Protocol):
    pid: int
    stdin: asyncio.StreamWriter | None
    stdout: asyncio.StreamReader
    stderr: asyncio.StreamReader

    async def wait(self) -> int:
        """Wait for the process itself to exit (not for children that inherited its pipes)."""
        ...

    @property
    def returncode(self) -> int | None: ...

    def send_signal(self, sig: int) -> None: ...

    def kill(self) -> None:
        """SIGKILL the whole process group."""
        ...


class SandboxBackend(Protocol):
    name: str
    python: str  # interpreter that runs the kernel runner inside this sandbox
    on_network: NetworkCallback | None  # set by the app before start()

    async def start(self, session: SessionInfo) -> None: ...

    async def spawn(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
        stdin: bool = False,
        purpose: Purpose,
        workspace_writable: bool = True,
    ) -> SandboxProcess: ...

    async def stop(self) -> None: ...

    def describe(self) -> str: ...


class SandboxUnavailable(Exception):
    pass


def sandbox_env(
    session: SessionInfo,
    passthrough: Sequence[str] = (),
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Environment for sandboxed processes, built from an allowlist: the host environment
    is never inherited wholesale (so API keys in it stay out)."""
    environ = os.environ if environ is None else environ
    env = {k: environ[k] for k in (*ENV_ALLOWLIST, *passthrough) if k in environ}
    env.setdefault("PATH", "/usr/bin:/bin:/usr/sbin:/sbin")
    env.setdefault("LANG", "en_US.UTF-8")
    env["HOME"] = str(session.sandbox_dir / "home")
    env["TMPDIR"] = str(session.sandbox_dir / "tmp")
    return env


def proxy_env(url: str) -> dict[str, str]:
    """Point HTTP clients (curl, pip, requests, urllib) at the allowlist proxy."""
    names = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY")
    return {k: url for name in names for k in (name, name.lower())}


def create_backend(cfg: SandboxConfig, *, loom_home: Path | None = None) -> SandboxBackend:
    """The configured backend. Availability (daemon, venv, image) is checked in start()."""
    from loom.session.store import loom_home as default_loom_home

    loom_home = loom_home or default_loom_home()
    match cfg.backend:
        case "none":
            from loom.sandbox.none import NoneBackend

            return NoneBackend()
        case "seatbelt":
            from loom.sandbox.seatbelt import SeatbeltBackend

            return SeatbeltBackend(
                loom_home=loom_home,
                deny_read=cfg.deny_read,
                allowlist=cfg.allowlist if cfg.network == "allowlist" else None,
            )
        case "docker":
            from loom.sandbox.docker import DockerBackend

            return DockerBackend(cfg, loom_home=loom_home)
