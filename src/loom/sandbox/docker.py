"""The `docker` backend (the default): one container per session, driven through the docker
CLI (docs/05 "docker"). No Docker Python SDK.

Killing the host `docker exec` process does not stop the process inside the container, so
every exec'd process writes its in-container pid to `<sandbox_dir>/pids/<token>` first.
Signals and kills are then delivered inside the container. Exec'd processes lead their own
session and process group there, so `kill -- -PID` takes down their children too.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import re
import shutil
import subprocess
import time
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path

from loom.config import SandboxConfig
from loom.sandbox.base import Purpose, SandboxUnavailable, SessionInfo
from loom.sandbox.none import LocalProcess, spawn_local

LABEL = "loom.sandbox"
PID_WAIT_S = 2.0
DOCKER_TIMEOUT_S = 30.0
# `sh` writes its pid, then becomes the real command.
EXEC_WRAPPER = 'echo $$ > "$0"; exec "$@"'


class DockerProcess:
    def __init__(self, local: LocalProcess, backend: DockerBackend, pidfile: Path) -> None:
        self._local = local
        self._backend = backend
        self._pidfile = pidfile
        self.pid = local.pid  # the host-side `docker exec`
        self.stdin = local.stdin
        self.stdout = local.stdout
        self.stderr = local.stderr

    async def wait(self) -> int:
        return await self._local.wait()  # docker exec exits with the command's status

    @property
    def returncode(self) -> int | None:
        return self._local.returncode

    def container_pid(self) -> int | None:
        deadline = time.monotonic() + PID_WAIT_S
        while True:
            with contextlib.suppress(OSError, ValueError):
                return int(self._pidfile.read_text().strip())
            if time.monotonic() > deadline or self._local.returncode is not None:
                return None
            time.sleep(0.02)

    def send_signal(self, sig: int) -> None:
        if (pid := self.container_pid()) is not None:
            self._backend.exec_quiet(f"kill -{int(sig)} {pid}")

    def kill(self) -> None:
        if (pid := self.container_pid()) is not None:
            self._backend.exec_quiet(f"kill -9 -- -{pid} 2>/dev/null; kill -9 {pid} 2>/dev/null")
        self._local.kill()
        with contextlib.suppress(OSError):
            self._pidfile.unlink()


class DockerBackend:
    name = "docker"
    python = "python"  # the image's interpreter

    def __init__(self, cfg: SandboxConfig, *, loom_home: Path) -> None:
        self.cfg = cfg
        self.image = cfg.docker.image
        self.loom_home = loom_home
        self.container: str | None = None
        self._session: SessionInfo | None = None
        self._writable: bool | None = None
        self._lock = asyncio.Lock()

    # --- lifecycle ---------------------------------------------------------------------

    async def start(self, session: SessionInfo) -> None:
        """Check Docker and the image; never fall back to a weaker backend (docs/00)."""
        if shutil.which("docker") is None:
            raise SandboxUnavailable(
                "the docker CLI is not installed. Install Docker Desktop (or OrbStack/Colima), "
                "or choose another sandbox with --sandbox seatbelt"
            )
        if (await _docker("info", "--format", "{{.ServerVersion}}"))[0] != 0:
            raise SandboxUnavailable(
                "Docker isn't running. Start Docker Desktop, or choose another sandbox with "
                "--sandbox seatbelt"
            )
        if (await _docker("image", "inspect", self.image))[0] != 0:
            raise SandboxUnavailable(
                f"the sandbox image {self.image} is missing; build it with "
                "`loom sandbox setup --docker`"
            )
        await self._remove_stale()
        self._session = session
        (session.sandbox_dir / "pids").mkdir(parents=True, exist_ok=True)

    async def stop(self) -> None:
        if self.container is not None:
            await _docker("rm", "-f", self.container)
        self.container = None
        self._writable = None
        self._session = None

    def describe(self) -> str:
        return "docker"

    async def _remove_stale(self) -> None:
        """Containers left by loom processes that no longer exist (crashes, kill -9)."""
        _, out, _ = await _docker(
            "ps", "-a", "--filter", f"label={LABEL}=1",
            "--format", '{{.Names}} {{.Label "loom.host_pid"}}',
        )  # fmt: skip
        for line in out.splitlines():
            name, _, pid = line.partition(" ")
            if pid.isdigit() and not _alive(int(pid)):
                await _docker("rm", "-f", name)

    async def _ensure_container(self, writable: bool) -> str:
        """Create the session's container, or recreate it when the workspace's writability
        changes (read-only mode); recreating restarts the kernel."""
        async with self._lock:
            if self.container is not None and self._writable == writable:
                return self.container
            if self._session is None:
                raise RuntimeError("DockerBackend.start() was not called")
            if self.container is not None:
                await _docker("rm", "-f", self.container)
            self.container = None
            name = _container_name(self._session)
            await _docker("rm", "-f", name)
            code, _, err = await _docker(*self.run_args(name, writable))
            if code != 0:
                raise SandboxUnavailable(f"could not start the sandbox container: {err.strip()}")
            self.container, self._writable = name, writable
            return name

    def run_args(self, name: str, writable: bool) -> list[str]:
        assert self._session is not None
        s = self._session
        ws, sandbox, out = (str(p.resolve()) for p in (s.workspace, s.sandbox_dir, s.out_dir))
        docker = self.cfg.docker
        return [
            "run", "-d", "--name", name, "--init",
            "--network", "none",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--pids-limit", "512", "--memory", docker.memory, "--cpus", f"{docker.cpus:g}",
            "--read-only", "--tmpfs", "/tmp:rw,size=1g",
            "--user", f"{os.getuid()}:{os.getgid()}",
            # Same absolute paths inside and out, so tool output paths work with read/edit.
            "-v", f"{ws}:{ws}:{'rw' if writable else 'ro'}",
            "-v", f"{sandbox}:{sandbox}:rw",
            "-v", f"{out}:{out}:ro",
            "-e", f"HOME={sandbox}/home",
            "-w", ws,
            "--label", f"{LABEL}=1",
            "--label", f"loom.host_pid={os.getpid()}",
            "--label", f"loom.session={s.id}",
            self.image, "sleep", "infinity",
        ]  # fmt: skip

    # --- processes ---------------------------------------------------------------------

    async def spawn(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
        stdin: bool = False,
        purpose: Purpose,
        workspace_writable: bool = True,
    ) -> DockerProcess:
        container = await self._ensure_container(workspace_writable)
        assert self._session is not None
        pidfile = self._session.sandbox_dir / "pids" / uuid.uuid4().hex
        # The image's PATH, not the host's; C.UTF-8 because the host's locale (e.g.
        # en_US.UTF-8) isn't installed in the image. The rest is the allowlisted env.
        container_env = {k: v for k, v in env.items() if k != "PATH"} | {"LANG": "C.UTF-8"}
        env_args = [arg for k, v in container_env.items() for arg in ("-e", f"{k}={v}")]
        command = [
            "docker", "exec", *(["-i"] if stdin else []), "-w", str(cwd), *env_args,
            container, "sh", "-c", EXEC_WRAPPER, str(pidfile), *argv,
        ]  # fmt: skip
        # The docker CLI itself runs on the host with the host environment (contexts, etc.).
        local = await spawn_local(command, cwd=cwd, env=os.environ, stdin=stdin)
        return DockerProcess(local, self, pidfile)

    def exec_quiet(self, script: str) -> None:
        """Run a short shell command in the container (signals). Blocking but brief."""
        if self.container is None:
            return
        with contextlib.suppress(OSError, subprocess.TimeoutExpired):
            subprocess.run(
                ["docker", "exec", self.container, "sh", "-c", script],
                capture_output=True,
                timeout=10,
                check=False,
            )


async def _docker(*args: str) -> tuple[int, str, str]:
    try:
        proc = await asyncio.create_subprocess_exec(
            "docker", *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
    except OSError as e:
        return 127, "", str(e)
    try:
        out, err = await asyncio.wait_for(proc.communicate(), DOCKER_TIMEOUT_S)
    except TimeoutError:
        proc.kill()
        return 124, "", f"docker {args[0]} timed out"
    return proc.returncode or 0, out.decode(errors="replace"), err.decode(errors="replace")


def _container_name(session: SessionInfo) -> str:
    raw = f"loom-{session.session_dir.parent.name}-{session.id}"
    return re.sub(r"[^a-zA-Z0-9_.-]", "-", raw)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True
