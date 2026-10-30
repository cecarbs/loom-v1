"""The `seatbelt` backend: macOS `sandbox-exec` with loom's profile (docs/05).

`sandbox-exec` is deprecated by Apple but still shipped; everything about it stays behind
this backend so it can be replaced.
"""

from __future__ import annotations

import shutil
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from loom.sandbox.base import (
    NetworkCallback,
    Purpose,
    SandboxUnavailable,
    SessionInfo,
    proxy_env,
)
from loom.sandbox.none import LocalProcess, spawn_local
from loom.sandbox.proxy import Allowlist, Proxy
from loom.tools.paths import SENSITIVE_HOME_PATHS

SANDBOX_EXEC = "/usr/bin/sandbox-exec"
TEMPLATE = (Path(__file__).with_name("seatbelt.sb.tmpl")).read_text(encoding="utf-8")


def sbpl_string(path: Path | str) -> str:
    text = str(path)
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def render_profile(
    *,
    home: Path,
    loom_home: Path,
    session: SessionInfo,
    extra_deny: Iterable[Path] = (),
    workspace_writable: bool = True,
    proxy_port: int | None = None,
) -> str:
    """The profile text. Paths are baked in as resolved literals (seatbelt matches real
    paths, e.g. /private/var rather than /var). With `proxy_port`, outbound TCP to that
    localhost port (the allowlist proxy) is the only network allowed."""
    deny = [home / p for p in SENSITIVE_HOME_PATHS]
    deny += [loom_home, *extra_deny]
    deny_lines = "\n".join(f"  (subpath {sbpl_string(_real(p))})" for p in deny)
    ancestors = _ancestors(_real(session.session_dir), stop_above=_real(loom_home))
    ancestors |= _ancestors(_real(loom_home / "sandbox-venv"), stop_above=_real(loom_home))
    # Never render an empty rule: `(allow file-read-metadata)` alone would allow everything.
    ancestor_rule = (
        "(allow file-read-metadata\n"
        + "\n".join(f"  (literal {sbpl_string(p)})" for p in sorted(ancestors))
        + ")"
        if ancestors
        else ""
    )
    workspace = f"  (subpath {sbpl_string(_real(session.workspace))})" if workspace_writable else ""
    network = (
        f'(allow network-outbound (remote tcp "localhost:{int(proxy_port)}"))'
        if proxy_port is not None
        else ""
    )
    return TEMPLATE.format(
        deny_read=deny_lines,
        ancestors=ancestor_rule,
        workspace_write=workspace,
        network=network,
    )


def _real(path: Path) -> Path:
    return path.expanduser().resolve()


def _ancestors(path: Path, *, stop_above: Path) -> set[Path]:
    """`path`'s parents up to and including `stop_above` (only if `path` is inside it)."""
    if not path.is_relative_to(stop_above):
        return set()
    out = set()
    for parent in path.parents:
        out.add(parent)
        if parent == stop_above:
            break
    return out


class SeatbeltBackend:
    name = "seatbelt"
    on_network: NetworkCallback | None = None

    def __init__(
        self,
        *,
        loom_home: Path,
        home: Path | None = None,
        python: str | None = None,
        deny_read: Sequence[str] = (),
        allowlist: Sequence[str] | None = None,  # None: no network at all
    ) -> None:
        self.home = home or Path.home()
        self.loom_home = loom_home
        self.venv = loom_home / "sandbox-venv"
        self.python = python or str(self.venv / "bin" / "python")
        self.deny_read = [Path(p).expanduser() for p in deny_read]
        self.allowlist = None if allowlist is None else list(allowlist)
        self._session: SessionInfo | None = None
        self._proxy: Proxy | None = None
        self._proxy_port: int | None = None

    async def start(self, session: SessionInfo) -> None:
        if not Path(SANDBOX_EXEC).exists():
            raise SandboxUnavailable("seatbelt needs /usr/bin/sandbox-exec (macOS only)")
        if shutil.which(self.python) is None and not Path(self.python).exists():
            raise SandboxUnavailable(
                f"the seatbelt kernel Python is missing ({self.python}); run `loom sandbox setup`"
            )
        self._session = session
        if self.allowlist is not None and self._proxy is None:
            # In the loom process: sandboxed code can reach this port and nothing else.
            self._proxy = Proxy(Allowlist(self.allowlist), self._network_event)
            self._proxy_port = await self._proxy.start("127.0.0.1", 0)

    def _network_event(self, event: dict) -> None:
        if self.on_network is not None:
            self.on_network(event)

    def profile(self, workspace_writable: bool) -> str:
        if self._session is None:
            raise RuntimeError("SeatbeltBackend.start() was not called")
        return render_profile(
            home=self.home,
            loom_home=self.loom_home,
            session=self._session,
            extra_deny=self.deny_read,
            workspace_writable=workspace_writable,
            proxy_port=self._proxy_port,
        )

    def params(self) -> list[str]:
        assert self._session is not None
        session = self._session
        values = {
            "VENV": self.venv,
            "SANDBOX_DIR": session.sandbox_dir,
            "OUT_DIR": session.out_dir,
        }
        out = []
        for key, path in values.items():
            out += ["-D", f"{key}={_real(path)}"]
        return out

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
        # sandbox-exec execs the command, so the pid (and process group) stay the same.
        command = [SANDBOX_EXEC, "-p", self.profile(workspace_writable), *self.params(), *argv]
        if self._proxy_port is not None:
            env = {**env, **proxy_env(f"http://127.0.0.1:{self._proxy_port}")}
        return await spawn_local(command, cwd=cwd, env=env, stdin=stdin)

    async def stop(self) -> None:
        if self._proxy is not None:
            await self._proxy.close()
        self._proxy = self._proxy_port = None
        self._session = None

    def describe(self) -> str:
        if self.allowlist is not None:
            return f"seatbelt · net: {len(self.allowlist)} hosts"
        return "seatbelt"
