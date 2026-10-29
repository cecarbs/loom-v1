# 05 — Sandbox

All model-driven execution (`bash`, the kernel runner) goes through a `SandboxBackend`. Host-side file tools (`read`/`write`/`edit`) and the broker do **not** run in the sandbox; they enforce their own policies.

## Interface (`loom.sandbox.base`)

```python
class SandboxBackend(Protocol):
    name: str
    python: str                                                    # kernel interpreter (below)
    async def start(self, session: SessionInfo) -> None: ...      # e.g. create container
    async def spawn(self, argv: list[str], *, cwd: Path, env: dict[str, str],
                    stdin: bool = False, purpose: Literal["bash", "kernel"],
                    workspace_writable: bool = True) -> SandboxProcess: ...   # False in read-only mode
    async def stop(self) -> None: ...
    def describe(self) -> str: ...                                 # shown in TUI footer

class SandboxProcess(Protocol):
    pid: int
    stdin: asyncio.StreamWriter | None
    stdout: asyncio.StreamReader
    stderr: asyncio.StreamReader
    returncode: int | None
    async def wait(self) -> int: ...       # the process itself, NOT children holding its pipes
    def send_signal(self, sig: int) -> None: ...
    def kill(self) -> None: ...                                    # whole process group
```

`env` passed to sandboxed processes is built from an **allowlist** (`PATH`, `HOME` (sandbox home), `LANG`, `TERM`, `TMPDIR`, plus `[sandbox] env_passthrough`) by `loom.sandbox.base.sandbox_env`. The host environment is never inherited wholesale.

Note on `wait()`: asyncio's `Process.wait()` also waits for the pipes to close, which never happens while a background child (`sleep 100 &`) still holds them; backends must return when the process itself exits.

`create_backend(cfg)` picks the backend; availability (daemon, image, venv) is checked in `start()`, and a `SandboxUnavailable` there ends `loom` with exit code 2 and a message saying how to fix it.

Every backend starts processes in a **new session / process group** (`start_new_session=True`), so terminal Ctrl+C is delivered only to loom, and `kill()` can take down the whole group.

**Sandbox dirs.** Each session has `<session_dir>/sandbox/` containing `runner.py` (copied at kernel start), `home/` (`HOME`) and `tmp/` (`TMPDIR`). This is the only part of the session dir a sandboxed process may write. `<session_dir>/out/` (spill files) is readable but not writable; `session.jsonl` is neither (invariant 8).

**Kernel Python per backend:** `none` → the host interpreter running loom (`sys.executable`); `seatbelt` → `~/.loom/sandbox-venv/bin/python`; `docker` → the image's `python`.

## Backends

| Backend | Isolation | Network | Startup | Use for |
|---|---|---|---|---|
| `none` | none | host | instant | Development of loom itself only. Prints a red warning in the footer. |
| `seatbelt` | macOS `sandbox-exec` profile | denied | ~instant | When Docker isn't available; fast daily use on your own repos |
| `docker` (default) | container (Docker Desktop, OrbStack, or Colima) | `none` by default | 1–2 s per session | Everything by default: strongest isolation, also for untrusted data or code |

With the default `docker` backend, loom checks at startup that the daemon answers and the image exists. If not, it exits with a message saying how to fix it (`start Docker Desktop`, `loom sandbox setup --docker`) or to choose `--sandbox seatbelt` explicitly. It never falls back to a weaker backend on its own.

### `seatbelt`

Runs processes with `sandbox-exec -p <profile> -D VENV=… -D SANDBOX_DIR=… -D OUT_DIR=… <argv>`. The profile is passed **inline**, never as a file: a profile file inside the session's writable sandbox dir could be rewritten by sandboxed code to loosen the next spawn. Paths are resolved before rendering (seatbelt matches real paths, `/private/var` not `/var`). `sandbox-exec` execs the command, so pid and process group are those of the command itself. `sandbox-exec` is deprecated by Apple but still shipped and used by other coding agents; isolate it behind this backend so it can be replaced.

Profile template (`seatbelt.sb.tmpl`) policy:

- `(deny default)`, then allow process exec/fork, sysctl reads, mach lookups needed by Python and common CLI tools.
- **Reads:** allowed broadly (system libraries, toolchains), **except** a denylist: `~/.ssh`, `~/.aws`, `~/.config/gcloud`, `~/.kube`, `~/.docker`, `~/.netrc`, `~/Library/Keychains`, browser profile dirs, `~/.loom` (other sessions, config, history) with carve-outs for `~/.loom/sandbox-venv`, the current `<session_dir>/sandbox/` and `<session_dir>/out/`, and any `[sandbox] deny_read` paths.
- Rule order matters: in SBPL the last matching rule wins, so the profile is broad allows → denylist → carve-outs. Directories on the way to the carve-outs (`~/.loom`, `~/.loom/sessions/<project>`, …) get `file-read-metadata` only (stat, not list). That rule is never rendered empty: `(allow file-read-metadata)` with no filter would allow everything.
- setuid binaries (e.g. `/bin/ps`) can't be exec'd inside the sandbox.
- **Writes:** only the workspace (omitted when `workspace_writable=False`) and `<session_dir>/sandbox/`.
- **Network:** denied entirely (`(deny network*)`). The kernel needs no network: tools go through the broker over stdio.
- Optional (M9+): `network = "allowlist"` mode, implemented by allowing outbound only to a local filtering proxy on 127.0.0.1 and setting `HTTPS_PROXY` for sandboxed processes.

Kernel Python for seatbelt: a dedicated venv at `~/.loom/sandbox-venv`, created by `loom sandbox setup` (`uv venv --python 3.12`, then `uv pip install` of `[sandbox] python_packages`, default `pandas numpy pyarrow`; falls back to `python -m venv` + pip without uv). If it's missing, `start()` says to run `loom sandbox setup`. The runner file is copied into the session dir at kernel start.

### `docker`

One container per session. `start()` checks the CLI, the daemon and the image, and removes stale containers; the container itself is created on the first spawn, with the workspace mount matching that spawn's `workspace_writable` (so starting in read-only mode doesn't create a container twice):

```
docker run -d --name loom-<session> --init
  --network none
  --cap-drop ALL --security-opt no-new-privileges
  --pids-limit 512 --memory 4g --cpus 2
  --read-only --tmpfs /tmp:rw,size=1g
  --user <uid>:<gid>
  -v <workspace>:<workspace>:rw          # same absolute path inside and out; :ro in read-only mode
  -v <session_dir>/sandbox:<session_dir>/sandbox:rw
  -v <session_dir>/out:<session_dir>/out:ro
  -e HOME=<session_dir>/sandbox/home
  -w <workspace>
  loom-sandbox:<version> sleep infinity
```

- Also on the container: labels `loom.sandbox=1`, `loom.host_pid=<loom's pid>`, `loom.session=<id>`. "Stale" = a `loom.sandbox` container whose `loom.host_pid` is no longer alive, so concurrent loom instances don't remove each other's containers.
- `spawn()` uses `docker exec [-i] -w <cwd> -e … <ctr> sh -c 'echo $$ > "$0"; exec "$@"' <pidfile> <argv…>` via the CLI (`asyncio.create_subprocess_exec`); no Docker Python SDK. The docker CLI process runs with the host environment (contexts, `DOCKER_HOST`); the container process gets the allowlisted env without `PATH` (the image's own) and with `LANG=C.UTF-8` (the host locale usually isn't installed in the image).
- Killing the host `docker exec` does **not** stop the process inside the container. So each exec'd process writes its in-container pid to `<session_dir>/sandbox/pids/<token>`, and `send_signal`/`kill` run `docker exec <ctr> sh -c 'kill -<sig> <pid>'` / `kill -9 -- -<pid>` (exec'd processes lead their own session and process group in the container, so the group kill takes their children too).
- The kernel runner is spawned with `docker exec -i … python -I <session_dir>/sandbox/runner.py`; interrupts reach it through the pid file as above.
- `workspace_writable` is fixed per container; switching into/out of read-only mode recreates the container (kernel restart).
- Image built from `sandbox/Dockerfile` by `loom sandbox setup --docker`: `python:3.12-slim` + `[sandbox] python_packages` + common CLIs (`git`, `ripgrep`, `jq`, `curl`, `procps`).
- Mounting the workspace at the **same absolute path** means paths in tool output match host paths, so `read`/`edit` work on them directly.
- Container removed in `stop()` and on startup cleanup of stale `loom-*` containers.

## Threat model (what this does and does not protect against)

Protects against: model-written code exfiltrating data over the network, reading credentials from disk, writing outside the workspace, runaway processes (Docker limits; seatbelt relies on timeouts).

Does not protect against: a malicious **tool package** (runs on host, trusted like any dependency), the model misusing an approved write tool (approvals and audit logs cover this), or sandboxed code reading secrets that live **inside the workspace** (e.g. `.env`) and printing them into the model context.

Prompt injection note: data fetched by tools (logs, tickets) can contain instructions. Because the sandbox has no network and all tool calls are policy-checked, injected code can at worst misuse tools the policy already allows. Keep write tools on `ask`.
