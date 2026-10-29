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

Until M5, `create_backend` only knows `none`; asking for `seatbelt`/`docker` is a startup error (exit 2).

Every backend starts processes in a **new session / process group** (`start_new_session=True`), so terminal Ctrl+C is delivered only to loom, and `kill()` can take down the whole group.

**Sandbox dirs.** Each session has `<session_dir>/sandbox/` containing `runner.py` (copied at kernel start), `home/` (`HOME`) and `tmp/` (`TMPDIR`). This is the only part of the session dir a sandboxed process may write. `<session_dir>/out/` (spill files) is readable but not writable; `session.jsonl` is neither (invariant 8).

**Kernel Python per backend:** `none` → the host interpreter running loom (`sys.executable`); `seatbelt` → `~/.loom/sandbox-venv/bin/python`; `docker` → the image's `python`.

## Backends

| Backend | Isolation | Network | Startup | Use for |
|---|---|---|---|---|
| `none` | none | host | instant | Development of loom itself only. Prints a red warning in the footer. |
| `seatbelt` (default) | macOS `sandbox-exec` profile | denied | ~instant | Daily use on your own repos |
| `docker` | container (Docker Desktop, OrbStack, or Colima) | `none` by default | 1–2 s per session | Untrusted data or code, stronger isolation |

### `seatbelt`

Runs processes with `sandbox-exec -f <profile> -D WORKSPACE=… -D SESSION_DIR=… -D HOME_DIR=… …`. `sandbox-exec` is deprecated by Apple but still shipped and used by other coding agents; isolate it behind this backend so it can be replaced.

Profile template (`seatbelt.sb.tmpl`) policy:

- `(deny default)`, then allow process exec/fork, sysctl reads, mach lookups needed by Python and common CLI tools.
- **Reads:** allowed broadly (system libraries, toolchains), **except** a denylist: `~/.ssh`, `~/.aws`, `~/.config/gcloud`, `~/.kube`, `~/.docker`, `~/.netrc`, `~/Library/Keychains`, browser profile dirs, `~/.loom` (other sessions, config, history) with carve-outs for `~/.loom/sandbox-venv`, the current `<session_dir>/sandbox/` and `<session_dir>/out/`, and any `[sandbox] deny_read` paths.
- **Writes:** only the workspace (omitted when `workspace_writable=False`) and `<session_dir>/sandbox/`.
- **Network:** denied entirely (`(deny network*)`). The kernel needs no network: tools go through the broker over stdio.
- Optional (M9+): `network = "allowlist"` mode, implemented by allowing outbound only to a local filtering proxy on 127.0.0.1 and setting `HTTPS_PROXY` for sandboxed processes.

Kernel Python for seatbelt: a dedicated venv at `~/.loom/sandbox-venv`, created by `loom sandbox setup` (uses `uv venv` and installs `[sandbox] python_packages`, default `pandas numpy pyarrow`). The runner file is copied into the session dir at kernel start.

### `docker`

One container per session, created at `start()`:

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

- `spawn()` uses `docker exec -i` via the CLI (`asyncio.create_subprocess_exec`); no Docker Python SDK.
- The kernel runner is spawned with `docker exec -i … python <session_dir>/sandbox/runner.py`. Interrupt = `docker exec <ctr> sh -c 'kill -INT <pid>'` (the runner reports its in-container pid in `ready`; slim images have no standalone `kill` binary, so use the shell builtin).
- `workspace_writable` is fixed per container; switching into/out of read-only mode recreates the container (kernel restart).
- Image built from `sandbox/Dockerfile` by `loom sandbox setup --docker`: `python:3.12-slim` + `[sandbox] python_packages` + common CLIs (`git`, `ripgrep`, `jq`, `curl`).
- Mounting the workspace at the **same absolute path** means paths in tool output match host paths, so `read`/`edit` work on them directly.
- Container removed in `stop()` and on startup cleanup of stale `loom-*` containers.

## Threat model (what this does and does not protect against)

Protects against: model-written code exfiltrating data over the network, reading credentials from disk, writing outside the workspace, runaway processes (Docker limits; seatbelt relies on timeouts).

Does not protect against: a malicious **tool package** (runs on host, trusted like any dependency), the model misusing an approved write tool (approvals and audit logs cover this), or sandboxed code reading secrets that live **inside the workspace** (e.g. `.env`) and printing them into the model context.

Prompt injection note: data fetched by tools (logs, tickets) can contain instructions. Because the sandbox has no network and all tool calls are policy-checked, injected code can at worst misuse tools the policy already allows. Keep write tools on `ask`.
