# 03 — Built-in tools

loom exposes exactly six tools to the model. Everything else (Splunk, GitHub, …) is a **function inside the kernel**, discovered with `search_tools`.

| Tool | Runs | Purpose |
|---|---|---|
| `read` | host | Read a text file (or image) with line numbers |
| `write` | host | Create/overwrite a file |
| `edit` | host | Exact string replacement in a file |
| `bash` | sandbox | Run a shell command (fresh shell each call) |
| `execute` | sandbox kernel | Run Python in the persistent kernel |
| `search_tools` | host (broker) | Find kernel functions by keyword |

## Tool interface (`loom.tools.base`)

```python
class Tool(Protocol):
    name: str
    description: str
    input_model: type[pydantic.BaseModel]   # JSON schema derived from this
    async def run(self, args: BaseModel, ctx: ToolContext) -> ToolResult: ...

@dataclass
class ToolResult:
    text: str
    is_error: bool = False
    images: list[ImageBlock] = field(default_factory=list)  # e.g. read on a PNG
    meta: dict[str, Any] = field(default_factory=dict)   # diff, exit_code, duration, spill path…
```

`ToolContext` gives access to: workspace path, session paths, sandbox backend, kernel client, broker, event bus (for streaming `ToolCallOutput`), and the checkpoint manager.

Invalid arguments (pydantic validation error) return `is_error=True` with the validation message so the model can correct itself.

## Path policy (host-side file tools)

`read`, `write`, `edit` may access:
- the workspace (cwd at launch, resolved, symlinks resolved),
- the session's spill directory (read-only),
- skill directories (read-only).

`write`/`edit` outside the workspace are always denied. `read` outside these roots, and `read` of **sensitive paths** anywhere (the sandbox denylist in `05-sandbox.md`, `~/.loom`, and `.env*` files including ones inside the workspace), needs approval (see below). The sensitive list is one shared constant used by both the path policy and the seatbelt profile.

Limitation: sandboxed `bash`/`execute` can read files in the workspace, including `.env*`. The path policy only governs the host-side file tools.

## Approvals for built-in tools

Built-in tools go through the same approval machinery as broker tools (`loom.broker.approval`, `ApprovalRequested`, session-log `approval` entries). The policy is **sandbox-aware**: changes to the workspace are undoable (`/undo`) and the sandbox contains everything else, so they don't prompt; without a sandbox, shell and kernel get full host access, so they do.

| Tool | `read-only` | `ask` (default) | `auto-write` | `yolo` |
|---|---|---|---|---|
| `read`, `search_tools` (allowed roots) | allow | allow | allow | allow |
| `read` outside roots or sensitive path | ask | ask | ask | allow |
| `write`, `edit` | deny | allow | allow | allow |
| `bash`, `execute` — sandboxed (`seatbelt`, `docker`) | allow, workspace mounted read-only | allow | allow | allow |
| `bash`, `execute` — backend `none` | ask | ask | allow | allow |

`[approvals.overrides]` accepts built-in tool names too (e.g. `"bash" = "allow"`). In `loom -p`, `ask` resolves to deny. Changing into or out of `read-only` mid-session with the docker backend recreates the container (kernel restart notice).

## Specs

### `read`
```
path: str
offset: int = 1        # 1-based line
limit: int = 2000
```
Returns `cat -n` style numbered lines. Long lines cut at 2,000 chars. PNG/JPEG/GIF/WebP returned as an `ImageBlock`. Records the file's mtime+hash in the session's "read set" (used by `edit`).

### `write`
```
path: str
content: str
```
Overwriting an existing file requires that it was `read` in this session and is unchanged since (else error: "Read the file first" / "changed since you last read it"). Creates parent dirs. Emits a diff in `meta` for the TUI. After a successful `write` or `edit`, the file's new mtime+hash replaces its read-set entry, so consecutive edits don't need a re-read.

### `edit`
```
path: str
old_string: str
new_string: str
replace_all: bool = False
```
File must have been `read` and unchanged since (mtime+hash). `old_string` must match exactly once unless `replace_all`. Errors are precise: "0 matches", "3 matches; add context or set replace_all". Emits a unified diff in `meta`.

### `bash`
```
command: str
timeout_s: int = 120     # max 600
```
Runs `/bin/bash -lc <command>` through the sandbox backend with cwd = workspace. Fresh shell each call (no persisted env). Streams stdout/stderr as `ToolCallOutput` events. Result: combined output + `exit code: N`. On timeout: kill the process group, return what was captured plus "timed out after Ns".

### `execute`
```
code: str
timeout_s: int = 300     # max 1800; clock pauses while waiting on an approval
```
Sends code to the persistent kernel (see `04-kernel-and-broker.md`). Result text is: captured stdout/stderr, then the repr of the last expression if any, then an error traceback if any. If the kernel died or was restarted, the result starts with `[kernel restarted — previous variables are gone]`.

### `search_tools`
```
query: str
limit: int = 10
```
Keyword/fuzzy search over the broker's catalog (name, docstring, package). Returns for each match:
```
splunk.search(spl: str, earliest: str = "-1h", latest: str = "now", max_rows: int = 1000) -> list[dict]   [read]
    Run an SPL search and return result rows. Aggregate with stats/timechart in SPL where possible.
```
An empty query lists packages with one-line descriptions.

## Output budget (`loom.tools.budget`)

Applied to **every** tool result before it enters the history.

- Limit: `budget.max_chars` (default 12,000 chars) per result.
- Over the limit: keep the first 8,000 and last 3,000 chars, insert a marker:
  `[… 184,220 chars omitted. Full output: <session_dir>/out/0007.txt (use read with offset). In kernel: _out[4]]`
- The full text is saved to the session spill dir. Spill files are numbered per session across all tools (`0007`).
- For `execute`, the runner also keeps the full output string in the kernel as `_out[n]`, where `n` is the execute number (execute id `e4` → `_out[4]`). It does so **only** when the output exceeds `max_chars` (passed in the `init` settings), so `_out` doesn't grow with every call. The `_out[...]` hint appears only for `execute` results.
- Also in the runner: `repr` of the last expression is limited with `reprlib` (max ~2,000 chars, 50 items per container) so a huge DataFrame/list repr can't blow the budget in the first place.
