# Changelog

## M4 — Broker, tool packages, approvals, secrets

- `loom.broker.registry`: `ToolPackage`, `@package.tool(effect=…)`, `@package.client(name)`,
  `Secret`; signature checks (ctx first, no `*args`/`**kwargs`, JSON defaults, reserved
  names); pydantic argument models; `loom.tools` entry-point discovery with load notices.
- Runner: stubs with real signatures/docstrings per package module (`demo.search_logs`,
  `tools.demo`, `import demo`), blocking `tool_call`/`tool_result` round trip, kernel-side
  `ToolError` hierarchy (`PermissionDenied`, …), bytes/datetime encoding both ways, stub
  frames hidden from tracebacks, dict reprs keep insertion order.
- `ToolBroker`: resolve (nearest-name hints), validate, approve, run with
  `call_timeout_s`, serialize, redact, `broker_call` log entries. `KernelClient` dispatches
  calls as tasks, pauses the execute clock during approval prompts, cancels abandoned calls.
- Approvals: broker table (read/write/destructive × mode), overrides and "allow for session"
  shared with built-in tools; `/mode`.
- Secrets: Keychain via `keyring` (then `LOOM_SECRET_*`), `loom secrets set|list|delete`;
  `Redactor` fed by every resolved secret plus token patterns, applied to tool results and
  every session-log record.
- `search_tools` tool, `/tools`, `loom tools list`.
- Packages (uv workspace members, installed by `uv sync --all-extras`): `loom-tools-demo`
  (48k synthetic log rows with a 02:00 checkout incident; tickets with write/destructive
  tools) and `loom-tools-splunk` (`search`, `list_indexes`, `create_saved_search`;
  `allowed_indexes` enforcement; tested against a fake service).
- Fix found while testing: runners outlived a killed host when blocked in a stub; the runner
  now exits on protocol EOF.
- Config: `api_key_secret` and `secret:` header values must be secret names; a pasted key
  is rejected at startup with a message that never echoes the value (it previously failed
  silently and fell back to the environment).
- Sign-off: one Sonnet run with `ANTHROPIC_API_KEY` unset (key from the Keychain), execute +
  demo write approval; the session log contains no trace of the key.
- Decision recorded: ship only demo + Splunk for now (`docs/00-overview.md`).

## M3 — Kernel, bash, `SandboxBackend` (`none` only)

- `loom.sandbox`: `SandboxBackend` / `SandboxProcess` protocols, `none` backend (own process
  group per process, group kill), allowlisted environment with sandbox `HOME`/`TMPDIR`.
  `seatbelt`/`docker` are a startup error until M5.
- `loom/kernel/runner.py` (stdlib only, checked by an AST scan; also tested on Python 3.11):
  private protocol fds, fd 0 → `/dev/null`, fds 1/2 captured by threads with a sync marker so
  output always precedes the result, persistent `__loom__` module namespace, last-expression
  repr via `reprlib`, top-level `await`, `_out[n]` for over-budget output, inventory answered
  while busy, SIGINT → `KeyboardInterrupt` only while executing, `sys.exit()` survivable.
- `KernelClient`: lazy start, `KernelStatus` events, interrupt (SIGINT, 3 s grace, then kill),
  timeout kill, crash notice, one-time restart prefix, `reset`, `inventory`, session log
  entries `kernel_start/execute/interrupt/kill/exit`.
- Tools `bash` (streaming, timeout, group kill, also after exit so background children can't
  hang it) and `execute` (streaming, `_out` hint in the budget marker, restart prefix).
- `loom.broker.approval`: sandbox-aware built-in policy table, overrides, per-session allows
  (per path for `read`), `approval` log entries; `read` outside the roots or of sensitive paths
  now asks instead of failing.
- TUI: `/kernel`, `/kernel reset`, kernel state in the toolbar; `/clear` shuts the old kernel
  down. `App.start()` / `App.aclose()`.
- Protocol additions (docs/04 updated): `ready` follows `init`; `execute_result` carries `vars`
  and `out_ref`; between-execution output has `id: null`.
- Fix found while testing: asyncio's `Process.wait()` waits for the pipes to close, so
  `bash -c "sleep 30 &"` hung until the child exited; `SandboxProcess.wait()` now returns when
  the process itself exits.

## M2 — Inline TUI

- `loom.tui`: prompt_toolkit input (persistent history, Enter / Alt+Enter / Ctrl+J, `/` and `@`
  completion, status toolbar, Ctrl+C clears input and twice quits, Ctrl+D quits); Rich renderer
  driven only by bus events (streaming Markdown in a screen-height-capped Live region, collapsed
  thinking, tool headers, previews and coloured diffs); single-key approval prompt; slash commands
  `/help /model /cost /clear /thinking /exit` (`/thinking` added since it's display-only).
- `loom.app.App` command API (`submit`, `switch_model`, `new_session`); `AgentLoop.switch_model`.
- `@path` mentions attach files through the read policy and output budget (`loom.core.mentions`).
- Events: `ApprovalRequested`, `Decision`, `EventBus.request_approval` / `respond`.
- `model_request` log entries record the request settings (max_tokens, thinking, cache, counts).
- Piped stdin still works (plain lines, no Live, approvals deny), for scripts and manual tests.
- Fixes found while testing in a real terminal: an empty assistant reply (seen with qwen3:8b) is no
  longer appended to history (it would break the next Anthropic request); resuming after an
  approval prompt uses a fresh Live region (a restarted one overwrote the prompt's lines); terminal
  echo is off during a turn (the echoed `^C` on cancel duplicated the first streamed line);
  modified keys from Ghostty (`CSI 27;2;27~` for Shift+Esc) are parsed instead of shown as text,
  and Shift/Ctrl/Alt+Enter insert a newline.
- Tests: import-boundary test in a fresh interpreter, renderer, approvals and prompt driven through
  prompt_toolkit pipe input, completion, commands, and a real SIGINT cancelling a FakeProvider turn.
- Docs: removed "Sonnet 5.5 thinks adaptively anyway" from the config example; without a
  `thinking` setting no thinking blocks came back in the M1 Sonnet runs.

## M1 — Core loop

- Canonical messages, event bus, `AgentLoop` (retries with backoff, step limit, usage/cost
  accounting, cancellation history repair, `max_tokens` tool calls not run), `FakeProvider`.
- `anthropic` adapter: streaming, two cache breakpoints (system, last message), adaptive thinking
  via `effort`, thinking blocks replayed verbatim, no sampling params, SDK retries off.
- `openai_compat` adapter (pulled forward from M7 for free local testing with Ollama): streaming
  text, tool-call accumulation, `reasoning` deltas, tool messages, `extra_body`, invalid-JSON args.
- Built-in `sonnet` profile (`claude-sonnet-5-5`, 1M context, pricing) and profile resolution.
- Tools `read` / `write` / `edit` with path policy, read-before-write/edit, CRLF-preserving edits;
  output budget with spill files; minimal env-only `SecretStore`.
- Session dir + append-only JSONL log; temporary plain-print UI (`loom.tui.basic`).
- Docs updated for the current Claude API: `budget_tokens` rejected on Sonnet 5.5 (use
  `effort`), no `temperature`, append-only history (preserved thinking), 1M context cost note.
- `write` also refuses to overwrite a file that changed since it was read (same rule as `edit`).

## M0 — Skeleton

- `uv` project (Python 3.12 pinned), `src/loom` layout, `loom` entry point, ruff/pyright/pytest config.
- `loom.config`: pydantic models for every section in `09-config.md` (`extra="forbid"`), layer merge
  (defaults → user → project → CLI), and the project-config restrictions (safe / tighten-only /
  restricted keys, one notice listing ignored keys). Trust is always off until M8.
- CLI: `--version`, `--model`, `--mode`, `--sandbox`; plain `loom` prints "not implemented" (exit 1);
  config errors exit 2.
- Pytest markers `macos` / `docker`, skipped automatically when unavailable.
- Decisions made while implementing (docs updated):
  - Project `[approvals.overrides]` accept only `deny` (an `ask` could loosen a read-only deny).
  - Built-in sandbox default is `none` until M5 switches it to `seatbelt`.
  - If the workspace is `$HOME`, the project config file is the user config and isn't filtered.
  - `docs/` is excluded from ruff (it formats Markdown code blocks).

## Docs review (before M0)

- Sandbox-aware approvals for built-in tools; project-config trust tiers; sandbox can't write the
  session log; cancellation history repair; tool results can carry images; `thinking`/`pricing`
  model keys; `[budget]` and `[packages.<name>]` sections; steering removed from v1.
