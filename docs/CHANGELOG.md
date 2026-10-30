# Changelog

## M10 — Sub-agents

- `loom.agents.subagent`: `agents.run(task, model=, max_turns=)` and
  `agents.map(tasks, concurrency=, model=, max_turns=)` in the kernel, as broker tools in the
  internal `agents` package. Each sub-agent is a headless `AgentLoop` with its own history,
  lazily started kernel, read set and private event bus. `app.py`'s loop assembly moved to
  `_make_loop`, used for both the main loop and children.
- `ToolPackage(internal=True)` (may use a reserved name; left out of sub-agents' brokers, which
  enforces the depth limit) and `@package.tool(unbounded=True)` (no `call_timeout_s`, pauses the
  `execute` clock).
- `Approvals.scoped` / `ToolBroker.scoped`: a sub-agent shares the mode, overrides, session
  grants, approval request ids, package clients and extension hooks, with its own log. Approval
  requests go to the main bus with `agent_id` set.
- Transcripts in `<session_dir>/agents/<id>.jsonl`; a `subagent` record in the parent log points
  to each one and carries its usage, which counts toward session totals and resume but not the
  parent's context size. `loom trace` shows it as a point event. The `read` tool can open
  transcripts without approval; the sandbox can't reach them.
- `agents.run` raises `ToolError` when a sub-agent ends without an answer; `agents.map` keeps the
  other answers and returns `[sub-agent <id> failed: …]` for that entry.
- Cancelling a turn stops all its sub-agents and their kernels before the turn ends;
  `App.aclose` and session switches clean up anything left.
- Fixes needed for concurrency: the kernel runner is copied atomically (a starting kernel could
  read a half-written `runner.py`); the renderer keeps the live region down until the last queued
  prompt is answered and holds notices back while a prompt is up.

## M9 — Tracing and network allowlist

- `loom.tracing.log`: the session log as a span tree (turn → model / tool calls; `execute` →
  kernel execution → broker calls; approvals, checkpoints, network, kernel and error records as
  point events). `loom trace [<id>] [--last N] [--code]` and `/trace [n]` print it.
- `loom.tracing.otel` (extra `loom[otel]`): with `[tracing] otlp_endpoint`, each turn is
  exported over OTLP/HTTP when it ends, built from the (redacted) log records, with
  OpenInference attributes (code on `kernel.execute`). Missing extra or collector → a notice.
- `SessionLog.listeners`; new log kinds `turn_end` and `network`.
- `[sandbox] network = "allowlist"`: a filtering HTTP proxy (`loom.sandbox.proxy`, stdlib-only;
  `CONNECT` and absolute `http://`) with `HTTP(S)_PROXY` set for sandboxed processes. seatbelt
  runs it in-process and allows only its localhost port; docker puts the container on a
  per-session `--internal` network with a locked-down proxy sidecar (source passed with `-c`),
  cleaned up with the container. Allowlist entries are validated (`host`, `*.domain`, `:port`).
- Footer shows `docker · net: N hosts` in allowlist mode, and the system prompt gets a `# Network`
  section naming the hosts (a manual run showed qwen blaming the server for the proxy 403).
- Docs: `/trace` now shows the tree (06 said "last n tool calls"); allowlist threat note.
- Fix: with no Anthropic key anywhere, the SDK's `TypeError` crashed the TUI; it is now an
  `auth` error naming the secret and `ANTHROPIC_API_KEY`.
- Test fix: the docker read-only test looks up the workspace mount by path (mount order varies).

## M8 — Context files, skills, extensions

- Context files: `~/.loom/AGENTS.md`, then `AGENTS.md` / `CLAUDE.md` from the git root down to
  the cwd (symlinked duplicates once), capped at `max_context_file_chars` with a notice.
- Skills (`loom.skills`): `SKILL.md` frontmatter discovery in `~/.loom/skills`,
  `.loom/skills` and (opt-in) `~/.claude/skills`; the user's own skills win name clashes; one
  `- name — description (path)` line each in the system prompt; skill dirs readable without
  approval; `/skills`, `/skill <name> [message]`.
- Extensions (`loom.extensions`): `register(api)` modules from `~/.loom/extensions` and (trusted)
  `.loom/extensions`; hooks for model requests/responses, built-in tool calls, broker calls,
  session start and turn end, with `Block(reason)`; slash commands with `ctx.submit`; tool
  packages, providers and built-in tools. Failures at load, in hooks or commands are notices.
- Workspace trust (`~/.loom/trust.json`): a fingerprint of the project config and extensions
  dir; interactive `loom` asks once when there is something to trust and asks again after any
  change. Trust now unlocks restricted project-config keys.
- `examples/skills/splunk-triage` and `examples/extensions/no_prod_writes.py`; `pyyaml` added.

## M7 — More providers, model profiles, capabilities

- `openai_responses` adapter: stateless (`store=false`), `function_call` / `function_call_output`
  items, encrypted reasoning items kept as `ThinkingBlock`s and replayed only to this adapter,
  tool-result images inline, `instructions` / developer / user system role, error events mapped.
- Text tool-call fallback (`providers/text_tools.py`) as a wrapper around any adapter when
  `tools = false`; history stays canonical. Malformed JSON is now checked before the tool name,
  so it gets the "resend valid JSON" error.
- Capability shaping: no thinking config at all when `thinking = false` (notice at start and on
  `/model`); `streaming = false` makes plain calls (OpenAI adapters; Anthropic always streams);
  `verify_tls` / `ca_bundle` wired into both SDK clients.
- Built-in aliases `opus` (`claude-opus-5-5`) and `haiku` (`claude-haiku-5-5`) with pricing from
  Anthropic's model table; `openai_responses` defaults to `images = true`.
- `extra_body` also may not set `system`, `input`, `instructions`.
- `loom models list` and `loom models test <alias>` (reply / tool call / image probes, suggested
  capability lines).
- Golden JSON request tests for all three adapters (`tests/fixtures/golden/`, regenerate with
  `LOOM_UPDATE_GOLDEN=1`).

## M6 — Sessions, compaction, checkpoints

- Resume: `loom -c`, `loom --resume <id|prefix>`, `/resume`, `/sessions`, `loom sessions`.
  History is rebuilt from `message` + `compaction` entries (identical to the live history), plus
  turns, usage, cost, checkpoints and undelivered notes; the session's last model is used unless
  `--model`. A pid lock file stops two looms from using one session. If the kernel was used, the
  next prompt says it is fresh.
- Compaction (`loom.core.compaction`): auto at `compact_at` (reported prompt tokens + estimate of
  newer messages), on `context_too_long` (then one retry), and `/compact [instructions]`. Keeps
  the last `keep_recent_turns` turns byte-identical, never splits tool pairs, falls back to a
  mid-turn cut in one long turn, appends the kernel inventory, logs the summary message.
  `Compacted` event.
- Checkpoints (`loom.session.checkpoints`): shadow git repo per project under
  `~/.loom/checkpoints`, per-session index and ref, one snapshot before the first mutating tool
  of a turn; `.gitignore` + defaults + >5 MB files excluded. `/undo [n]` restores changed files,
  removes created ones, checkpoints "before /undo" first, and tells the model via a logged
  `note`.
- New log kinds: `session_resume`, `checkpoint`, `undo`, `note`; `model_request` gains `purpose`
  and `cost_usd`.
- Not restored on resume (by design): session approval grants, the read set.

## M5 — Sandboxes

- Decision (owner): **`docker` is the default backend**. If Docker isn't running or the image
  is missing, loom exits with a fix-it message; it never falls back to a weaker sandbox.
- `docker` backend: container per session (created on first use), `--network none`,
  `--cap-drop ALL`, no-new-privileges, pids/memory/cpu limits, read-only root, same-path
  mounts (workspace `ro` in read-only mode, recreated on switch), labelled stale-container
  cleanup, in-container signals/kills through pid files, `LANG=C.UTF-8`. Image from
  `src/loom/sandbox/Dockerfile`.
- `seatbelt` backend: `sandbox-exec` with an inline profile (never a file the sandbox could
  rewrite), shared sensitive-path denylist, `~/.loom` denied with carve-outs for the venv and
  this session's `sandbox/` and `out/`, no network.
- `loom sandbox setup [--docker]`; `create_backend(SandboxConfig)`; toolbar shows backend
  `none` as a red "NO SANDBOX" badge.
- Tests (`@macos` / `@docker`, both run here): network blocked, `~/.ssh` and other sessions
  unreadable, writes only in the workspace (none in read-only mode), `session.jsonl` not
  writable, broker tools and kernel inventory across calls, Ctrl+C / timeouts / cancel
  reaching processes inside the sandbox, container recreate, stale cleanup, startup errors.
  Unit tests run on `none` via a conftest fixture since the built-in default is docker.

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
