# 10 — Roadmap (build in this order)

Each milestone lists what to build, the docs to re-read, and acceptance criteria. Do not start a milestone until the previous one passes. Milestones M0–M4 run with the `none` sandbox backend: **dev mode only, use on trusted tasks.**

```mermaid
flowchart LR
    M0[M0 Skeleton] --> M1[M1 Core loop] --> M2[M2 Inline TUI] --> M3[M3 Kernel + bash]
    M3 --> M4[M4 Broker + tools] --> M5[M5 Sandboxes] --> M6[M6 Sessions + compaction]
    M6 --> M7[M7 OpenAI-compat] --> M8[M8 Skills + extensions] --> M9[M9 Tracing]
    M9 --> M10[M10 Sub-agents] --> M11[M11 MCP adapter]
```

## M0 — Skeleton
Docs: 01, 09.
- `uv` project (pin Python 3.12 via `.python-version`), `src/loom`, entry point, ruff/pyright/pytest config, `.gitignore`, `docs/CHANGELOG.md`. The repo is already git-initialised.
- Config loading with pydantic models and precedence rules, including the project-config restrictions in `09-config.md` (no trust mechanism yet: restricted keys are always ignored from project config until M8).
- `loom --version`, `loom` prints "not implemented" cleanly.

Acceptance: `uv run loom --version` works; config tests cover precedence, tighten-only keys (a project `mode = "yolo"` is ignored; `mode = "read-only"` applies) and restricted keys (a project `[models.sonnet] base_url` is ignored with a notice); CI commands from CLAUDE.md are clean.

## M1 — Core loop
Docs: 02, 03 (read/write/edit, budget), 07 (log format).
- Canonical messages, event bus, `AgentLoop`, `FakeProvider`, `anthropic` adapter (streaming, prompt caching, thinking blocks preserved).
- Minimal `openai_compat` adapter (streaming text + tool-call accumulation, tool messages, `base_url`, `extra_body`, `timeout_s`), pulled forward from M7 so day-to-day manual testing runs free against a local Ollama model. Profiles/capability shaping, `openai_responses`, the text tool-call fallback, headers and TLS options stay in M7.
- Minimal `SecretStore` (env `LOOM_SECRET_<NAME>`, then the provider's env var / SDK credential chain); Keychain arrives in M4.
- Minimal `ModelProfile` with the built-in `sonnet` profile (`claude-sonnet-5-5`; context window and pricing taken from Anthropic's docs). M7 generalises profiles.
- Tools: `read`, `write`, `edit` with path policy and read-before-edit; output budget with spill files. Until M3 adds approval wiring, `read` outside the allowed roots or of sensitive paths returns an error.
- Minimal append-only session JSONL writer (every message, model request, tool call).
- Temporary plain-print UI (`loom.tui.basic`) to exercise the loop.

Acceptance: scripted FakeProvider test performs read → edit → final answer; cancellation tests (mid-stream and mid-tool-batch) leave valid tool-use/result pairing as specified in `02-…`; budget test truncates a 200k-char result and spills it; adapter tests replay recorded SSE streams (Anthropic and OpenAI-compatible) through mocked HTTP; a manual run against a local Ollama model edits a file; one short manual run against Anthropic (`sonnet`) edits a file.

## M2 — Inline TUI
Docs: 06.
- prompt_toolkit input (history, multiline, `/` and `@` completion), Rich renderer driven only by bus events, status toolbar, Ctrl+C semantics, approval prompt component (wired to a test event), slash command framework with `/help /model /cost /clear /exit`.

Acceptance: no core module imports `loom.tui` (add a test that checks imports); streaming Markdown renders without flicker or interleaving with the prompt; Ctrl+C cancels a running FakeProvider turn and returns to the prompt.

## M3 — Kernel, bash, `SandboxBackend` (`none` only)
Docs: 03 (bash, execute), 04 (kernel sections), 05 (interface).
- `SandboxBackend` interface + `none` backend; env allowlist.
- `bash` tool with streaming output, timeout, process-group kill.
- `runner.py` (stdlib only): fd redirection, persistent namespace, last-expression repr with reprlib limits, top-level await, `_out`, inventory, SIGINT handling.
- `KernelClient`: lifecycle, JSON-lines protocol, interrupt, restart notice.
- `execute` tool; `/kernel`, `/kernel reset`.
- Built-in tool approval policy (`03-tools.md`) using the M2 approval prompt: with backend `none` in `ask` mode, `bash` and `execute` ask; `read` outside roots asks. Approval decisions logged.

Acceptance: variables persist across two `execute` calls; a subprocess writing to fd 1 does not corrupt the protocol; `while True: pass` is interrupted by Ctrl+C and the kernel stays alive; sandboxed children run in their own process group; a timeout kills and later restarts the kernel with the restart prefix; with backend `none` and mode `ask`, `bash` emits `ApprovalRequested` and a denial returns an error result; `runner.py` imports only stdlib (test by AST scan).

## M4 — Broker, tool packages, approvals, secrets
Docs: 04 (broker sections), 08 (not needed yet), 09 (`[approvals]`, `[packages.*]`).
- `ToolPackage`, `@package.tool(effect=…)`, entry-point discovery, catalog → stub generation in the runner (signatures + docstrings).
- `tool_call`/`tool_result` round trip; arg validation; timeouts; redaction; broker logging.
- Approval modes + overrides + TUI prompt; timeout clock pause.
- Keychain secrets + `loom secrets` CLI.
- `search_tools` tool and `/tools`.
- Packages: `loom-tools-demo` (synthetic logs and tickets) and `loom-tools-splunk` (`search`, `list_indexes`, `create_saved_search`) using the Splunk Python SDK.

Acceptance: in the kernel, `rows = demo.search_logs(...)` returns 10k rows while the model-visible result stays under budget; a `write` tool triggers an approval and a denial surfaces as `PermissionDenied` in the kernel; a secret value returned by a package is redacted; secrets never appear in the session log (test greps the log).

## M5 — Sandboxes
Docs: 05.
- `seatbelt` backend with the profile template; `loom sandbox setup` creates `~/.loom/sandbox-venv`.
- `docker` backend (container per session, same-path mounts, limits, `--network none`, cleanup of stale containers, `loom sandbox setup --docker`).
- Default backend → `seatbelt`; footer shows the backend; red warning on `none`.

Acceptance (`@macos` / `@docker` tests): from `execute` and `bash`, network access fails; reading `~/.ssh` fails; writing outside the workspace fails; writing inside the workspace works (and fails in `read-only` mode); writing `session.jsonl` fails; reading other sessions under `~/.loom/sessions` fails; tool calls through the broker still work; the kernel inventory survives across calls in both backends.

## M6 — Sessions, compaction, checkpoints
Docs: 07.
- Resume (`-c`, `--resume`), `/sessions`, `/resume`.
- Auto + manual compaction with kernel inventory injection; never splits tool pairs.
- Shadow-git checkpoints and `/undo`.
- (Steering was removed from v1; see `00-overview.md`.)

Acceptance: a session resumes with identical history; compaction triggers at the threshold in a FakeProvider test and the next request contains the summary + inventory; `/undo` restores an edited file and removes a created one.

## M7 — More providers, model profiles, capabilities
Docs: 02 (adapters, model profiles, text tool-call fallback, headers), 09 (model entry reference).
- `ModelProfile` / `ModelCapabilities`, built-in profiles (`providers/profiles.py`), capability-driven request shaping (images, thinking, cache hints, parallel calls, streaming, `system_role`).
- `openai_compat` adapter: complete it (capability shaping, `system_role`, images in tool results, no thinking replay); the streaming core landed in M1.
- `openai_responses` adapter (function_call items, encrypted reasoning items replayed statelessly).
- Text tool-call fallback for `tools = false` models.
- Custom `headers` (with `secret:` values), `extra_body` (with reserved-key validation), `timeout_s`, `verify_tls`, `ca_bundle`.
- `/model` switching mid-session (thinking blocks dropped when switching provider), `loom models list`, `loom models test <alias>`.

Acceptance:
- Golden JSON tests: the same canonical conversations (text, tool use, tool result, image, thinking) convert correctly for `anthropic`, `openai_compat` and `openai_responses`.
- With `images = false`, no image content appears in the request body; with `thinking = false`, no reasoning config is sent.
- Text fallback: a FakeProvider emitting `<tool_call>{…}</tool_call>` text results in a real tool execution; malformed JSON produces a corrective error result.
- `extra_body` containing `messages` is rejected at config load; `secret:` header values never appear in the session log.
- Switching from an Anthropic model to an OpenAI-compatible one mid-session produces a valid request (no thinking blocks, tool pairs intact).
- Manual: a local Ollama model completes an `execute`-based task; `loom models test` reports its capabilities.

## M8 — Context files, skills, extensions
Docs: 08.
- AGENTS.md/CLAUDE.md loading, skills discovery and listing, `/skills`, `/skill`.
- Extension loader, trust prompt for project extensions (`~/.loom/trust.json`, also unlocks restricted project-config keys), hooks, commands, `Block`.
- Ship `splunk-triage` example skill.

Acceptance: a `before_broker_call` hook blocks a call and the model receives the reason; skills appear in the system prompt as one line each; an extension raising an exception produces a notice, not a crash.

## M9 — Tracing and network allowlist
Docs: 07, 05.
- `loom trace`, `/trace`; optional OTel exporter with OpenInference attributes.
- `network = "allowlist"` mode via a local filtering proxy (both backends).

Acceptance: trace tree matches the session log for a scripted run; spans export to a local collector; with an allowlist of one host, only that host is reachable from `bash`.

## M10 — Sub-agents
Docs: 08.
- `agents.run` / `agents.map` broker tools, child loops with own kernels, approval bubbling, depth and concurrency limits, transcripts.

Acceptance: `agents.map` over 4 FakeProvider tasks runs concurrently (bounded) and returns 4 strings; a sub-agent's write approval appears in the TUI labelled with its id; a sub-agent cannot spawn a sub-agent; a sub-agent call longer than `call_timeout_s` and the `execute` timeout still completes.

## M11 — MCP adapter and polish
Docs: 08, 09.
- `loom-mcp` extension (stdio + streamable HTTP clients), annotation → effect mapping.
- `loom -p` one-shot mode polish; first-run tutorial using the demo package.

Acceptance: a stdio MCP server's tools appear in `search_tools` and are callable from the kernel; `loom -p` returns a final answer with exit code 0 and denies approvals by default.
