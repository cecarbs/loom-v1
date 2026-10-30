# CLAUDE.md — building `loom`

You are building **loom**, a minimal, terminal-based coding/ops agent harness in Python with **code mode** as a first-class feature: the model works through a sandboxed, *persistent* Python kernel whose tool functions are proxied to a host-side broker. Large data stays in the kernel; only small summaries reach the model's context.

`loom` is a placeholder name. If the owner renames it, rename the package, CLI entry point, and `~/.loom` paths together.

## Read these first (in order)

1. `docs/00-overview.md` — goals, non-goals, principles, open decisions
2. `docs/01-architecture.md` — components, diagrams, package layout
3. `docs/02-agent-loop-and-providers.md`
4. `docs/03-tools.md`
5. `docs/04-kernel-and-broker.md` — the core of code mode
6. `docs/05-sandbox.md`
7. `docs/06-tui.md`
8. `docs/07-sessions-context-observability.md`
9. `docs/08-skills-extensions-subagents.md`
10. `docs/09-config.md`
11. `docs/10-roadmap.md` — **build in this milestone order**

## How to work

- Build **one milestone at a time** from `docs/10-roadmap.md`. A milestone is done only when its acceptance criteria pass, including tests.
- Before starting a milestone, re-read the docs it references. If a doc is ambiguous or wrong, propose a fix to the doc first, then implement. Keep docs and code in sync.
- Do not jump ahead to later-milestone features, but do respect the interfaces later milestones depend on (e.g. the `SandboxBackend` interface exists from M3 even though only `none` is implemented).
- Ask the owner before changing any decision listed under "Decisions" in `docs/00-overview.md`.

## Hard rules (architecture invariants)

1. **`loom.core` never imports `loom.tui`.** The core emits `AgentEvent`s on an event bus; the TUI subscribes. A future Textual UI must be able to reuse the core unchanged.
2. **`loom/kernel/runner.py` is stdlib-only** and must run on any Python ≥ 3.11 inside the sandbox. It never imports `loom`.
3. **Tool package code runs on the host (broker), never in the sandbox.** The sandbox gets generated stubs only.
4. **Secrets never enter the sandbox or the model context.** Secrets are resolved only on the host by `SecretStore`; its only consumers are the broker (tool packages), provider adapters (API keys, `secret:` headers) and host-side extensions via `api.secret` (loom-mcp's server credentials).
5. **Every tool result passes through the output budget** (`loom/tools/budget.py`) before reaching the model.
6. **Every model request, tool call, kernel execution, and approval decision is written to the session log.**
7. Tool-use / tool-result pairing in message history must always be valid, including after cancellation (insert synthetic error results).
8. **Sandboxed processes can never write the session log** (or anything else in the session dir outside `sandbox/`). The audit trail must not be tamperable by model-written code.
9. **Untrusted project config cannot route credentials or code.** Keys that choose endpoints, secrets, or what runs (see `docs/09-config.md`) are ignored from `<workspace>/.loom/config.toml` unless the workspace is trusted.

## Tech and conventions

- Python **3.12+**, `uv` for env and packaging, `src/` layout, entry point `loom = "loom.cli:main"`.
- Async: `asyncio` throughout the core. Blocking work (subprocess I/O, SDK calls without async) goes through `asyncio.to_thread` or async APIs.
- Libraries: `anthropic`, `openai`, `prompt_toolkit`, `rich`, `pydantic` (v2, for config and tool schemas), `keyring`, `opentelemetry-*` (optional extra), `pyyaml`; `mcp` and `httpx2` in `packages/loom-mcp` only. Docker is driven through the `docker` CLI (no Python SDK).
- Not used in v1: Textual (reserved for a later full-screen UI on the same event bus).
- Typing: full type hints; `pyright` in basic mode. Lint/format: `ruff`.
- Tests: `pytest`, `pytest-asyncio`. No real network or real model calls in unit tests — use `FakeProvider` (scripted responses). Tests needing macOS `sandbox-exec` or Docker are marked `@pytest.mark.macos` / `@pytest.mark.docker` and skipped when unavailable.

## Commands

Prerequisite: `uv` (`brew install uv`). The host may only have an older system Python; `uv` installs and pins 3.12 (`.python-version`).

```bash
uv sync --all-extras          # install
uv run pytest -q              # tests
uv run ruff check . && uv run ruff format --check .
uv run pyright
uv run loom                   # run the harness in the current directory
```

## Definition of done (every PR/milestone)

- Acceptance criteria from the roadmap pass.
- New behavior has tests; `pytest`, `ruff`, `pyright` are clean.
- Docs updated if behavior or interfaces changed.
- Short note appended to `docs/CHANGELOG.md` (create it in M0).
