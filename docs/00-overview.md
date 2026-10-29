# 00 — Overview

## What loom is

A minimal terminal agent harness (in the spirit of Pi and Claude Code) that runs inline in the user's terminal on macOS. Its distinguishing feature is **code mode done properly**:

- The model can run Python in a **persistent, sandboxed kernel**. Variables survive between calls, so a 50,000-row Splunk result can be queried, filtered and joined across many turns without ever entering the context window.
- External systems (Splunk, GitHub, Jira, Kubernetes, databases…) are exposed as **Python functions inside the kernel**. Calls to those functions are proxied to a **host-side tool broker**, which holds credentials, enforces approvals, and logs everything.
- The sandbox itself has **no network and no secrets** by default.

## Goals

1. Small core, short system prompt, readable code (target: core + kernel + broker under ~5k lines).
2. Code mode as the default way to do multi-step or data-heavy work.
3. Safe by default: sandboxed execution, read/write approval policy, secrets isolation.
4. Multi-provider: Anthropic and any OpenAI-compatible API (OpenAI, OpenRouter, Ollama, vLLM, LM Studio).
5. Inline TUI that stays in normal terminal scrollback.
6. Full traceability: every program the agent ran is logged and inspectable.
7. Extensible: tool packages, skills, extensions, and (later) an MCP adapter.

## Non-goals (v1)

- Windows/Linux support (design keeps it possible; only macOS is implemented and tested).
- Full-screen UI (Textual) — later, on the same event bus.
- IDE integrations, web UI, multi-user/remote operation.
- Native MCP in the core (an adapter extension comes later).
- Persisting kernel memory across process restarts (possible later via snapshots).
- Steering (typing a message while a turn runs). Conflicts with the inline TUI's terminal-ownership rule; revisit with the Textual UI.

## Design principles

- **The model sees summaries; the kernel holds data.** Tools and prompts push the model to aggregate in code and print small results.
- **One choke point for side effects.** Everything that touches the outside world goes through the broker, so approvals, secrets, redaction and logging live in one place.
- **Core and UI are decoupled** via typed events.
- **Boring, explicit protocols** (JSON lines over stdio) over clever ones.
- **Defaults are safe; power users can loosen them** (`--yolo`, network allowlists).

## Decisions (confirmed with owner)

| Topic | Decision |
|---|---|
| Language | Python 3.12+ |
| Providers | Multi-provider: Anthropic, OpenAI-compatible (any `base_url`, incl. local servers) and OpenAI Responses adapters; per-model capability flags; extensions can add providers |
| UI | Inline scrolling: `prompt_toolkit` for input, `rich` for output. Textual later. |
| OS | macOS only for v1 |
| Sandbox | Pluggable backends: `none` (dev only), `seatbelt` (default), `docker` (strict) |
| Scope | Full design documented; built in milestones (`10-roadmap.md`) |
| Name | `loom` (package `loom`, CLI `loom`, data dir `~/.loom`) |
| Default model | Anthropic `claude-sonnet-5-5` (built-in profile, alias `sonnet`) |
| Built-in tool approvals | Sandbox-aware: workspace mutations (`write`/`edit`/`bash`/`execute`) are auto-allowed in `ask` mode when sandboxed (undo covers them); `bash`/`execute` ask when the backend is `none`. Table in `03-tools.md`. |
| Steering | Not in v1 (see Non-goals) |
| First tool packages | `loom-tools-demo` and `loom-tools-splunk` only; others (GitHub, Jira, kubectl…) decided later, once the broker API has seen real use (asked at M4) |
| Repo | This directory (`loom/`), git-initialised; docs live in `docs/` |

## Open questions (Claude Code: ask the owner when you reach the relevant milestone)

1. Default sandbox (M5): `seatbelt` (fast, no dependencies) vs `docker` (stronger isolation). Docs assume `seatbelt`.
2. Whether to read Claude Code skills from `~/.claude/skills` for compatibility (default: off, opt-in) (M8).
3. Default model for non-Anthropic providers (M7).

Resolved: product name (`loom`), default Anthropic model (`claude-sonnet-5-5`); see Decisions.
