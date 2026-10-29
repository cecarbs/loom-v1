# loom — design docs

Design documentation for **loom**, a Python terminal agent harness with a sandboxed, persistent code-mode kernel.

## How to use this with Claude Code

1. This directory is the repo (`CLAUDE.md`, `docs/`, and later `pyproject.toml` and `src/` live at its root).
2. Start Claude Code in this repo and say:

   > Read CLAUDE.md and all files in docs/ in order. Then start milestone M0 from docs/10-roadmap.md. Before writing code, list any questions or doc issues you find.

3. After each milestone, review, commit, and ask it to continue with the next one.

## Contents

| File | Topic |
|---|---|
| `CLAUDE.md` | Instructions and invariants for the coding agent |
| `docs/00-overview.md` | Goals, non-goals, decisions, open questions |
| `docs/01-architecture.md` | Components, diagrams, package layout |
| `docs/02-agent-loop-and-providers.md` | Messages, provider adapters, loop, events |
| `docs/03-tools.md` | The six built-in tools and the output budget |
| `docs/04-kernel-and-broker.md` | Code mode: kernel protocol, tool packages, approvals, secrets |
| `docs/05-sandbox.md` | Sandbox backends (none, seatbelt, docker) and threat model |
| `docs/06-tui.md` | Inline prompt_toolkit + Rich UI |
| `docs/07-sessions-context-observability.md` | Session log, compaction, undo, tracing |
| `docs/08-skills-extensions-subagents.md` | Skills, extensions, sub-agents, MCP adapter |
| `docs/09-config.md` | Config file and CLI |
| `docs/10-roadmap.md` | Milestones M0–M11 with acceptance criteria |
