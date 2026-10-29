# Changelog

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
