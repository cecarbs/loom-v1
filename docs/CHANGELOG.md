# Changelog

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
