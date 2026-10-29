# 06 — TUI (inline)

The UI runs **inline**: output goes to normal terminal scrollback (copy/paste and search work like any CLI), and an input prompt sits at the bottom. No full-screen takeover. Libraries: `prompt_toolkit` for input, `rich` for rendering.

The TUI only talks to the core through the event bus (`AgentEvent`s) and a small command API (`submit`, `cancel`, `respond`, slash-command handlers). It must not reach into core internals.

## Screen anatomy

```
╭─ you ─────────────────────────────────────────────╮
│ why did checkout errors spike at 2am?              │
╰────────────────────────────────────────────────────╯
I'll pull the error counts by host around 02:00.

▸ search_tools  "splunk"                        0.1s
▸ execute                                       2.4s
  │ rows = splunk.search('index=web status>=500 | stats count by host, uri',
  │                      earliest="-6h")
  │ sorted(rows, key=lambda r: -int(r["count"]))[:5]
  └ [{'host': 'web-3', 'uri': '/checkout', 'count': '4127'}, …]

web-3 accounts for 92% of the 500s, all on /checkout …

 > _
 sonnet · ctx 18% · $0.04 · seatbelt · ask · kernel idle (3 vars)
```

## Input (`tui/input.py`)

- `prompt_toolkit.PromptSession` with `prompt_async()`, persistent history at `~/.loom/history`.
- **Enter** submits; **Alt+Enter** or **Ctrl+J** inserts a newline; pasted multi-line text is kept as-is (bracketed paste).
- Completion: `/` slash commands, `@` file paths relative to the workspace (an `@path` mention attaches the file's contents to the message, subject to the read path policy).
- `bottom_toolbar` renders the status line: model alias, context %, session cost (`—` if the model has no pricing), sandbox backend, approval mode, kernel state + variable count.

## Rendering (`tui/render.py`)

Subscribes to the bus and renders with a single `rich.console.Console`:

| Event | Rendering |
|---|---|
| `TextDelta` | Accumulate; render the in-progress assistant message as `rich.markdown.Markdown` inside `rich.live.Live` (refresh ~8/s). On `AssistantMessageDone`, stop Live and print the final Markdown permanently. |
| `ThinkingDelta` | Dim, collapsed to one line ("thinking…") unless `/thinking on`. |
| `ToolCallStarted` | Header line `▸ <tool> <short args>`. For `execute`, print the code with `rich.syntax.Syntax(..., "python")`. |
| `ToolCallOutput` | Stream into a Live region (last ~12 lines visible). |
| `ToolCallFinished` | Collapse to a short preview (≤ 8 lines) + duration; errors in red. `edit`/`write` show a coloured unified diff. |
| `ApprovalRequested` | See below. |
| `Compacted`, `Notice`, `ErrorEvent` | One-line dim/yellow/red notices. |

**Terminal ownership rule (important pitfall):** prompt_toolkit and Rich `Live` must never draw at the same time. While a turn runs, the prompt is **not active**; the renderer owns the terminal. When the turn finishes, the renderer stops all Live regions, then the input loop calls `prompt_async()` again. Approval prompts temporarily stop Live, prompt, then resume. Wrap the app in `prompt_toolkit.patch_stdout()` only while the prompt is active.

Ctrl+C during a turn is caught via `loop.add_signal_handler(SIGINT, …)` and calls `app.cancel_turn()`. Sandboxed children run in their own process groups, so they never see the terminal's SIGINT directly.

Because the prompt is inactive during a turn, typing ahead/steering is not supported in v1.

## Approvals (`tui/approvals.py`)

```
⚠ splunk.create_saved_search  [write]
  name = "checkout-5xx-alert"
  spl  = "index=web status>=500 uri=/checkout | stats count"
  [y] allow once   [a] allow this tool for session   [n] deny   [m] deny with message
```

Single-key answer via a short prompt_toolkit key-binding app. Calls `bus.respond(request_id, Decision)`. Unanswered approvals never auto-allow. Concurrent requests are queued and shown one at a time; requests from sub-agents are prefixed with the sub-agent id.

## Keys

| Key | Idle | During a turn |
|---|---|---|
| Enter | submit | — |
| Ctrl+C | clear input; twice on empty input exits | cancel the turn |
| Ctrl+D | exit (on empty input) | — |
| Esc Esc | open `/undo` picker (later) | — |

## Slash commands (`tui/commands.py`)

| Command | Action |
|---|---|
| `/help` | list commands |
| `/model [alias]` | show or switch model |
| `/mode [read-only\|ask\|auto-write\|yolo]` | approval mode |
| `/kernel`, `/kernel reset` | inventory; restart kernel |
| `/tools [query]` | browse tool catalog |
| `/compact [instructions]` | manual compaction |
| `/undo` | restore files to the previous checkpoint |
| `/sessions`, `/resume <id>` | list / switch sessions |
| `/trace [n]` | show the last n tool/kernel calls with timing |
| `/cost` | token usage and cost |
| `/clear` | start a new session (same workspace) |
| `/thinking on\|off` | show or collapse thinking output (display only; enabling thinking is a model config setting) |
| `/exit` | quit |

Extensions can register more (`08-skills-extensions-subagents.md`).

## Future: Textual UI

A later `loom --ui textual` would subscribe to the same bus. Nothing in the core may assume a line-based UI.
