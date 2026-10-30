# 06 — TUI (inline)

The UI runs **inline**: output goes to normal terminal scrollback (copy/paste and search work like any CLI), and an input prompt sits at the bottom. No full-screen takeover. Libraries: `prompt_toolkit` for input, `rich` for rendering.

The TUI only talks to the core through the event bus (`AgentEvent`s) and a small command API, `loom.app.App`: `submit(text)` (runs a turn; the UI cancels it by cancelling that task), `switch_model(alias)`, `new_session()`, plus `bus.respond(...)` for approvals and read-only state (`profile`, `agent.usage`, `agent.cost_usd`). `loom.app` is shared by every UI and, like the core, never imports `loom.tui`. Slash-command handlers get a `CommandContext` (app, console, renderer, registry).

`loom -p` has no renderer at all: `loom.oneshot.OneShot` subscribes to the bus, prints the final answer to stdout and everything else to stderr (`09-config.md` "CLI").

When stdin is not a terminal (piped input, scripts), the same renderer runs on a plain console without Live regions, input is read line by line, and approval requests resolve as deny.

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
 sonnet · ctx 18% · $0.04 · docker · ask · kernel idle (3 vars)
```

## Input (`tui/input.py`)

- `prompt_toolkit.PromptSession` with `prompt_async()`, persistent history at `~/.loom/history`.
- **Enter** submits; **Alt+Enter**, **Ctrl+J** or **Shift+Enter** (in terminals that report it, via xterm modifyOtherKeys or `CSI u` encodings, which loom registers with prompt_toolkit's parser; unknown modified keys like Shift+Esc are ignored instead of being inserted as text) inserts a newline; pasted multi-line text is kept as-is (bracketed paste).
- Completion: `/` slash commands (at the start of the input), `@` file paths relative to the workspace (hidden entries only after a typed `.`; `.git`, `node_modules`, `.venv`, `__pycache__` skipped; never outside the workspace).
- `@path` mentions (`@` at the start of the input or after whitespace, so `me@example.com` is not one) attach the file to the user message as a `<file path="…">` text block, rendered exactly like a `read` result (numbered lines, images as image blocks). Attachments go through the read path policy (sensitive or outside paths are not attached; a notice says why) and the output budget, and count as a read so the model can `edit` the file directly. Mentions that aren't existing files stay plain text.
- A line is a slash command only if its first word is `/name` with no further `/`, so `/usr/bin/env is broken` is sent as a message.
- `bottom_toolbar` renders the status line: model alias, context %, session cost (`—` if the model has no pricing), sandbox backend, approval mode, kernel state + variable count (`kernel off` until the first `execute`, then e.g. `kernel idle (3 vars)`). prompt_toolkit only draws the toolbar once the terminal answers its cursor-position request, which every real terminal does. The toolbar sits on the window's bottom row, not directly under the prompt.
- The prompt is erased on submit (`erase_when_done`) and the input is re-printed as the `you` panel.

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

Live regions show only the last lines that fit on the screen (tool output: the last 12), because a region taller than the terminal can't be redrawn in place and leaves duplicates in scrollback. The full text is printed once when the region closes.

**Terminal ownership rule (important pitfall):** prompt_toolkit and Rich `Live` must never draw at the same time. While a turn runs, the prompt is **not active**; the renderer owns the terminal. When the turn finishes, the renderer stops all Live regions, then the input loop calls `prompt_async()` again. Approval prompts temporarily stop Live, prompt, then resume with a *new* `Live` (a restarted one remembers its old height and overwrites the lines the prompt printed). Wrap the app in `prompt_toolkit.patch_stdout()` only while the prompt is active.

While a turn runs, terminal echo is off (`ECHO`/`ECHOCTL` cleared, `ISIG` kept): an echoed `^C` or typed key would land inside the Live region, shift the cursor, and make its erase miss a line (seen as a duplicated first line). Keys typed during the turn are flushed when it ends.

Ctrl+C during a turn is caught via `loop.add_signal_handler(SIGINT, …)` and cancels the turn's task. Sandboxed children run in their own process groups, so they never see the terminal's SIGINT directly.

Because the prompt is inactive during a turn, typing ahead/steering is not supported in v1.

## Approvals (`tui/approvals.py`)

```
⚠ splunk.create_saved_search  [write]
  name = "checkout-5xx-alert"
  spl  = "index=web status>=500 uri=/checkout | stats count"
  [y] allow once   [a] allow this tool for session   [n] deny   [m] deny with message
```

Single-key answer via a short prompt_toolkit key-binding app. Calls `bus.respond(request_id, Decision)`. Unanswered approvals never auto-allow. Concurrent requests are queued and shown one at a time; requests from sub-agents are prefixed with the sub-agent id. While any prompt is up or queued, the live region stays down and notices are held back, then printed when the last prompt is answered.

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
| `/tutorial` | a short tour on the demo package (steps to type); says how to get `loom-tools-demo` if it isn't installed. Interactive `loom` mentions it on the very first run (no `~/.loom/sessions` yet) |
| `/model [alias]` | show or switch model |
| `/mode [read-only\|ask\|auto-write\|yolo]` | approval mode (kept for later sessions; not saved to config) |
| `/kernel`, `/kernel reset` | inventory; restart kernel |
| `/tools [query]` | browse tool catalog |
| `/skills`, `/skill <name> [message]` | list skills; attach a skill's `SKILL.md` to the next message (or send `message` with it now) |
| `/compact [instructions]` | manual compaction (Ctrl+C cancels it; history is then unchanged) |
| `/undo [n]` | list this session's checkpoints whose turn changed files (newest = 1) and ask which to restore (Enter = 1); `/undo n` restores directly, and is the only form when there's no terminal to ask on |
| `/sessions`, `/resume <id>` | list / switch sessions (`<id>` may be a unique prefix); resuming prints a short recap |
| `/trace [n]` | the trace tree of this session (the last n turns): model calls, tool calls, kernel executions and their broker calls, with timing (docs/07) |
| `/cost` | token usage and cost |
| `/clear` | start a new session (same workspace) |
| `/thinking on\|off` | show or collapse thinking output (display only; enabling thinking is a model config setting) |
| `/exit` | quit |

Extensions can register more (`08-skills-extensions-subagents.md`).

## Future: Textual UI

A later `loom --ui textual` would subscribe to the same bus. Nothing in the core may assume a line-based UI.
