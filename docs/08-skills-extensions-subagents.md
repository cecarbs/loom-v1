# 08 — Skills, extensions, sub-agents, MCP adapter

## Context files

Loaded into the system prompt at session start (in this order, concatenated with headers):
1. `~/.loom/AGENTS.md`
2. `AGENTS.md` and `CLAUDE.md` from the git root down to the cwd.

Outside a git repo, only the cwd's files are read. A `CLAUDE.md` that is a symlink to `AGENTS.md` is included once. Each file gets a `## <path>` header under `# Context files`. Total size capped at `[context] max_context_file_chars` (default 40,000); over the cap, truncate with a notice. Context files are not trust-gated (they are text, like the rest of the repo); they are re-read at each session start (`/clear`, `/resume`).

## Skills (`loom.skills`)

Format: a directory with a `SKILL.md` whose YAML frontmatter has `name` and `description` (the Agent Skills format used by Claude Code and Pi). Optional extra files (scripts, references) live alongside.

```markdown
---
name: splunk-triage
description: Use when investigating errors, latency or outages with Splunk data.
---
1. Run search_tools("splunk") to see available functions.
2. Start with aggregate SPL (stats count by host, status, uri) over the incident window …
3. Keep raw events in kernel variables; print only top-N tables …
```

Discovery paths, in precedence order: `~/.loom/skills/*/SKILL.md`, `<workspace>/.loom/skills/*/SKILL.md`, and (opt-in, `[skills] include_claude = true`) `~/.claude/skills/*/SKILL.md`. On a name clash the earlier one wins (a cloned repo can't shadow your own skill) and the later one is skipped with a notice, as are files without valid `name`/`description` frontmatter. Descriptions are folded to one line (max 300 chars).

Progressive disclosure: the system prompt lists only `- name — description (path)` under `# Skills`. The model reads the full `SKILL.md` with `read` when relevant; skill directories are readable by the file tools without approval (even under `~/.loom`). `/skills` lists them; `/skill <name> [message]` force-loads one into the next user message (sent right away with `message`, if given). The session log's `session_start` entry lists the skills and extensions in use.

An example skill ships in `examples/skills/splunk-triage/`; copy it to `~/.loom/skills/` to use it.

## Extensions (`loom.extensions`)

Python modules that customize loom without forking it.

Locations: `~/.loom/extensions/*.py` (trusted) and `<workspace>/.loom/extensions/*.py` (requires a one-time trust prompt per workspace, stored in `~/.loom/trust.json` so loom never rewrites the user's TOML). Files starting with `_` are skipped (helpers). Each module is imported on the host and its `register(api)` called once per loom process; the extension's name is the file stem.

Trust: when the workspace has project extensions, or project-config settings that are ignored untrusted, interactive `loom` lists them and asks `Trust <workspace>? [y/N]` before loading config. The same grant unlocks the restricted keys of the workspace's `.loom/config.toml` (`09-config.md`). The trust record holds a fingerprint (SHA-256) of the project config and every file under `.loom/extensions/` (bytecode caches excluded); if either changes — including by model-written code, since the sandbox can write the workspace — the grant no longer matches and loom asks again. A "no" isn't remembered. Subcommands (`loom models …`) use an existing grant but never ask. Untrusted, project extensions aren't loaded and a notice says so. When the workspace is `$HOME`, its `.loom/extensions` is the user's own dir and needs no grant.

```python
# ~/.loom/extensions/no_prod_writes.py
from loom.extensions import ExtensionAPI, Block

def register(api: ExtensionAPI) -> None:
    @api.on("before_broker_call")
    async def guard(call):
        if call.effect != "read" and "prod" in str(call.args):
            return Block("Writes mentioning prod are disabled by policy.")

    @api.command("incident")
    async def incident(args: str, ctx):
        await ctx.submit(f"Load the splunk-triage skill and investigate: {args}")
```

API surface:

| Method | Purpose |
|---|---|
| `api.on(event, handler)` | Hooks (also usable as `@api.on(event)`): see the table below. Returning `Block(reason)` from a `before_*` hook blocks the action. |
| `api.command(name, help="")` | Register a slash command: `handler(args: str, ctx)`; `ctx.submit(text)` sends a user message and runs the turn, `ctx.print(text)`, `ctx.app`. Names of built-in commands can't be taken (skipped with a notice). Help defaults to the docstring's first line. |
| `api.add_tool_package(pkg)` | Register a `ToolPackage` without an entry point |
| `api.add_provider(name, factory)` | Register a model provider: `factory(profile, secrets) -> Provider`, used by models with `provider = "<name>"` (built-in names can't be replaced). `loom models …` subcommands don't load extensions. |
| `api.add_builtin_tool(tool)` | Add a model-visible tool (discouraged; prefer kernel functions) |

Hook handlers may be sync or async and get one payload (`from loom.extensions import …`):

| Event | Payload | Effect of the return value |
|---|---|---|
| `session_start` | `SessionStart(session_id, workspace, model, resumed)` | ignored |
| `before_model_request` | `ModelRequest` | a `ModelRequest` replaces it; `Block` ends the turn with an error (the request is not sent) |
| `after_model_response` | the assistant `Message` | ignored |
| `before_tool_call` | `ToolCall(id, name, input)` | mutate `input` (validated afterwards); `Block` → error tool result `Blocked by extension <name>: <reason>` |
| `after_tool_call` | `ToolCall` with `result: ToolResult` | mutate `result`; redaction and the output budget still apply after |
| `before_broker_call` | `BrokerCall(tool, args, effect, call_id)` | mutate `args` (validated again); `Block` → the kernel stub raises `PermissionDenied("Blocked by extension <name>: <reason>")`, before the approval policy runs |
| `after_broker_call` | `BrokerCall` with `ok`, `value`, `error` | ignored |
| `turn_finished` | `TurnEnd(usage, cancelled)` | ignored |

Hooks run in extension load order (user extensions first, then project, each sorted by file name); the first `Block` wins. Blocks are logged (`hook_block`, and `blocked` on the `broker_call` entry).

Extension errors (at import, in `register`, in a hook or a command) are caught, logged, and shown as a notice; they never crash the loop. A hook that raises counts as returning nothing; an extension whose `register` raises is dropped entirely. An example lives in `examples/extensions/no_prod_writes.py`.

## Sub-agents (`loom.agents`)

Exposed **inside the kernel** so they compose with code:

```python
summary = agents.run("Find which deploy preceded the 02:00 spike on web-3", max_turns=15)
results = agents.map(
    [f"Summarize 5xx errors for {h} in the last 6h" for h in top_hosts],
    concurrency=4,
)
```

- Stubs call the broker (`agents.run` is a broker tool with effect `read`; it inherits the parent's approval mode). `agents.*` calls are exempt from `[broker] call_timeout_s` and pause the parent's `execute` timeout clock; they are bounded by `max_turns` (default 15) instead.
- Each sub-agent is a headless `AgentLoop` with: its own history, its **own fresh kernel** (same sandbox backend), the same tool catalog, an optional `model=` alias override, and a system prompt suffix telling it to return a concise final answer.
- Approval requests from sub-agents bubble up to the TUI, labelled with the sub-agent id.
- Depth limit 1 (sub-agents cannot spawn sub-agents). Max concurrency `[agents] max_concurrency` (default 4).
- Returns the final assistant text (string). Full sub-agent transcripts are written to the parent session dir and referenced in the log.

## MCP adapter (extension, later)

`loom-mcp` extension: connects to configured MCP servers (stdio or streamable HTTP) on the host and registers each MCP tool as a broker tool named `mcp_<server>.<tool>`, with stubs in the kernel like any other package.

- Effect mapping from MCP tool annotations: `readOnlyHint` → `read`, `destructiveHint` → `destructive`, otherwise `write`.
- MCP servers run on the host (outside the sandbox) and are trusted like tool packages; their credentials stay host-side.
- This makes MCP servers usable in code mode without loading their schemas into the model's context: the model discovers them with `search_tools`.
