# 08 — Skills, extensions, sub-agents, MCP adapter

## Context files

Loaded into the system prompt at session start (in this order, concatenated with headers):
1. `~/.loom/AGENTS.md`
2. `AGENTS.md` and `CLAUDE.md` from the git root down to the cwd.

Total size capped at `[context] max_context_file_chars` (default 40,000); over the cap, truncate with a notice.

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

Discovery paths: `~/.loom/skills/*/SKILL.md`, `<workspace>/.loom/skills/*/SKILL.md`, and (opt-in, `[skills] include_claude = true`) `~/.claude/skills/*/SKILL.md`.

Progressive disclosure: the system prompt lists only `name — description (path)`. The model reads the full `SKILL.md` with `read` when relevant. `/skills` lists them; `/skill <name>` force-loads one into the next user message.

## Extensions (`loom.extensions`)

Python modules that customize loom without forking it.

Locations: `~/.loom/extensions/*.py` (trusted) and `<workspace>/.loom/extensions/*.py` (requires a one-time trust prompt per workspace, stored in `~/.loom/trust.json` so loom never rewrites the user's TOML). The same trust grant unlocks the restricted keys of the workspace's `.loom/config.toml` (`09-config.md`). The trust record includes a hash of the extensions dir and project config; if either changes, loom asks again.

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
| `api.on(event, handler)` | Hooks: `session_start`, `before_model_request` (may modify request), `after_model_response`, `before_tool_call` / `after_tool_call` (built-in tools), `before_broker_call` / `after_broker_call` (kernel tools), `turn_finished`. Returning `Block(reason)` from a `before_*` hook blocks the action; the reason is returned to the model. |
| `api.command(name)` | Register a slash command |
| `api.add_tool_package(pkg)` | Register a `ToolPackage` without an entry point |
| `api.add_provider(name, factory)` | Register a model provider |
| `api.add_builtin_tool(tool)` | Add a model-visible tool (discouraged; prefer kernel functions) |

Extension errors are caught, logged, and shown as a notice; they never crash the loop.

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
