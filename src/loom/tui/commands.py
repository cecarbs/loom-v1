"""Slash command framework (docs/06 "Slash commands"); extensions register more (docs/08)."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass, field
from typing import Any, get_args

from rich.console import Console
from rich.table import Table
from rich.text import Text

from loom.app import App
from loom.config import ApprovalMode
from loom.extensions.api import ExtensionCommand
from loom.providers.profiles import profile_notices
from loom.session.checkpoints import UndoEntry
from loom.session.store import SessionError, SessionSummary
from loom.tui.render import Renderer, format_cost

Ask = Callable[[str], Awaitable[str | None]]  # prompt → answer (None: cancelled)
Runner = Callable[[Coroutine[Any, Any, Any]], Awaitable[None]]
Sender = Callable[[str], Awaitable[None]]  # shows a user message and runs its turn


@dataclass
class CommandContext:
    app: App
    console: Console
    renderer: Renderer
    commands: CommandRegistry
    exit_requested: bool = False
    ask: Ask | None = None  # None when there's no terminal to ask on
    run: Runner | None = None  # runs cancellable work (Ctrl+C) the way a turn runs
    send: Sender | None = None

    async def submit(self, text: str) -> None:
        """Send `text` as a user message and run the turn (extension commands use this)."""
        if self.send is not None:
            await self.send(text)
        else:
            await self.app.submit(text)

    def print(self, text: str) -> None:
        self.console.print(Text(text), soft_wrap=True)


Handler = Callable[[CommandContext, str], Awaitable[None] | None]


@dataclass(frozen=True)
class Command:
    name: str
    help: str
    handler: Handler
    usage: str = ""
    aliases: tuple[str, ...] = ()


@dataclass
class CommandRegistry:
    _commands: dict[str, Command] = field(default_factory=dict)
    _aliases: dict[str, str] = field(default_factory=dict)

    def register(self, command: Command) -> None:
        self._commands[command.name] = command
        for alias in command.aliases:
            self._aliases[alias] = command.name

    def get(self, name: str) -> Command | None:
        return self._commands.get(self._aliases.get(name, name))

    def commands(self) -> list[Command]:
        return sorted(self._commands.values(), key=lambda c: c.name)

    @staticmethod
    def is_command(line: str) -> bool:
        """`/help` is a command; `/usr/bin/env is broken` is a message about a path."""
        word = line.strip().split(maxsplit=1)[0] if line.strip() else ""
        return word.startswith("/") and len(word) > 1 and "/" not in word[1:]

    async def run(self, ctx: CommandContext, line: str) -> None:
        name, _, args = line.strip()[1:].partition(" ")
        command = self.get(name)
        if command is None:
            ctx.console.print(Text(f"Unknown command /{name}. Type /help.", style="red"))
            return
        result = command.handler(ctx, args.strip())
        if inspect.isawaitable(result):
            await result


# --- built-in commands -------------------------------------------------------------


def _help(ctx: CommandContext, args: str) -> None:
    table = Table.grid(padding=(0, 2))
    for command in ctx.commands.commands():
        table.add_row(Text(f"/{command.name} {command.usage}".rstrip(), "bold"), Text(command.help))
    ctx.console.print(table)
    ctx.console.print(
        Text(
            "Enter sends · Shift+Enter, Alt+Enter or Ctrl+J for a newline · "
            "@path attaches a file · Ctrl+C cancels a turn · Ctrl+D quits",
            style="dim",
        )
    )


def _model(ctx: CommandContext, args: str) -> None:
    app = ctx.app
    if not args:
        profile = app.profile
        ctx.console.print(f"Model: [bold]{profile.alias}[/] ({profile.provider}: {profile.model})")
        ctx.console.print(Text("Available: " + ", ".join(app.model_aliases()), style="dim"))
        return
    try:
        profile = app.switch_model(args)
    except KeyError:
        ctx.console.print(
            Text(f"Unknown model '{args}'. Available: {', '.join(app.model_aliases())}", "red")
        )
        return
    ctx.renderer.status.cost_usd = app.agent.cost_usd
    ctx.console.print(f"Switched to [bold]{profile.alias}[/] ({profile.provider}: {profile.model})")
    for notice in profile_notices(profile):
        ctx.console.print(Text(notice, style="yellow"))


def _cost(ctx: CommandContext, args: str) -> None:
    agent = ctx.app.agent
    usage = agent.usage
    table = Table.grid(padding=(0, 2))
    table.add_row("Input tokens", f"{usage.input_tokens:,}")
    table.add_row("Cache reads", f"{usage.cache_read_tokens:,}")
    table.add_row("Cache writes", f"{usage.cache_write_tokens:,}")
    table.add_row("Output tokens", f"{usage.output_tokens:,}")
    table.add_row("Session cost", format_cost(agent.cost_usd))
    ctx.console.print(table)
    if agent.cost_usd is None:
        ctx.console.print(Text("No pricing configured for this model.", style="dim"))


async def _clear(ctx: CommandContext, args: str) -> None:
    session_id = await ctx.app.new_session()
    sync_status(ctx)
    ctx.console.print(Text(f"New session {session_id}", style="dim"))


def sync_status(ctx: CommandContext) -> None:
    agent = ctx.app.agent
    ctx.renderer.status.context_pct = agent.context_pct if agent.history else 0.0
    ctx.renderer.status.cost_usd = agent.cost_usd


def _sessions(ctx: CommandContext, args: str) -> None:
    sessions = ctx.app.sessions()
    if not sessions:
        ctx.console.print(Text("No sessions in this workspace.", style="dim"))
        return
    ctx.console.print(session_table(sessions[:20], current=ctx.app.session_id))
    if len(sessions) > 20:
        ctx.console.print(Text(f"… and {len(sessions) - 20} older", style="dim"))
    ctx.console.print(Text("Switch with /resume <id> (a unique prefix is enough).", style="dim"))


def session_table(sessions: list[SessionSummary], current: str | None = None) -> Table:
    table = Table(box=None, padding=(0, 2), show_edge=False)
    table.add_column("session", style="bold")
    table.add_column("last active", style="cyan")
    table.add_column("model", style="dim")
    table.add_column("first prompt", no_wrap=True, overflow="ellipsis")
    for s in sessions:
        marker = " *" if s.id == current else ""
        title = Text(s.title) if s.title else Text("(no prompts)", style="dim")
        table.add_row(s.id + marker, f"{s.modified:%Y-%m-%d %H:%M}", s.model or "?", title)
    return table


async def _resume(ctx: CommandContext, args: str) -> None:
    if not args:
        _sessions(ctx, args)
        return
    try:
        await ctx.app.resume(args)
    except SessionError as e:
        ctx.console.print(Text(str(e), style="red"))
        return
    sync_status(ctx)
    print_resumed(ctx.console, ctx.app)


def print_resumed(console: Console, app: App) -> None:
    """A short recap: the last few prompts and the final reply."""
    history = app.agent.history
    console.print(
        Text(
            f"Resumed session {app.session_id} · {app.agent.turns} "
            f"turn{'' if app.agent.turns == 1 else 's'}",
            style="bold",
        )
    )
    prompts = [m for m in history if m.role == "user" and m.text() and not m.meta.get("compaction")]
    if any(m.meta.get("compaction") for m in history):
        console.print(Text("  (earlier turns are compacted into a summary)", style="dim"))
    for msg in prompts[-3:]:
        line = msg.text().strip().splitlines()[0] if msg.text().strip() else ""
        console.print(Text(f"  > {line[:100]}", style="dim"))
    replies = [m for m in history if m.role == "assistant" and m.text().strip()]
    if replies:
        lines = replies[-1].text().strip().splitlines()
        shown = "\n".join(lines[:12]) + ("\n…" if len(lines) > 12 else "")
        console.print(Text(shown), soft_wrap=True)
    if app.kernel.state == "off" and any(
        m.role == "assistant" and any(b.name == "execute" for b in m.tool_uses()) for m in history
    ):
        console.print(Text("The kernel starts fresh: earlier variables are gone.", style="dim"))


async def _compact(ctx: CommandContext, args: str) -> None:
    work = ctx.app.compact(args)
    if ctx.run is not None:
        await ctx.run(work)
    else:
        await work
    sync_status(ctx)


async def _undo(ctx: CommandContext, args: str) -> None:
    entries = await ctx.app.undo_entries()
    if not entries:
        ctx.console.print(Text("No file changes to undo in this session.", style="dim"))
        return
    if not args:
        ctx.console.print(undo_table(entries[:10]))
        if ctx.ask is None:
            ctx.console.print(Text("Restore one with /undo <n>.", style="dim"))
            return
        shown = min(len(entries), 10)
        answer = await ctx.ask(f"Restore files to which checkpoint? [1-{shown}, Enter = 1] ")
        if answer is None:
            ctx.console.print(Text("Cancelled.", style="dim"))
            return
        args = answer.strip() or "1"
    if not args.isdigit() or not 1 <= int(args) <= len(entries):
        ctx.console.print(Text(f"Pick a checkpoint between 1 and {len(entries)}.", style="red"))
        return
    entry = entries[int(args) - 1]
    result = await ctx.app.undo(entry)
    ctx.console.print(
        f"Restored files to before turn {entry.checkpoint.turn}: "
        f"{len(result.restored)} restored, {len(result.removed)} removed."
    )
    ctx.console.print(
        Text(
            "Undo covers workspace files only, not kernel variables or external systems. "
            "/undo again can revert this.",
            style="dim",
        )
    )


CHANGE_WORDS = {"A": "new", "M": "edited", "D": "deleted"}


def undo_table(entries: list[UndoEntry]) -> Table:
    table = Table(box=None, padding=(0, 2), show_edge=False)
    table.add_column("#", style="bold", justify="right")
    table.add_column("before", style="cyan")
    table.add_column("prompt")
    table.add_column("changed", style="dim")
    for i, entry in enumerate(entries, 1):
        cp = entry.checkpoint
        paths = [f"{c.path} ({CHANGE_WORDS.get(c.status, 'changed')})" for c in entry.changes]
        files = ", ".join(paths[:3]) + (f", +{len(paths) - 3} more" if len(paths) > 3 else "")
        table.add_row(str(i), f"turn {cp.turn}", Text(cp.prompt), files)
    return table


async def _kernel(ctx: CommandContext, args: str) -> None:
    app = ctx.app
    if args == "reset":
        await app.reset_kernel()
        ctx.console.print(Text("Kernel restarted; its variables are gone.", style="dim"))
        return
    if args:
        ctx.console.print(Text("Usage: /kernel [reset]", style="red"))
        return
    kernel = app.kernel
    pid = f" (pid {kernel.pid})" if kernel.pid else ""
    ctx.console.print(f"Kernel: [bold]{kernel.state}[/]{pid} · sandbox {app.backend.describe()}")
    variables = await app.kernel_inventory()
    if not variables:
        ctx.console.print(Text("No variables.", style="dim"))
        return
    table = Table(box=None, padding=(0, 2), show_edge=False)
    table.add_column("name", style="bold")
    table.add_column("type", style="cyan")
    table.add_column("summary", style="dim")
    for var in variables:
        table.add_row(var.name, var.type, var.summary)
    ctx.console.print(table)


def _tools(ctx: CommandContext, args: str) -> None:
    ctx.console.print(Text(ctx.app.broker.search(args, limit=50)), soft_wrap=True)


def _mode(ctx: CommandContext, args: str) -> None:
    modes = get_args(ApprovalMode)
    if args:
        if args not in modes:
            ctx.console.print(Text(f"Unknown mode '{args}'. Modes: {', '.join(modes)}", "red"))
            return
        ctx.app.set_mode(args)  # type: ignore[arg-type]  # checked above
    ctx.console.print(f"Approval mode: [bold]{ctx.app.mode}[/]")
    if not args:
        ctx.console.print(Text("Modes: " + ", ".join(modes), style="dim"))


def _thinking(ctx: CommandContext, args: str) -> None:
    if args in ("on", "off"):
        ctx.renderer.show_thinking = args == "on"
    state = "shown" if ctx.renderer.show_thinking else "collapsed"
    ctx.console.print(Text(f"Thinking output {state}.", style="dim"))


def _skills(ctx: CommandContext, args: str) -> None:
    skills = ctx.app.skills
    if not skills:
        ctx.console.print(
            Text(
                "No skills found. Add <name>/SKILL.md under ~/.loom/skills or .loom/skills.",
                style="dim",
            )
        )
        return
    table = Table(box=None, padding=(0, 2), show_edge=False)
    table.add_column("skill", style="bold")
    table.add_column("source", style="cyan")
    table.add_column("description")
    for skill in skills:
        table.add_row(skill.name, skill.source, skill.description)
    ctx.console.print(table)
    ctx.console.print(Text("Load one into your next message with /skill <name>.", style="dim"))


async def _skill(ctx: CommandContext, args: str) -> None:
    name, _, message = args.partition(" ")
    if not name:
        _skills(ctx, "")
        return
    skill = ctx.app.find_skill(name)
    if skill is None:
        known = ", ".join(s.name for s in ctx.app.skills) or "none"
        ctx.console.print(Text(f"Unknown skill '{name}'. Skills: {known}", style="red"))
        return
    ctx.app.load_skill(skill)
    if message.strip():
        await ctx.submit(message.strip())
    else:
        ctx.console.print(Text(f"{skill.name} will be sent with your next message.", style="dim"))


def extension_command(ext: ExtensionCommand) -> Command:
    """Wrap an extension's `handler(args, ctx)`; its errors are shown, never raised."""

    async def run(ctx: CommandContext, args: str) -> None:
        try:
            result = ext.handler(args, ctx)
            if inspect.isawaitable(result):
                await result
        except Exception as e:
            ctx.app.agent.log.append("error", text=f"extension {ext.extension} /{ext.name}: {e}")
            ctx.console.print(
                Text(
                    f"Extension {ext.extension}: /{ext.name} failed: {type(e).__name__}: {e}",
                    style="yellow",
                )
            )

    return Command(ext.name, f"{ext.help} ({ext.extension} extension)".lstrip(), run, "[args]")


def _exit(ctx: CommandContext, args: str) -> None:
    ctx.exit_requested = True


def builtin_commands(
    extensions: list[ExtensionCommand] | None = None, notices: list[str] | None = None
) -> CommandRegistry:
    """Built-ins, then extension commands; a clash keeps the existing command and adds a
    notice."""
    registry = CommandRegistry()
    for command in BUILTINS:
        registry.register(command)
    for ext in extensions or ():
        if registry.get(ext.name) is not None:
            if notices is not None:
                notices.append(
                    f"Extension {ext.extension}: /{ext.name} is already a command; skipped."
                )
            continue
        registry.register(extension_command(ext))
    return registry


TUTORIAL = """\
A short tour on made-up web logs and support tickets; nothing here touches a real system.
Type each quoted line as a message.

1. Code mode: "Using the demo tools, which host had the most 500 errors, and when?"
   The rows land in a kernel variable; only the summary the code prints reaches the model.
2. The kernel persists: run /kernel to see that variable, then ask a follow-up:
   "From those rows, what was the p95 latency on /checkout by hour?"
3. Approvals: "Open a demo ticket for this incident."
   Writes to outside systems stop for your answer, whatever the code that made them.
4. Undo: "Write a short incident note to incident.md", then /undo to restore the workspace.
5. Trace: /trace shows every model call, kernel execution and tool call of this session.

Next: /tools to browse functions, /skills, /help. Scripts can use: loom -p "<prompt>"
"""


def _tutorial(ctx: CommandContext, args: str) -> None:
    if "demo" not in ctx.app.broker.packages:
        ctx.console.print(
            Text(
                "The tutorial needs the demo tools. Install loom-tools-demo next to loom, "
                "then restart.",
                style="yellow",
            )
        )
        return
    ctx.console.print(Text(TUTORIAL))


def _trace(ctx: CommandContext, args: str) -> None:
    from loom.tracing.log import build_trace, render_trace

    arg = args.strip()
    if arg and not arg.isdigit():
        ctx.console.print(Text("Usage: /trace [n]  (the last n turns)", style="dim"))
        return
    trace = build_trace(ctx.app.agent.log.read())
    lines = render_trace(trace, last=int(arg) if arg else None)
    if not lines:
        ctx.console.print(Text("Nothing traced yet.", style="dim"))
        return
    for line in lines:
        ctx.console.print(Text(line, no_wrap=True, overflow="ellipsis"))
    ctx.console.print(Text(f"Kernel code: loom trace {ctx.app.session_id} --code", style="dim"))


BUILTINS = (
    Command("help", "list commands", _help),
    Command("tutorial", "a short tour using the demo tools", _tutorial),
    Command("model", "show or switch the model", _model, "[alias]"),
    Command("cost", "token usage and cost for this session", _cost),
    Command("clear", "start a new session in this workspace", _clear),
    Command("sessions", "list this workspace's sessions", _sessions),
    Command("resume", "switch to another session", _resume, "<id>"),
    Command("trace", "show this session's trace tree", _trace, "[n]"),
    Command("compact", "summarize older messages to free context", _compact, "[instructions]"),
    Command("undo", "restore files to a checkpoint", _undo, "[n]"),
    Command("kernel", "show kernel variables, or restart it", _kernel, "[reset]"),
    Command("tools", "browse kernel tool functions", _tools, "[query]"),
    Command("skills", "list available skills", _skills),
    Command("skill", "load a skill into the next message", _skill, "<name> [message]"),
    Command("mode", "show or set the approval mode", _mode, "[read-only|ask|auto-write|yolo]"),
    Command("thinking", "show or collapse thinking output", _thinking, "on|off"),
    Command("exit", "quit", _exit, aliases=("quit",)),
)
