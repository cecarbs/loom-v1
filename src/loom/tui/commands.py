"""Slash command framework (docs/06 "Slash commands"). Extensions register more in M8."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import get_args

from rich.console import Console
from rich.table import Table
from rich.text import Text

from loom.app import App
from loom.config import ApprovalMode
from loom.tui.render import Renderer, format_cost


@dataclass
class CommandContext:
    app: App
    console: Console
    renderer: Renderer
    commands: CommandRegistry
    exit_requested: bool = False


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
        table.add_row(Text(f"/{command.name} {command.usage}".rstrip(), "bold"), command.help)
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
    ctx.renderer.status.context_pct = 0.0
    ctx.renderer.status.cost_usd = ctx.app.agent.cost_usd
    ctx.console.print(Text(f"New session {session_id}", style="dim"))


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


def _exit(ctx: CommandContext, args: str) -> None:
    ctx.exit_requested = True


def builtin_commands() -> CommandRegistry:
    registry = CommandRegistry()
    for command in BUILTINS:
        registry.register(command)
    return registry


BUILTINS = (
    Command("help", "list commands", _help),
    Command("model", "show or switch the model", _model, "[alias]"),
    Command("cost", "token usage and cost for this session", _cost),
    Command("clear", "start a new session in this workspace", _clear),
    Command("kernel", "show kernel variables, or restart it", _kernel, "[reset]"),
    Command("tools", "browse kernel tool functions", _tools, "[query]"),
    Command("mode", "show or set the approval mode", _mode, "[read-only|ask|auto-write|yolo]"),
    Command("thinking", "show or collapse thinking output", _thinking, "on|off"),
    Command("exit", "quit", _exit, aliases=("quit",)),
)
