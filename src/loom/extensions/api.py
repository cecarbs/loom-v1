"""The extension API (docs/08 "Extensions") and the hook runner the core calls.

Kept free of runtime imports from the broker, tools and providers: they import this module.
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeVar

from loom.core.events import EventBus, Notice, Usage

if TYPE_CHECKING:
    from loom.broker.registry import ToolPackage
    from loom.broker.secrets import SecretStore
    from loom.providers.registry import ProviderFactory
    from loom.session.store import SessionLog
    from loom.tools.base import Tool, ToolResult

log = logging.getLogger(__name__)

HookEvent = Literal[
    "session_start",
    "before_model_request",
    "after_model_response",
    "before_tool_call",
    "after_tool_call",
    "before_broker_call",
    "after_broker_call",
    "turn_finished",
    "shutdown",
]
HOOK_EVENTS: frozenset[str] = frozenset(HookEvent.__args__)  # type: ignore[attr-defined]

P = TypeVar("P")
HookHandler = Callable[[Any], Any]  # sync or async; one payload argument


@dataclass(frozen=True)
class Block:
    """Returned from a `before_*` hook: don't do it. The reason goes back to the model."""

    reason: str


# --- hook payloads -------------------------------------------------------------------


@dataclass(frozen=True)
class SessionStart:
    session_id: str
    workspace: Path
    model: str
    resumed: bool


@dataclass
class ToolCall:
    """`before_tool_call` / `after_tool_call` (built-in tools). A `before` hook may change
    `input`; it is validated afterwards. `result` is set for `after_tool_call`."""

    id: str
    name: str
    input: dict[str, Any]
    result: ToolResult | None = None


@dataclass
class BrokerCall:
    """`before_broker_call` / `after_broker_call` (kernel tools). A `before` hook may change
    `args`; they are validated again. `ok`/`value`/`error` are set for `after_broker_call`."""

    tool: str
    args: dict[str, Any]
    effect: str
    call_id: str
    ok: bool | None = None
    value: Any = None
    error: str | None = None


@dataclass(frozen=True)
class TurnEnd:
    usage: Usage
    cancelled: bool = False


# --- what an extension registers -----------------------------------------------------


class CommandContext(Protocol):
    """What an extension command gets as `ctx` (the UI provides it)."""

    @property
    def app(self) -> Any: ...

    async def submit(self, text: str) -> None:
        """Send `text` as a user message and run the turn."""
        ...

    def print(self, text: str) -> None: ...


CommandHandler = Callable[[str, CommandContext], Awaitable[None] | None]


@dataclass(frozen=True)
class Hook:
    extension: str
    event: str
    handler: HookHandler


@dataclass(frozen=True)
class ExtensionCommand:
    extension: str
    name: str
    help: str
    handler: CommandHandler


@dataclass(frozen=True)
class ExtensionEnv:
    """What `register(api)` can read from the host: `[packages.*]` config, secrets, loom's home."""

    packages: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    secrets: SecretStore | None = None
    home: Path | None = None


class ExtensionAPI:
    """Passed to an extension's `register(api)`."""

    def __init__(self, name: str, env: ExtensionEnv | None = None) -> None:
        self.name = name
        self.env = env or ExtensionEnv()
        self.notices: list[str] = []
        self.hooks: list[Hook] = []
        self.commands: list[ExtensionCommand] = []
        self.packages: list[ToolPackage] = []
        self.providers: dict[str, ProviderFactory] = {}
        self.tools: list[Tool] = []

    def on(self, event: HookEvent, handler: HookHandler | None = None) -> Any:
        """Register a hook: `api.on("turn_finished", fn)` or `@api.on("turn_finished")`."""
        if event not in HOOK_EVENTS:
            known = ", ".join(sorted(HOOK_EVENTS))
            raise ValueError(f"unknown hook event {event!r} (known: {known})")

        def register(fn: HookHandler) -> HookHandler:
            self.hooks.append(Hook(self.name, event, fn))
            return fn

        return register(handler) if handler is not None else register

    def command(self, name: str, help: str = "") -> Callable[[CommandHandler], CommandHandler]:
        """`@api.command("incident")` on `async def handler(args: str, ctx)`."""
        name = name.removeprefix("/")
        if not name or any(c.isspace() or c == "/" for c in name):
            raise ValueError(f"invalid command name {name!r}")

        def register(fn: CommandHandler) -> CommandHandler:
            text = help or (inspect.getdoc(fn) or "").split("\n")[0]
            self.commands.append(ExtensionCommand(self.name, name, text, fn))
            return fn

        return register

    def package_config(self, name: str) -> dict[str, Any]:
        """The `[packages.<name>]` table (restricted: a project sets it only when trusted)."""
        return dict(self.env.packages.get(name, {}))

    def secret(self, name: str) -> str | None:
        """Resolve a secret on the host; the value is registered with the redactor. Tool
        packages should use `ctx.secret`: this is for what connects before a session exists."""
        return self.env.secrets.get(name) if self.env.secrets is not None else None

    def notice(self, text: str) -> None:
        """Show `text` to the user when the session starts."""
        self.notices.append(f"Extension {self.name}: {text}")

    def add_tool_package(self, package: ToolPackage) -> None:
        self.packages.append(package)

    def add_provider(self, name: str, factory: ProviderFactory) -> None:
        """`factory(profile, secrets) -> Provider`; use it with `provider = "<name>"`."""
        self.providers[name] = factory

    def add_builtin_tool(self, tool: Tool) -> None:
        """A model-visible tool (discouraged: prefer kernel functions in a tool package)."""
        self.tools.append(tool)


@dataclass
class Extensions:
    """Everything the loaded extensions registered."""

    names: list[str] = field(default_factory=list)
    hooks: list[Hook] = field(default_factory=list)
    commands: list[ExtensionCommand] = field(default_factory=list)
    packages: list[ToolPackage] = field(default_factory=list)
    providers: dict[str, ProviderFactory] = field(default_factory=dict)
    tools: list[Tool] = field(default_factory=list)
    notices: list[str] = field(default_factory=list)

    def add(self, api: ExtensionAPI) -> None:
        self.names.append(api.name)
        self.hooks.extend(api.hooks)
        self.commands.extend(api.commands)
        self.packages.extend(api.packages)
        self.providers.update(api.providers)
        self.tools.extend(api.tools)
        self.notices.extend(api.notices)


# --- the hook runner -----------------------------------------------------------------


@dataclass(frozen=True)
class Blocked:
    extension: str
    reason: str

    @property
    def text(self) -> str:
        return f"Blocked by extension {self.extension}: {self.reason}"


class Hooks:
    """Runs extension hooks for one session. A hook that raises is logged and shown as a
    notice, then treated as if it returned nothing: extensions never crash the loop."""

    def __init__(
        self,
        hooks: Sequence[Hook] = (),
        *,
        bus: EventBus | None = None,
        session_log: SessionLog | None = None,
    ) -> None:
        self._hooks = list(hooks)
        self.bus = bus
        self.log = session_log

    def has(self, event: HookEvent) -> bool:
        return any(h.event == event for h in self._hooks)

    async def before(self, event: HookEvent, payload: P) -> tuple[P, Blocked | None]:
        """Run `event` hooks in order. A returned `Block` stops there; a returned object of
        the payload's type replaces it for the hooks after (and the caller)."""
        for hook in self._hooks:
            if hook.event != event:
                continue
            ok, result = await self._call(hook, payload)
            if not ok or result is None:
                continue
            if isinstance(result, Block):
                blocked = Blocked(hook.extension, result.reason)
                if self.log is not None:
                    self.log.append(
                        "hook_block", extension=hook.extension, event=event, reason=result.reason
                    )
                return payload, blocked
            if isinstance(result, type(payload)):
                payload = result
        return payload, None

    async def after(self, event: HookEvent, payload: object) -> None:
        for hook in self._hooks:
            if hook.event == event:
                await self._call(hook, payload)

    async def _call(self, hook: Hook, payload: object) -> tuple[bool, Any]:
        try:
            result = hook.handler(payload)
            if inspect.isawaitable(result):
                result = await result
            return True, result
        except Exception as e:
            text = f"Extension {hook.extension}: {hook.event} hook failed: {type(e).__name__}: {e}"
            log.warning("%s", text, exc_info=True)
            if self.log is not None:
                self.log.append("error", text=text)
            if self.bus is not None:
                await self.bus.emit(Notice(text))
            return False, None
