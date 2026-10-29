"""Wires config, provider, tools and session into an AgentLoop. Shared by every UI.

`App` is the small command API a UI talks to (docs/06): submit a turn, switch model, start
a new session. UIs cancel a turn by cancelling the task running `submit`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from loom import __version__
from loom.broker.approval import Approvals
from loom.broker.broker import ToolBroker
from loom.broker.registry import ToolPackage, discover_packages
from loom.broker.secrets import KeyringKeychain, SecretStore
from loom.config import ApprovalMode, LoomConfig
from loom.core import events as ev
from loom.core.agent import AgentLoop
from loom.core.context import ContextBuilder
from loom.core.events import EventBus
from loom.core.mentions import expand_mentions
from loom.kernel.client import ExecClock, KernelClient
from loom.kernel.protocol import VarInfo
from loom.providers.base import Provider
from loom.providers.profiles import ModelProfile, resolve_profile
from loom.providers.registry import create_provider
from loom.sandbox.base import SandboxBackend, SessionInfo, create_backend, sandbox_env
from loom.session.store import SessionLog, new_session
from loom.tools.base import ToolContext, ToolSet
from loom.tools.budget import OutputBudget
from loom.tools.builtin.bash import BashTool
from loom.tools.builtin.edit import EditTool
from loom.tools.builtin.execute import ExecuteTool
from loom.tools.builtin.read import ReadTool
from loom.tools.builtin.search_tools import SearchToolsTool
from loom.tools.builtin.write import WriteTool
from loom.tools.paths import PathPolicy, ReadSet


def build_agent(
    workspace: Path,
    config: LoomConfig,
    bus: EventBus,
    *,
    provider: Provider | None = None,
    secrets: SecretStore | None = None,
    home: Path | None = None,
    model: str | None = None,
    backend: SandboxBackend | None = None,
    packages: Sequence[ToolPackage] | None = None,
) -> AgentLoop:
    """Raises SandboxUnavailable if the configured backend can't be used. `packages`
    defaults to the installed tool packages."""
    profile = resolve_profile(model or config.default_model, config)
    secrets = secrets or SecretStore()
    provider = provider or create_provider(profile, secrets)
    backend = backend or create_backend(config.sandbox, loom_home=home)
    if packages is None:
        packages, _ = discover_packages()

    paths = new_session(workspace, home)
    session = SessionInfo(paths.id, workspace, paths.root)
    session_log = SessionLog(paths.log, secrets.redactor)
    session_log.append(
        "session_start",
        session_id=paths.id,
        workspace=str(workspace),
        model=profile.alias,
        provider=profile.provider,
        sandbox=backend.name,
        mode=config.approvals.mode,
        loom_version=__version__,
    )

    env = sandbox_env(session, config.sandbox.env_passthrough)
    workspace_writable = config.approvals.mode != "read-only"
    approvals = Approvals(
        config.approvals, sandboxed=backend.name != "none", bus=bus, session_log=session_log
    )
    broker = ToolBroker(
        packages,
        approvals=approvals,
        secrets=secrets,
        session_log=session_log,
        package_config=config.packages,
        call_timeout_s=config.broker.call_timeout_s,
        log_results=config.tracing.log_tool_results,
    )

    async def handle_tool_call(
        tool: str, args: object, call_id: str, clock: ExecClock
    ) -> dict[str, Any]:
        reply = await broker.call(tool, args, call_id=call_id, clock=clock)
        return reply.message_for(call_id)

    kernel = KernelClient(
        backend,
        session,
        env=env,
        bus=bus,
        session_log=session_log,
        settings={"repr_max": REPR_MAX, "out_max_chars": config.budget.max_chars},
        workspace_writable=workspace_writable,
        catalog=broker.catalog(),
        tool_handler=handle_tool_call,
    )
    tools = ToolSet(
        [
            ReadTool(),
            WriteTool(),
            EditTool(),
            BashTool(),
            ExecuteTool(config.kernel.default_timeout_s),
            SearchToolsTool(broker),
        ],
        OutputBudget(config.budget, paths.out_dir),
        session_log,
        bus,
        approvals,
        secrets.redactor,
    )
    tool_ctx = ToolContext(
        workspace=workspace,
        paths=PathPolicy(workspace, read_roots=[paths.out_dir]),
        read_set=ReadSet(),
        bus=bus,
        model_accepts_images=profile.capabilities.images,
        sandbox=backend,
        sandbox_env=env,
        kernel=kernel,
        broker=broker,
        workspace_writable=workspace_writable,
    )
    return AgentLoop(
        provider=provider,
        profile=profile,
        tools=tools,
        tool_ctx=tool_ctx,
        context=ContextBuilder(profile, workspace, tools.names()),
        bus=bus,
        session_log=session_log,
        max_steps=config.agent.max_steps,
    )


REPR_MAX = 2000  # chars of a last-expression repr (docs/03 "Output budget")

ProviderFactory = Callable[[ModelProfile], Provider]


class App:
    def __init__(
        self,
        workspace: Path,
        config: LoomConfig,
        *,
        bus: EventBus | None = None,
        provider_factory: ProviderFactory | None = None,
        secrets: SecretStore | None = None,
        home: Path | None = None,
        packages: Sequence[ToolPackage] | None = None,
    ) -> None:
        self.workspace = workspace
        self.config = config
        self.bus = bus or EventBus()
        self.home = home
        self.secrets = secrets or SecretStore(keychain=KeyringKeychain())
        self._make_provider = provider_factory or (lambda p: create_provider(p, self.secrets))
        if packages is None:
            packages, self.notices = discover_packages()
        else:
            self.notices = []
        self.packages = list(packages)
        self.backend = create_backend(config.sandbox, loom_home=home)
        self.agent = self._build(config.default_model)

    def _build(self, alias: str) -> AgentLoop:
        profile = resolve_profile(alias, self.config)
        return build_agent(
            self.workspace,
            self.config,
            self.bus,
            provider=self._make_provider(profile),
            home=self.home,
            model=alias,
            backend=self.backend,
            secrets=self.secrets,
            packages=self.packages,
        )

    @property
    def profile(self) -> ModelProfile:
        return self.agent.profile

    @property
    def session_id(self) -> str:
        return self.agent.log.path.parent.name

    def model_aliases(self) -> list[str]:
        return sorted(self.config.model_aliases())

    def switch_model(self, alias: str) -> ModelProfile:
        """Raises KeyError for an unknown alias."""
        if alias not in self.config.model_aliases():
            raise KeyError(alias)
        profile = resolve_profile(alias, self.config)
        self.agent.switch_model(profile, self._make_provider(profile))
        return profile

    @property
    def kernel(self) -> KernelClient:
        kernel = self.agent.tool_ctx.kernel
        assert kernel is not None
        return kernel

    async def start(self) -> None:
        """Prepare the sandbox for this session (and start the kernel if `eager`)."""
        await self.backend.start(self._session_info())
        if self.config.kernel.eager:
            await self.kernel.start()

    async def aclose(self) -> None:
        await self.kernel.shutdown()
        await self.backend.stop()

    async def new_session(self) -> str:
        """Start a fresh session (new log, empty history, new kernel) on the current model."""
        await self.kernel.shutdown()
        await self.backend.stop()
        self.agent = self._build(self.agent.profile.alias)
        await self.start()
        return self.session_id

    @property
    def broker(self) -> ToolBroker:
        broker = self.agent.tool_ctx.broker
        assert broker is not None
        return broker

    @property
    def mode(self) -> ApprovalMode:
        approvals = self.agent.tools.approvals
        return approvals.mode if approvals is not None else self.config.approvals.mode

    def set_mode(self, mode: ApprovalMode) -> None:
        """Change the approval mode for this session and later ones (`/mode`)."""
        approvals = self.agent.tools.approvals
        if approvals is not None:
            approvals.mode = mode
        writable = mode != "read-only"
        self.agent.tool_ctx.workspace_writable = writable
        self.kernel.workspace_writable = writable  # applies to the next kernel start
        self.config = self.config.model_copy(
            update={"approvals": self.config.approvals.model_copy(update={"mode": mode})}
        )
        self.agent.log.append("mode_change", mode=mode)

    async def kernel_inventory(self) -> list[VarInfo]:
        return await self.kernel.inventory()

    async def reset_kernel(self) -> None:
        await self.kernel.reset()

    def _session_info(self) -> SessionInfo:
        return self.kernel.session

    async def submit(self, text: str) -> None:
        mentions = await expand_mentions(text, self.agent.tool_ctx, self.agent.tools.budget)
        for notice in mentions.notices:
            await self.bus.emit(ev.Notice(notice))
        if mentions.attached:
            await self.bus.emit(ev.Notice("Attached " + ", ".join(mentions.attached)))
        await self.agent.run_turn(text, mentions.blocks)
