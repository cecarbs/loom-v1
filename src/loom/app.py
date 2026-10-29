"""Wires config, provider, tools and session into an AgentLoop. Shared by every UI.

`App` is the small command API a UI talks to (docs/06): submit a turn, switch model, start
a new session. UIs cancel a turn by cancelling the task running `submit`.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from loom import __version__
from loom.broker.approval import Approvals
from loom.broker.secrets import SecretStore
from loom.config import LoomConfig
from loom.core import events as ev
from loom.core.agent import AgentLoop
from loom.core.context import ContextBuilder
from loom.core.events import EventBus
from loom.core.mentions import expand_mentions
from loom.kernel.client import KernelClient
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
) -> AgentLoop:
    """Raises SandboxUnavailable if the configured backend can't be used."""
    profile = resolve_profile(model or config.default_model, config)
    provider = provider or create_provider(profile, secrets or SecretStore())
    backend = backend or create_backend(config.sandbox.backend)

    paths = new_session(workspace, home)
    session = SessionInfo(paths.id, workspace, paths.root)
    session_log = SessionLog(paths.log)
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
    kernel = KernelClient(
        backend,
        session,
        env=env,
        bus=bus,
        session_log=session_log,
        settings={"repr_max": REPR_MAX, "out_max_chars": config.budget.max_chars},
        workspace_writable=workspace_writable,
    )
    approvals = Approvals(
        config.approvals, sandboxed=backend.name != "none", bus=bus, session_log=session_log
    )
    tools = ToolSet(
        [
            ReadTool(),
            WriteTool(),
            EditTool(),
            BashTool(),
            ExecuteTool(config.kernel.default_timeout_s),
        ],
        OutputBudget(config.budget, paths.out_dir),
        session_log,
        bus,
        approvals,
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
    ) -> None:
        self.workspace = workspace
        self.config = config
        self.bus = bus or EventBus()
        self.home = home
        secrets = secrets or SecretStore()
        self._make_provider = provider_factory or (lambda p: create_provider(p, secrets))
        self.backend = create_backend(config.sandbox.backend)
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
