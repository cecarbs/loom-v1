"""Wires config, provider, tools and session into an AgentLoop. Shared by every UI.

`App` is the small command API a UI talks to (docs/06): submit a turn, switch model, start
a new session. UIs cancel a turn by cancelling the task running `submit`.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from loom import __version__
from loom.broker.secrets import SecretStore
from loom.config import LoomConfig
from loom.core import events as ev
from loom.core.agent import AgentLoop
from loom.core.context import ContextBuilder
from loom.core.events import EventBus
from loom.core.mentions import expand_mentions
from loom.providers.base import Provider
from loom.providers.profiles import ModelProfile, resolve_profile
from loom.providers.registry import create_provider
from loom.session.store import SessionLog, new_session
from loom.tools.base import ToolContext, ToolSet
from loom.tools.budget import OutputBudget
from loom.tools.builtin.edit import EditTool
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
) -> AgentLoop:
    profile = resolve_profile(model or config.default_model, config)
    provider = provider or create_provider(profile, secrets or SecretStore())

    paths = new_session(workspace, home)
    session_log = SessionLog(paths.log)
    session_log.append(
        "session_start",
        session_id=paths.id,
        workspace=str(workspace),
        model=profile.alias,
        provider=profile.provider,
        sandbox=config.sandbox.backend,
        mode=config.approvals.mode,
        loom_version=__version__,
    )

    tools = ToolSet(
        [ReadTool(), WriteTool(), EditTool()],
        OutputBudget(config.budget, paths.out_dir),
        session_log,
        bus,
    )
    tool_ctx = ToolContext(
        workspace=workspace,
        paths=PathPolicy(workspace, read_roots=[paths.out_dir]),
        read_set=ReadSet(),
        bus=bus,
        model_accepts_images=profile.capabilities.images,
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

    def new_session(self) -> str:
        """Start a fresh session (new log, empty history) on the current model."""
        self.agent = self._build(self.agent.profile.alias)
        return self.session_id

    async def submit(self, text: str) -> None:
        mentions = await expand_mentions(text, self.agent.tool_ctx, self.agent.tools.budget)
        for notice in mentions.notices:
            await self.bus.emit(ev.Notice(notice))
        if mentions.attached:
            await self.bus.emit(ev.Notice("Attached " + ", ".join(mentions.attached)))
        await self.agent.run_turn(text, mentions.blocks)
