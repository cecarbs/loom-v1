"""Wires config, provider, tools and session into an AgentLoop. Shared by every UI."""

from __future__ import annotations

from pathlib import Path

from loom import __version__
from loom.broker.secrets import SecretStore
from loom.config import LoomConfig
from loom.core.agent import AgentLoop
from loom.core.context import ContextBuilder
from loom.core.events import EventBus
from loom.providers.base import Provider
from loom.providers.profiles import resolve_profile
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
) -> AgentLoop:
    profile = resolve_profile(config.default_model, config)
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
