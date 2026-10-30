"""loom-mcp: connects to the MCP servers in `[packages.mcp.servers.<name>]` and registers each
one as a tool package `mcp_<name>` (docs/08 "MCP adapter").

Servers run on the host, outside the sandbox, and are trusted like tool packages. The config
is restricted (docs/09): a project sets it only in a trusted workspace, since it chooses what
runs. The MCP SDK is imported only when a server is configured.
"""

from __future__ import annotations

from pydantic import ValidationError

from loom.extensions import ExtensionAPI
from loom_mcp.config import ServerConfig


def register(api: ExtensionAPI) -> None:
    servers: dict[str, ServerConfig] = {}
    for name, raw in (api.package_config("mcp").get("servers") or {}).items():
        try:
            servers[name] = ServerConfig.model_validate(raw)
        except ValidationError as e:
            # msg and loc only: the input may hold a token.
            problems = "; ".join(
                f"{'.'.join(str(p) for p in err['loc']) or '<server>'}: {err['msg']}"
                for err in e.errors()
            )
            api.notice(f"server {name!r} skipped: {problems}")
    if not servers:
        return

    from loom_mcp.client import McpManager

    home = api.env.home
    manager = McpManager(servers, api.secret, home / "logs" if home else None)
    packages, notices = manager.start()
    for package in packages:
        api.add_tool_package(package)
    for notice in notices:
        api.notice(notice)
    api.on("shutdown", lambda _: manager.aclose())
