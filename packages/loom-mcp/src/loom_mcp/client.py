"""Connections to MCP servers.

`register(api)` runs before the broker's catalog is built, and it is sync, so the tool lists
must be known before it returns. The connections live on a daemon thread with its own event
loop, one long-lived task per server: the SDK's anyio scopes must be entered and left by the
same task. Tool calls hop onto that loop and back; cancelling one cancels it there too.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
from collections.abc import AsyncIterator, Callable, Iterator, Mapping
from pathlib import Path
from typing import Any, TextIO

import httpx2
from mcp import Client, Implementation, StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamable_http_client

from loom import __version__
from loom.broker import ToolPackage
from loom.session.store import loom_home
from loom_mcp.config import ServerConfig
from loom_mcp.tools import McpToolError, identifier, package_for, result_value

SECRET_PREFIX = "secret:"
CLOSE_TIMEOUT_S = 10.0


def reason(error: BaseException) -> str:
    """Task groups wrap the real failure in exception groups."""
    while isinstance(error, BaseExceptionGroup):
        error = error.exceptions[0]
    return f"{type(error).__name__}: {error}"


def resolve_secrets(
    values: Mapping[str, str], secret: Callable[[str], str | None]
) -> dict[str, str]:
    resolved = dict(values)
    for key, value in values.items():
        if value.startswith(SECRET_PREFIX):
            name = value.removeprefix(SECRET_PREFIX)
            found = secret(name)
            if not found:
                raise McpToolError(f"secret {name!r} is not set (loom secrets set {name})")
            resolved[key] = found
    return resolved


@contextlib.asynccontextmanager
async def _transport(
    config: ServerConfig, secret: Callable[[str], str | None], errlog: TextIO
) -> AsyncIterator[Any]:
    if config.url:
        headers = resolve_secrets(config.headers, secret)
        timeout = httpx2.Timeout(30, read=300)  # servers may hold a response stream open
        async with httpx2.AsyncClient(headers=headers, timeout=timeout) as http:
            async with streamable_http_client(config.url, http_client=http) as streams:
                yield streams
    else:
        assert config.command is not None
        params = StdioServerParameters(
            command=config.command,
            args=config.args,
            # On top of the SDK's minimal environment: nothing else of ours reaches the server.
            env=resolve_secrets(config.env, secret),
            cwd=Path(config.cwd).expanduser() if config.cwd else None,
        )
        async with stdio_client(params, errlog=errlog) as streams:
            yield streams


async def _list_tools(client: Client) -> list[Any]:
    tools: list[Any] = []
    cursor: str | None = None
    while True:
        page = await client.list_tools(cursor=cursor)
        tools.extend(page.tools)
        cursor = page.next_cursor
        if cursor is None:
            return tools


class McpManager:
    def __init__(
        self,
        servers: Mapping[str, ServerConfig],
        secret: Callable[[str], str | None],
        log_dir: Path | None = None,
    ) -> None:
        self.servers = servers
        self.secret = secret
        self.log_dir = log_dir or loom_home() / "logs"
        self._loop = asyncio.new_event_loop()
        self._clients: dict[str, Client] = {}
        self._gone: dict[str, str] = {}  # why a connected server dropped
        self._tasks: list[asyncio.Task[None]] = []
        self._stop: asyncio.Event | None = None

    def start(self) -> tuple[list[ToolPackage], list[str]]:
        """Connect to every server, at once; blocks until each is up or has failed."""

        def run() -> None:
            self._loop.run_forever()
            self._loop.close()

        threading.Thread(target=run, name="loom-mcp", daemon=True).start()
        return asyncio.run_coroutine_threadsafe(self._connect_all(), self._loop).result()

    async def aclose(self) -> None:
        future = asyncio.run_coroutine_threadsafe(self._close(), self._loop)
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(asyncio.wrap_future(future), CLOSE_TIMEOUT_S)
        self._loop.call_soon_threadsafe(self._loop.stop)

    async def call(self, server: str, tool: str, args: dict[str, Any]) -> Any:
        """Called on the host's loop."""
        future = asyncio.run_coroutine_threadsafe(self._call(server, tool, args), self._loop)
        return await asyncio.wrap_future(future)

    # --- on the manager's loop --------------------------------------------------------

    async def _connect_all(self) -> tuple[list[ToolPackage], list[str]]:
        self._stop = asyncio.Event()
        results = await asyncio.gather(*(self._connect(n, c) for n, c in self.servers.items()))
        packages = [package for package, _ in results if package is not None]
        return packages, [notice for _, notices in results for notice in notices]

    async def _connect(
        self, name: str, config: ServerConfig
    ) -> tuple[ToolPackage | None, list[str]]:
        ready: asyncio.Future[list[Any]] = self._loop.create_future()
        task = self._loop.create_task(self._serve(name, config, ready))
        self._tasks.append(task)
        try:
            tools = await asyncio.wait_for(ready, config.startup_timeout_s)
        except TimeoutError:
            task.cancel()
            return None, [
                f"server {name!r} didn't connect in {config.startup_timeout_s:g}s "
                f"(its stderr is in {self._log_path(name)})"
            ]
        except Exception as e:
            return None, [
                f"server {name!r} failed to connect: {reason(e)} "
                f"(its stderr is in {self._log_path(name)})"
            ]

        async def call(tool: str, args: dict[str, Any]) -> Any:
            return await self.call(name, tool, args)

        return package_for(name, config.description, tools, call)

    def _log_path(self, name: str) -> Path:
        return self.log_dir / f"mcp-{identifier(name)}.log"

    @contextlib.contextmanager
    def _errlog(self, name: str) -> Iterator[TextIO]:
        """A server's stderr: on the terminal it would write over the inline UI."""
        self.log_dir.mkdir(parents=True, exist_ok=True)
        with self._log_path(name).open("a", encoding="utf-8") as f:
            yield f

    async def _serve(
        self, name: str, config: ServerConfig, ready: asyncio.Future[list[Any]]
    ) -> None:
        """Hold the connection until shutdown; the SDK's scopes stay in this one task."""
        assert self._stop is not None
        try:
            with self._errlog(name) as errlog:
                info = Implementation(name="loom", version=__version__)
                async with Client(
                    _transport(config, self.secret, errlog), client_info=info
                ) as client:
                    tools = await _list_tools(client)
                    self._clients[name] = client
                    if not ready.done():
                        ready.set_result(tools)
                    await self._stop.wait()
        except Exception as e:
            self._gone[name] = reason(e)
            if not ready.done():
                ready.set_exception(e)
        finally:
            self._clients.pop(name, None)

    async def _call(self, server: str, tool: str, args: dict[str, Any]) -> Any:
        client = self._clients.get(server)
        if client is None:
            why = self._gone.get(server, "connection closed")
            raise McpToolError(f"server {server!r} is gone ({why}); restart loom to reconnect")
        return result_value(await client.call_tool(tool, args))

    async def _close(self) -> None:
        assert self._stop is not None
        self._stop.set()
        await asyncio.gather(*self._tasks, return_exceptions=True)
