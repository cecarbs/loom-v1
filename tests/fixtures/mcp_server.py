"""A small MCP server for tests/test_mcp.py: stdio by default, `http <port>` for streamable HTTP."""

import asyncio
import os
import sys

from mcp.server.mcpserver import MCPServer
from mcp_types import ToolAnnotations

server = MCPServer("loom-test")
READ = ToolAnnotations(read_only_hint=True)


@server.tool(annotations=READ, structured_output=False)
def add(a: int, b: int = 1) -> int:
    """Add two numbers.

    Second paragraph."""
    return a + b


@server.tool(annotations=READ, structured_output=False)
def rows() -> str:
    """Rows as JSON text."""
    return '[{"host": "web-3", "count": 7}]'


@server.tool(annotations=READ, structured_output=False)
def env_var(name: str) -> str:
    """One environment variable of the server process."""
    return os.environ.get(name, "<unset>")


@server.tool(name="make-note", structured_output=False)
def make_note(text: str) -> str:
    """Store a note (no annotations: a write)."""
    return f"noted: {text}"


@server.tool(annotations=ToolAnnotations(destructive_hint=True), structured_output=False)
def wipe() -> str:
    """Delete everything."""
    return "wiped"


@server.tool(annotations=READ)
def boom() -> str:
    """Always fails."""
    raise ValueError("kaboom")


@server.tool(annotations=READ, structured_output=False)
def crash() -> str:
    """The server process dies."""
    os._exit(3)


@server.tool(annotations=READ, structured_output=False)
async def slow() -> str:
    """Takes a while."""
    await asyncio.sleep(60)
    return "done"


if __name__ == "__main__":
    print("fixture server stderr line", file=sys.stderr, flush=True)
    if sys.argv[1:2] == ["http"]:
        server.run("streamable-http", host="127.0.0.1", port=int(sys.argv[2]))
    else:
        server.run()
