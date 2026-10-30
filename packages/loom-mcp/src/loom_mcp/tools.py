"""MCP tools as loom tools: names, signatures, effects and result values. No SDK imports."""

from __future__ import annotations

import inspect
import json
import keyword
import re
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel

from loom.broker import Effect, PackageError, ToolPackage

Call = Callable[[str, dict[str, Any]], Awaitable[Any]]  # (MCP tool name, arguments)

_TYPES: dict[str, Any] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}


class McpToolError(Exception):
    """The server reported a failure, or can't be reached."""


def identifier(raw: str) -> str:
    """`raw` as a public Python identifier (kernel stubs and packages need them)."""
    name = re.sub(r"\W", "_", raw, flags=re.ASCII).lstrip("_") or "x"
    if name[0].isdigit():
        name = "x" + name
    return name + "_" if keyword.iskeyword(name) else name


def effect_of(annotations: Any) -> Effect:
    """Annotations are hints from a trusted server. If they conflict, the stricter one wins."""
    if annotations is None:
        return "write"
    if annotations.destructive_hint:
        return "destructive"
    return "read" if annotations.read_only_hint else "write"


def _annotation(schema: dict[str, Any]) -> Any:
    branches = schema.get("anyOf") or schema.get("oneOf")
    if isinstance(branches, list):
        kinds = [b.get("type") if isinstance(b, dict) else None for b in branches]
    elif isinstance(schema.get("type"), list):
        kinds = schema["type"]
    else:
        return _TYPES.get(schema.get("type") or "", Any)
    concrete = [k for k in kinds if k != "null"]
    base = _TYPES.get(concrete[0] or "", Any) if len(concrete) == 1 else Any
    return base | None if base is not Any and "null" in kinds else base


def _parameter_name(raw: str, taken: set[str]) -> str:
    name = identifier(raw)
    if hasattr(BaseModel, name):  # `json`, `copy`…: the args model can't have these fields
        name += "_"
    while name in taken:
        name += "_"
    taken.add(name)
    return name


def _function(tool: Any, call: Call) -> Callable[..., Any]:
    """An async function with the tool's signature, in the form `make_spec` reads. Required
    parameters come first, as Python needs."""
    schema = tool.input_schema
    properties: dict[str, Any] = schema.get("properties") or {}
    required = set(schema.get("required") or [])
    params = [inspect.Parameter("ctx", inspect.Parameter.POSITIONAL_ONLY)]
    remote: dict[str, str] = {}
    described: list[str] = []
    taken = {"ctx"}
    for raw in sorted(properties, key=lambda p: p not in required):
        prop = properties[raw] if isinstance(properties[raw], dict) else {}
        name = _parameter_name(raw, taken)
        remote[name] = raw
        annotation = _annotation(prop)
        default: Any = inspect.Parameter.empty
        if raw not in required:
            default = prop.get("default")
            if default is None and annotation is not Any:
                annotation = annotation | None
        params.append(
            inspect.Parameter(
                name,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                default=default,
                annotation=annotation,
            )
        )
        if prop.get("description"):
            described.append(f"  {name}: {prop['description']}")

    async def function(ctx: Any, /, **kwargs: Any) -> Any:
        # None means "not given" for an optional parameter: leave it to the server.
        return await call(
            tool.name,
            {remote[k]: v for k, v in kwargs.items() if v is not None or remote[k] in required},
        )

    function.__name__ = function.__qualname__ = identifier(tool.name)
    doc = tool.description or tool.title or ""
    function.__doc__ = doc + ("\n\nArgs:\n" + "\n".join(described) if described else "")
    function.__signature__ = inspect.Signature(params)  # type: ignore[attr-defined]
    return function


def package_for(
    server: str, description: str, tools: list[Any], call: Call
) -> tuple[ToolPackage, list[str]]:
    """The tool package of a server, and a notice per tool loom can't offer."""
    package = ToolPackage(
        name=f"mcp_{identifier(server)}",
        description=description or f"Tools from the {server} MCP server.",
    )
    notices: list[str] = []
    for tool in tools:
        try:
            package.tool(effect=effect_of(tool.annotations))(_function(tool, call))
        except (PackageError, ValueError, TypeError) as e:
            notices.append(f"server {server!r}: skipped tool {tool.name!r}: {e}")
    return package, notices


def _text_value(text: str) -> Any:
    """Many servers put JSON in a text block; in the kernel it's more use as data."""
    if text.lstrip()[:1] in ("{", "["):
        try:
            return json.loads(text)
        except ValueError:
            pass
    return text


def _block_value(block: Any) -> Any:
    match block.type:
        case "text":
            return _text_value(block.text)
        case "image" | "audio":
            return {"type": block.type, "mime_type": block.mime_type, "data": block.data}
        case "resource":
            resource = block.resource
            body = {"text": resource.text} if hasattr(resource, "text") else {"blob": resource.blob}
            return {"type": "resource", "uri": str(resource.uri), **body}
        case "resource_link":
            return {"type": "resource_link", "uri": str(block.uri), "name": block.name}
        case _:
            return block.model_dump(mode="json", by_alias=True, exclude_none=True)


def result_value(result: Any) -> Any:
    """What a kernel call returns: structured content, else the content blocks (one block
    unwrapped). A result flagged as an error raises."""
    if result.is_error:
        text = "\n".join(b.text for b in result.content if b.type == "text")
        raise McpToolError(text or "the tool reported an error")
    if result.structured_content is not None:
        return result.structured_content
    values = [_block_value(b) for b in result.content]
    if not values:
        return None
    return values[0] if len(values) == 1 else values
