from __future__ import annotations

import difflib

from pydantic import BaseModel, Field

from loom.broker.approval import Access
from loom.tools.base import ToolContext, ToolResult


class WriteArgs(BaseModel):
    path: str = Field(description="File path, absolute or relative to the workspace")
    content: str = Field(description="Complete new file content")


class WriteTool:
    name = "write"
    description = (
        "Create a file, or overwrite one completely. Overwriting requires that you read the "
        "file first. Prefer edit for changes to existing files."
    )
    input_model = WriteArgs

    def access(self, args: WriteArgs, ctx: ToolContext) -> Access:
        return Access("write", ctx.paths.display(ctx.paths.resolve(args.path)))

    async def run(self, args: WriteArgs, ctx: ToolContext) -> ToolResult:
        path = ctx.paths.resolve(args.path)
        if error := ctx.paths.write_error(path):
            return ToolResult(error, True)
        shown = ctx.paths.display(path)
        old: str | None = None
        if path.exists():
            if path.is_dir():
                return ToolResult(f"{shown} is a directory.", True)
            if error := ctx.read_set.check(path):
                return ToolResult(f"Cannot overwrite {shown}: {error}", True)
            old = path.read_bytes().decode("utf-8", errors="replace")

        data = args.content.encode("utf-8")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        ctx.read_set.record(path, data)

        lines = args.content.count("\n") + (0 if args.content.endswith("\n") else 1)
        verb = "Created" if old is None else "Overwrote"
        return ToolResult(
            f"{verb} {shown} ({lines} lines).",
            meta={"diff": unified_diff(old or "", args.content, shown)},
        )


def unified_diff(old: str, new: str, name: str) -> str:
    return "".join(
        difflib.unified_diff(
            old.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile=f"a/{name}",
            tofile=f"b/{name}",
        )
    )
