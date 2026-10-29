from __future__ import annotations

from pydantic import BaseModel, Field

from loom.broker.approval import Access
from loom.tools.base import ToolContext, ToolResult
from loom.tools.builtin.write import unified_diff


class EditArgs(BaseModel):
    path: str = Field(description="File path, absolute or relative to the workspace")
    old_string: str = Field(description="Exact text to replace (include enough context)")
    new_string: str = Field(description="Replacement text")
    replace_all: bool = Field(False, description="Replace every occurrence")


class EditTool:
    name = "edit"
    description = (
        "Replace exact text in a file. old_string must match exactly once (include surrounding "
        "lines to make it unique) unless replace_all is true. Read the file first."
    )
    input_model = EditArgs

    def access(self, args: EditArgs, ctx: ToolContext) -> Access:
        return Access("write", ctx.paths.display(ctx.paths.resolve(args.path)))

    async def run(self, args: EditArgs, ctx: ToolContext) -> ToolResult:
        path = ctx.paths.resolve(args.path)
        if error := ctx.paths.write_error(path):
            return ToolResult(error, True)
        shown = ctx.paths.display(path)
        if not path.is_file():
            return ToolResult(f"File not found: {shown}", True)
        if error := ctx.read_set.check(path):
            return ToolResult(f"Cannot edit {shown}: {error}", True)
        if not args.old_string:
            return ToolResult("old_string must not be empty; use write to create a file.", True)
        if args.old_string == args.new_string:
            return ToolResult("old_string and new_string are identical; nothing to change.", True)

        text = path.read_bytes().decode("utf-8")
        count = text.count(args.old_string)
        if count == 0:
            return ToolResult(f"0 matches for old_string in {shown}.", True)
        if count > 1 and not args.replace_all:
            return ToolResult(
                f"{count} matches for old_string in {shown}; add context or set replace_all.",
                True,
            )
        new = text.replace(args.old_string, args.new_string, -1 if args.replace_all else 1)
        data = new.encode("utf-8")
        path.write_bytes(data)
        ctx.read_set.record(path, data)

        replaced = count if args.replace_all else 1
        plural = "s" if replaced > 1 else ""
        return ToolResult(
            f"Edited {shown} ({replaced} replacement{plural}).",
            meta={"diff": unified_diff(text, new, shown)},
        )
