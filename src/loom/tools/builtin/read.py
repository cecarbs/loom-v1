from __future__ import annotations

import base64

from pydantic import BaseModel, Field

from loom.core.messages import ImageBlock
from loom.tools.base import ToolContext, ToolResult

MAX_LINE_CHARS = 2000
IMAGE_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


class ReadArgs(BaseModel):
    path: str = Field(description="File path, absolute or relative to the workspace")
    offset: int = Field(1, ge=1, description="1-based line to start at")
    limit: int = Field(2000, ge=1, description="Maximum number of lines to return")


class ReadTool:
    name = "read"
    description = (
        "Read a text file (cat -n style numbered lines) or an image. Use offset/limit to page "
        "through large files. Read a file before editing or overwriting it."
    )
    input_model = ReadArgs

    async def run(self, args: ReadArgs, ctx: ToolContext) -> ToolResult:
        path = ctx.paths.resolve(args.path)
        if error := ctx.paths.read_error(path):
            return ToolResult(error, True)
        if not path.exists():
            return ToolResult(f"File not found: {path}", True)
        if path.is_dir():
            return ToolResult(f"{path} is a directory, not a file.", True)

        data = path.read_bytes()
        shown = ctx.paths.display(path)
        if media_type := IMAGE_TYPES.get(path.suffix.lower()):
            if not ctx.model_accepts_images:
                return ToolResult(f"{shown} is an image; this model does not accept images.")
            image = ImageBlock(media_type, base64.b64encode(data).decode("ascii"))
            return ToolResult(f"Image {shown} ({len(data):,} bytes)", images=[image])
        if b"\0" in data[:8192]:
            return ToolResult(f"{shown} looks like a binary file; not shown.", True)

        ctx.read_set.record(path, data)
        lines = data.decode("utf-8", errors="replace").splitlines()
        if not lines:
            return ToolResult(f"{shown} is empty.")
        start = args.offset - 1
        if start >= len(lines):
            return ToolResult(
                f"{shown} has {len(lines)} lines; offset {args.offset} is past the end."
            )
        selected = lines[start : start + args.limit]
        out = [f"{n:>6}\t{_clip(line)}" for n, line in enumerate(selected, start=args.offset)]
        remaining = len(lines) - (start + len(selected))
        if remaining > 0:
            out.append(
                f"[… {remaining} more lines; continue with offset={start + len(selected) + 1}]"
            )
        return ToolResult("\n".join(out))


def _clip(line: str) -> str:
    if len(line) <= MAX_LINE_CHARS:
        return line
    return line[:MAX_LINE_CHARS] + f" [… {len(line) - MAX_LINE_CHARS} chars cut]"
