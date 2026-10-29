"""`@path` mentions: attach a file's contents to a user message (docs/06 "Input").

Attachments go through the same path policy and output budget as the `read` tool, and
count as a read, so the model can edit a mentioned file without reading it first.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from loom.core.messages import Block, TextBlock
from loom.tools.base import ToolContext
from loom.tools.budget import OutputBudget
from loom.tools.builtin.read import ReadArgs, ReadTool

MENTION = re.compile(r"(?:^|(?<=\s))@(\S+)")
_TRAILING = ".,;:!?)]}'\""


@dataclass
class Mentions:
    blocks: list[Block] = field(default_factory=list)
    attached: list[str] = field(default_factory=list)  # display paths
    notices: list[str] = field(default_factory=list)


def find_mentions(text: str) -> list[str]:
    return [m.group(1).rstrip(_TRAILING) for m in MENTION.finditer(text)]


async def expand_mentions(text: str, ctx: ToolContext, budget: OutputBudget) -> Mentions:
    """Mentions that aren't existing files (`@here`, `@alice`) are left as plain text."""
    out = Mentions()
    seen = set()
    for raw in find_mentions(text):
        if not raw:
            continue
        path = ctx.paths.resolve(raw)
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        if error := ctx.paths.read_error(path):
            out.notices.append(f"@{raw} not attached: {error}")
            continue
        result = await ReadTool().run(ReadArgs.model_validate({"path": str(path)}), ctx)
        if result.is_error:
            out.notices.append(f"@{raw} not attached: {result.text}")
            continue
        shown, _ = budget.apply(result.text)
        display = ctx.paths.display(path)
        out.blocks.append(TextBlock(f'<file path="{display}">\n{shown}\n</file>'))
        out.blocks.extend(result.images)
        out.attached.append(display)
    return out
