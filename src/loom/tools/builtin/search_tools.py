from __future__ import annotations

from pydantic import BaseModel, Field

from loom.broker.approval import Access
from loom.broker.broker import ToolBroker
from loom.tools.base import ToolContext, ToolResult


class SearchToolsArgs(BaseModel):
    query: str = Field("", description="Keywords; empty lists the installed packages")
    limit: int = Field(10, ge=1, le=50)


class SearchToolsTool:
    name = "search_tools"
    description = (
        "Find functions available inside the execute kernel (Splunk, tickets, …) by keyword. "
        "Returns signatures and summaries; call them from execute, e.g. "
        "`rows = splunk.search(...)`. Use help(pkg.func) in the kernel for full docs."
    )
    input_model = SearchToolsArgs

    def __init__(self, broker: ToolBroker) -> None:
        self.broker = broker

    def access(self, args: SearchToolsArgs, ctx: ToolContext) -> Access:
        return Access("read")

    async def run(self, args: SearchToolsArgs, ctx: ToolContext) -> ToolResult:
        return ToolResult(self.broker.search(args.query, args.limit))
