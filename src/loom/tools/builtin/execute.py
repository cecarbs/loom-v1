from __future__ import annotations

from pydantic import BaseModel, Field

from loom.broker.approval import Access
from loom.core.events import ToolCallOutput
from loom.kernel.runner import compose_text
from loom.tools.base import ToolContext, ToolResult

RESTARTED = "[kernel restarted — previous variables are gone]"


class ExecuteArgs(BaseModel):
    code: str = Field(description="Python code; the value of a final expression is shown")
    timeout_s: int | None = Field(
        None, ge=1, le=1800, description="Seconds before the kernel is killed (default 300)"
    )


class ExecuteTool:
    name = "execute"
    description = (
        "Run Python in a persistent kernel: variables, imports and functions survive between "
        "calls. Prints and the repr of a final expression are returned; top-level await works. "
        "Keep bulk data in variables and print small summaries."
    )
    input_model = ExecuteArgs

    def __init__(self, default_timeout_s: int = 300) -> None:
        self.default_timeout_s = default_timeout_s

    def access(self, args: ExecuteArgs, ctx: ToolContext) -> Access:
        return Access("exec", args.code)

    async def run(self, args: ExecuteArgs, ctx: ToolContext) -> ToolResult:
        if ctx.kernel is None:
            return ToolResult("execute is not available: no kernel.", True)
        timeout_s = args.timeout_s or self.default_timeout_s

        async def stream(text: str) -> None:
            await ctx.bus.emit(ToolCallOutput(ctx.call_id, text))

        r = await ctx.kernel.execute(args.code, timeout_s=timeout_s, on_output=stream)
        text = compose_text(r.output, r.value_repr, r.error.traceback if r.error else None)
        if r.status == "timeout":
            text = compose_text(
                text,
                None,
                f"Timed out after {timeout_s}s; the kernel was killed and its variables are "
                "gone. Use a larger timeout_s or smaller steps.",
            )
        elif r.status == "died":
            text = compose_text(text, None, "The kernel restarts on the next execute.")
        if r.restarted:
            text = f"{RESTARTED}\n{text}"
        meta: dict[str, object] = {"status": r.status}
        if r.out_ref is not None:
            meta["kernel_ref"] = r.out_ref
        return ToolResult(text or "(no output)", is_error=r.status != "ok", meta=meta)
