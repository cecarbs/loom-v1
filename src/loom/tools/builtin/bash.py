from __future__ import annotations

import asyncio
import codecs

from pydantic import BaseModel, Field

from loom.broker.approval import Access
from loom.core.events import ToolCallOutput
from loom.tools.base import ToolContext, ToolResult

MAX_CAPTURE_CHARS = 5_000_000  # beyond this, output is dropped (the budget keeps far less)
DRAIN_TIMEOUT_S = 1.0


class BashArgs(BaseModel):
    command: str = Field(description="Shell command, run with /bin/bash -lc")
    timeout_s: int = Field(120, ge=1, le=600, description="Seconds before the command is killed")


class BashTool:
    name = "bash"
    description = (
        "Run a shell command in the workspace (fresh shell each call: cd and variables don't "
        "persist; background processes are killed when the command exits). Returns combined "
        "stdout/stderr and the exit code."
    )
    input_model = BashArgs

    def access(self, args: BashArgs, ctx: ToolContext) -> Access:
        return Access("exec", args.command)

    async def run(self, args: BashArgs, ctx: ToolContext) -> ToolResult:
        if ctx.sandbox is None:
            return ToolResult("bash is not available: no sandbox backend.", True)
        proc = await ctx.sandbox.spawn(
            ["/bin/bash", "-lc", args.command],
            cwd=ctx.workspace,
            env=ctx.sandbox_env,
            purpose="bash",
            workspace_writable=ctx.workspace_writable,
        )
        chunks: list[str] = []
        size = 0

        async def pump(stream: asyncio.StreamReader) -> None:
            nonlocal size
            decoder = codecs.getincrementaldecoder("utf-8")("replace")
            while True:
                data = await stream.read(65536)
                text = decoder.decode(data, final=not data)
                if text:
                    if size < MAX_CAPTURE_CHARS:
                        chunks.append(text)
                    size += len(text)
                    await ctx.bus.emit(ToolCallOutput(ctx.call_id, text))
                if not data:
                    return

        readers = [asyncio.create_task(pump(proc.stdout)), asyncio.create_task(pump(proc.stderr))]
        timed_out = False
        try:
            try:
                code = await asyncio.wait_for(proc.wait(), args.timeout_s)
            except TimeoutError:
                timed_out = True
                proc.kill()
                code = await proc.wait()
            proc.kill()  # leftover background children would hold the pipes open
            await asyncio.wait(readers, timeout=DRAIN_TIMEOUT_S)
        finally:  # also on cancel: take down the whole process group
            proc.kill()
            for reader in readers:
                reader.cancel()

        output = "".join(chunks)
        if size > MAX_CAPTURE_CHARS:
            output += f"\n[… {size - MAX_CAPTURE_CHARS:,} more chars not captured]"
        if output and not output.endswith("\n"):
            output += "\n"
        if timed_out:
            status = f"timed out after {args.timeout_s}s (process group killed)"
        else:
            status = f"exit code: {code}"
        return ToolResult(
            output + status, is_error=timed_out or code != 0, meta={"exit_code": code}
        )
