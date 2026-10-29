"""System prompt assembly and request building (docs/02 "System prompt").

The system prompt is rendered once per session and stays byte-stable for prompt caching.
"""

from __future__ import annotations

import platform
from collections.abc import Sequence
from datetime import date
from pathlib import Path

from loom.core.messages import Message
from loom.providers.base import ModelRequest, ToolSchema
from loom.providers.profiles import ModelProfile

_TEMPLATE = (Path(__file__).parent / "prompts" / "system.md").read_text(encoding="utf-8")

# Guidance lines, included only for tools that exist in this session.
TOOL_GUIDANCE: dict[str, str] = {
    "execute": (
        "- Use `execute` for anything with more than ~2 steps, loops, data processing, or "
        "external systems. Variables persist in the kernel; reuse them instead of re-fetching. "
        "Aggregate in the source system and in code; print summaries, never raw bulk data."
    ),
    "search_tools": "- Use `search_tools` to find kernel functions before calling them.",
    "read": "- Use `read` to view files. Read a file before editing or overwriting it.",
    "edit": (
        "- Use `edit` for targeted changes; `old_string` must match exactly once, so include "
        "surrounding lines."
    ),
    "write": "- Use `write` to create files or fully rewrite small ones.",
    "bash": "- Use `bash` for shell commands (fresh shell each call).",
}


def render_system_prompt(workspace: Path, tool_names: Sequence[str], today: date) -> str:
    guidance = "\n".join(TOOL_GUIDANCE[n] for n in TOOL_GUIDANCE if n in tool_names)
    os_name = (
        f"macOS {platform.mac_ver()[0]}" if platform.system() == "Darwin" else platform.system()
    )
    return _TEMPLATE.format(
        os=os_name, date=today.isoformat(), workspace=workspace, tool_guidance=guidance
    ).strip()


class ContextBuilder:
    def __init__(
        self,
        profile: ModelProfile,
        workspace: Path,
        tool_names: Sequence[str],
        today: date | None = None,
    ) -> None:
        self.profile = profile
        self.system = render_system_prompt(workspace, tool_names, today or date.today())

    def build_request(self, history: list[Message], tools: list[ToolSchema]) -> ModelRequest:
        return ModelRequest(
            model=self.profile.model,
            system=self.system,
            messages=list(history),
            tools=tools,
            max_tokens=self.profile.max_tokens,
            thinking=self.profile.thinking,
        )
