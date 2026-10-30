"""System prompt assembly and request building (docs/02 "System prompt").

The system prompt is rendered once per session and stays byte-stable for prompt caching.
"""

from __future__ import annotations

import platform
from collections.abc import Sequence
from dataclasses import dataclass, field
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


CONTEXT_FILE_NAMES = ("AGENTS.md", "CLAUDE.md")
SKILLS_INTRO = (
    "Skills are instructions for specific tasks. When a task matches a skill's description, "
    "read its SKILL.md with `read` before starting."
)


@dataclass(frozen=True)
class PromptExtras:
    """What is appended to the system prompt at session start (docs/02 "System prompt")."""

    context_files: str = ""  # rendered "# Context files" section, may be empty
    skill_lines: Sequence[str] = ()
    notices: list[str] = field(default_factory=list)


def context_file_paths(workspace: Path, home: Path) -> list[Path]:
    """`<home>/AGENTS.md`, then AGENTS.md and CLAUDE.md from the git root down to the
    workspace (just the workspace outside a git repo). `home` is loom's home (~/.loom)."""
    workspace = workspace.resolve()
    root = next((d for d in (workspace, *workspace.parents) if (d / ".git").exists()), workspace)
    dirs = [d for d in (workspace, *workspace.parents) if d.is_relative_to(root)][::-1]
    candidates = [home / "AGENTS.md", *(d / n for d in dirs for n in CONTEXT_FILE_NAMES)]
    seen: set[Path] = set()
    paths = []
    for path in candidates:
        if not path.is_file():
            continue
        resolved = path.resolve()  # CLAUDE.md is often a symlink to AGENTS.md
        if resolved not in seen:
            seen.add(resolved)
            paths.append(path)
    return paths


def load_context_files(paths: Sequence[Path], max_chars: int) -> tuple[str, list[str]]:
    """The "# Context files" section (empty if there are none) and notices."""
    notices: list[str] = []
    parts = []
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeDecodeError) as e:
            notices.append(f"Skipped context file {path}: {e}")
            continue
        if text:
            parts.append(f"## {path}\n{text}")
    body = "\n\n".join(parts)
    if not body:
        return "", notices
    if len(body) > max_chars:
        body = body[:max_chars].rstrip() + (
            f"\n[loom: context files truncated at {max_chars:,} characters]"
        )
        notices.append(
            f"Context files exceed {max_chars:,} characters and were truncated "
            "([context] max_context_file_chars)."
        )
    return "# Context files\n" + body, notices


def render_system_prompt(
    workspace: Path,
    tool_names: Sequence[str],
    today: date,
    extras: PromptExtras | None = None,
) -> str:
    guidance = "\n".join(TOOL_GUIDANCE[n] for n in TOOL_GUIDANCE if n in tool_names)
    os_name = (
        f"macOS {platform.mac_ver()[0]}" if platform.system() == "Darwin" else platform.system()
    )
    sections = [
        _TEMPLATE.format(
            os=os_name, date=today.isoformat(), workspace=workspace, tool_guidance=guidance
        ).strip()
    ]
    if extras is not None:
        if extras.context_files:
            sections.append(extras.context_files)
        if extras.skill_lines:
            sections.append("\n".join(["# Skills", SKILLS_INTRO, *extras.skill_lines]))
    return "\n\n".join(sections)


class ContextBuilder:
    def __init__(
        self,
        profile: ModelProfile,
        workspace: Path,
        tool_names: Sequence[str],
        today: date | None = None,
        extras: PromptExtras | None = None,
    ) -> None:
        self.profile = profile
        self.system = render_system_prompt(workspace, tool_names, today or date.today(), extras)

    def build_request(self, history: list[Message], tools: list[ToolSchema]) -> ModelRequest:
        return ModelRequest(
            model=self.profile.model,
            system=self.system,
            messages=list(history),
            tools=tools,
            max_tokens=self.profile.max_tokens,
            thinking=self.profile.thinking if self.profile.capabilities.thinking else None,
        )
