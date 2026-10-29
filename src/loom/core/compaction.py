"""Compaction helpers (docs/07 "Context management"): where to cut, what to ask, and the
summary message. AgentLoop.compact runs the procedure.

The cut is the only history rewrite loom performs: the prefix before it is replaced by one
user message, and everything from the cut on stays byte-identical (thinking blocks remain
valid). A cut never falls inside a tool-use/result pair.
"""

from __future__ import annotations

from collections.abc import Sequence

from loom.core.messages import Message, TextBlock, ToolResultBlock
from loom.kernel.protocol import VarInfo
from loom.session.store import is_turn_start

SUMMARY_PROMPT = """\
Summarize the conversation so far. The summary replaces the earlier messages, so it must \
hold everything needed to continue the work:

- Goal: what the user wants, including constraints and preferences they stated.
- Decisions made, and why.
- Files read, created or changed (paths), and what changed.
- Findings: key facts, numbers, errors, and where they came from.
- Kernel variables that matter and what they hold.
- Open questions.
- Next steps.

Be concise and factual; use the headings above with short bullet points. Reply with the \
summary only and do not call any tools."""

SUMMARY_PREFACE = (
    "[This session was compacted: the earlier conversation is replaced by the summary below. "
    "The most recent messages follow it unchanged.]"
)

MAX_INVENTORY_VARS = 50


def summary_request(prefix: Sequence[Message], instructions: str = "") -> list[Message]:
    prompt = SUMMARY_PROMPT
    if instructions:
        prompt += f"\n\nAdditional instructions from the user: {instructions}"
    return [*prefix, Message("user", [TextBlock(prompt)])]


def format_inventory(variables: Sequence[VarInfo]) -> str:
    if not variables:
        return ""
    shown = []
    for var in variables[:MAX_INVENTORY_VARS]:
        detail = f"{var.type}, {var.summary}" if var.summary else var.type
        shown.append(f"{var.name} ({detail})")
    more = len(variables) - len(shown)
    tail = f", … and {more} more" if more > 0 else ""
    return "Variables available in the kernel: " + ", ".join(shown) + tail


def summary_message(summary: str, variables: Sequence[VarInfo]) -> Message:
    body = summary.strip()
    if inventory := format_inventory(variables):
        body += "\n\n" + inventory
    text = f"{SUMMARY_PREFACE}\n<summary>\n{body}\n</summary>"
    return Message("user", [TextBlock(text)], {"compaction": True})


def cut_point(history: Sequence[Message], keep_turns: int, *, mid_turn: bool) -> int | None:
    """Index of the first kept message, or None if there's nothing worth summarizing.

    Normally the last `keep_turns` user turns are kept. When they alone are too much
    (`mid_turn`, used by auto-compaction inside a long turn), cut before the latest
    assistant message instead: the kept tail is then that message and its tool results.
    """
    starts = [i for i, m in enumerate(history) if is_turn_start(m)]
    if len(starts) > keep_turns:
        cut = starts[-keep_turns] if keep_turns else len(history)
        if _summarizable(history, cut):
            return cut
    if mid_turn:
        for i in range(len(history) - 1, 0, -1):
            if history[i].role == "assistant":
                return i if _summarizable(history, i) else None
    return None


def _summarizable(history: Sequence[Message], cut: int) -> bool:
    """Something besides an earlier summary lies before the cut, and the cut doesn't
    separate tool results from their tool uses."""
    if cut < 1 or (cut == 1 and history[0].meta.get("compaction")):
        return False
    return cut == len(history) or not any(
        isinstance(b, ToolResultBlock) for b in history[cut].content
    )
