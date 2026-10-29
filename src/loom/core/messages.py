"""Canonical, provider-neutral message model. Adapters convert at the edge. See docs/02."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["user", "assistant"]


@dataclass(frozen=True)
class TextBlock:
    text: str


@dataclass(frozen=True)
class ToolUseBlock:
    id: str
    name: str
    input: dict[str, Any]


@dataclass(frozen=True)
class ImageBlock:
    media_type: str
    data_b64: str


@dataclass(frozen=True)
class ToolResultBlock:
    tool_use_id: str
    content: str | list[TextBlock | ImageBlock]
    is_error: bool = False


@dataclass(frozen=True)
class ThinkingBlock:
    text: str
    provider: str
    opaque: dict[str, Any]  # e.g. the raw Anthropic block, replayed verbatim


Block = TextBlock | ToolUseBlock | ToolResultBlock | ImageBlock | ThinkingBlock


@dataclass
class Message:
    role: Role
    content: list[Block]
    meta: dict[str, Any] = field(default_factory=dict)  # model id, usage, stop reason, …

    def tool_uses(self) -> list[ToolUseBlock]:
        return [b for b in self.content if isinstance(b, ToolUseBlock)]

    def text(self) -> str:
        return "".join(b.text for b in self.content if isinstance(b, TextBlock))


INTERRUPTED = "Interrupted by user"


def interrupted_result(tool_use_id: str) -> ToolResultBlock:
    return ToolResultBlock(tool_use_id, INTERRUPTED, is_error=True)


# --- JSON (session log) ------------------------------------------------------


def block_to_dict(block: Block) -> dict[str, Any]:
    match block:
        case TextBlock(text):
            return {"type": "text", "text": text}
        case ToolUseBlock(id_, name, input_):
            return {"type": "tool_use", "id": id_, "name": name, "input": input_}
        case ToolResultBlock(tool_use_id, content, is_error):
            body = content if isinstance(content, str) else [block_to_dict(b) for b in content]
            return {
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": body,
                "is_error": is_error,
            }
        case ImageBlock(media_type, data_b64):
            return {"type": "image", "media_type": media_type, "data_b64": data_b64}
        case ThinkingBlock(text, provider, opaque):
            return {"type": "thinking", "text": text, "provider": provider, "opaque": opaque}


def block_from_dict(d: dict[str, Any]) -> Block:
    match d["type"]:
        case "text":
            return TextBlock(d["text"])
        case "tool_use":
            return ToolUseBlock(d["id"], d["name"], d["input"])
        case "tool_result":
            content = d["content"]
            if not isinstance(content, str):
                content = [block_from_dict(b) for b in content]
            return ToolResultBlock(d["tool_use_id"], content, d.get("is_error", False))  # type: ignore[arg-type]
        case "image":
            return ImageBlock(d["media_type"], d["data_b64"])
        case "thinking":
            return ThinkingBlock(d["text"], d["provider"], d["opaque"])
        case other:
            raise ValueError(f"unknown block type {other!r}")


def message_to_dict(msg: Message) -> dict[str, Any]:
    return {
        "role": msg.role,
        "content": [block_to_dict(b) for b in msg.content],
        "meta": msg.meta,
    }


def message_from_dict(d: dict[str, Any]) -> Message:
    return Message(d["role"], [block_from_dict(b) for b in d["content"]], d.get("meta", {}))
