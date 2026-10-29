"""`loom models test`: probe which capabilities a configured model actually has (docs/02).

Each probe forces the capability on for one small request, so the result says what the
server can do, not what the config claims. The caller prints the results.
"""

from __future__ import annotations

import struct
import time
import zlib
from base64 import b64encode
from collections.abc import Callable
from dataclasses import dataclass, replace

from loom.core.messages import ImageBlock, Message, TextBlock
from loom.providers import base as pv
from loom.providers.profiles import ModelProfile
from loom.providers.text_tools import TextToolCallProvider

SYSTEM = "You are a capability probe. Follow the instruction exactly and keep replies short."
ADD_TOOL = pv.ToolSchema(
    "add",
    "Add two integers and return the sum.",
    {
        "type": "object",
        "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
        "required": ["a", "b"],
    },
)

ProviderFactory = Callable[[ModelProfile], pv.Provider]


@dataclass(frozen=True)
class ProbeResult:
    name: str
    ok: bool
    detail: str
    seconds: float = 0.0


@dataclass
class _Reply:
    text: str = ""
    thinking: bool = False
    tool_names: tuple[str, ...] = ()
    error: pv.ProviderError | None = None


async def probe_model(profile: ModelProfile, make_provider: ProviderFactory) -> list[ProbeResult]:
    """`make_provider` must build the adapter without the text-tool wrapper."""
    results: list[ProbeResult] = []

    reply, secs = await _ask(profile, make_provider, "Reply with exactly the word: pong")
    if reply.error is not None:
        return [ProbeResult("reply", False, _error(reply.error), secs)]
    results.append(ProbeResult("reply", "pong" in reply.text.lower(), _clip(reply.text), secs))
    results.append(
        ProbeResult(
            "reasoning",
            reply.thinking,
            "reasoning text streamed" if reply.thinking else "no reasoning text seen",
        )
    )

    prompt = "Use the add tool to add 2 and 3. Do not answer directly."
    native = replace(profile, capabilities=replace(profile.capabilities, tools=True))
    reply, secs = await _ask(native, make_provider, prompt, tools=True)
    native_ok = "add" in reply.tool_names
    results.append(ProbeResult("tools", native_ok, _tool_detail(reply), secs))
    if not native_ok:
        reply, secs = await _ask(
            profile, lambda p: TextToolCallProvider(make_provider(p)), prompt, tools=True
        )
        results.append(
            ProbeResult("text tool calls", "add" in reply.tool_names, _tool_detail(reply), secs)
        )

    with_images = replace(profile, capabilities=replace(profile.capabilities, images=True))
    image = ImageBlock("image/png", red_png())
    reply, secs = await _ask(
        with_images,
        make_provider,
        "What color is this image? Answer with one word.",
        image=image,
    )
    detail = _error(reply.error) if reply.error else _clip(reply.text)
    results.append(ProbeResult("images", "red" in reply.text.lower(), detail, secs))
    return results


def suggestions(profile: ModelProfile, results: list[ProbeResult]) -> list[str]:
    """Config lines for capabilities whose probe disagrees with the profile. Seeing reasoning
    text doesn't show that the server accepts a reasoning setting, so `thinking` isn't
    suggested."""
    found = {r.name: r.ok for r in results}
    caps = profile.capabilities
    lines: list[str] = []
    if "tools" in found and found["tools"] != caps.tools:
        lines.append(f"tools = {_toml(found['tools'])}")
    if "images" in found and found["images"] != caps.images:
        lines.append(f"images = {_toml(found['images'])}")
    if not lines:
        return []
    return [f"[models.{profile.alias}.capabilities]", *lines]


async def _ask(
    profile: ModelProfile,
    make_provider: ProviderFactory,
    prompt: str,
    *,
    tools: bool = False,
    image: ImageBlock | None = None,
) -> tuple[_Reply, float]:
    provider = make_provider(profile)
    content: list = [TextBlock(prompt)] + ([image] if image else [])
    request = pv.ModelRequest(
        model=profile.model,
        system=SYSTEM,
        messages=[Message("user", content)],
        tools=[ADD_TOOL] if tools else [],
        max_tokens=profile.max_tokens,
        thinking=profile.thinking if profile.capabilities.thinking else None,
        cache=False,
    )
    reply = _Reply()
    text: list[str] = []
    names: list[str] = []
    start = time.monotonic()
    async for event in provider.stream(request):
        match event:
            case pv.TextDelta(chunk):
                text.append(chunk)
            case pv.ThinkingDelta(chunk):
                reply.thinking = reply.thinking or bool(chunk.strip())
            case pv.ToolUseStart(_, name):
                names.append(name)
            case pv.ProviderError():
                reply.error = event
    reply.text = "".join(text).strip()
    reply.tool_names = tuple(names)
    return reply, time.monotonic() - start


def red_png(size: int = 16) -> str:
    """A solid red PNG, base64-encoded."""

    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    header = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)  # 8-bit RGB
    rows = b"".join(b"\x00" + b"\xff\x00\x00" * size for _ in range(size))
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b"")
    )
    return b64encode(png).decode()


def _tool_detail(reply: _Reply) -> str:
    if reply.error is not None:
        return _error(reply.error)
    if reply.tool_names:
        return "called " + ", ".join(reply.tool_names)
    return f"no tool call; replied {_clip(reply.text)}"


def _error(error: pv.ProviderError) -> str:
    return f"error ({error.kind}): {_clip(error.message)}"


def _clip(text: str, limit: int = 60) -> str:
    text = " ".join(text.split())
    return repr(text if len(text) <= limit else text[: limit - 1] + "…")


def _toml(value: bool) -> str:
    return "true" if value else "false"
