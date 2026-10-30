"""Adapter tests: recorded SSE streams through a mocked HTTP transport, plus request shaping."""

import json
from dataclasses import replace
from pathlib import Path

import httpx2
import pytest

from loom.broker.secrets import SecretStore
from loom.core.messages import (
    ImageBlock,
    Message,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from loom.providers import base as pv
from loom.providers.anthropic import AnthropicProvider
from loom.providers.anthropic import build_params as anthropic_params
from loom.providers.openai_compat import OpenAICompatProvider
from loom.providers.openai_compat import build_params as openai_params
from loom.providers.profiles import (
    BUILTIN_PROFILES,
    PROVIDER_CAPABILITIES,
    ModelCapabilities,
    ModelProfile,
)

FIXTURES = Path(__file__).parent / "fixtures"
SONNET = BUILTIN_PROFILES["sonnet"]
LOCAL = ModelProfile(
    alias="local",
    provider="openai_compat",
    model="qwen3:8b",
    base_url="http://localhost:11434/v1",
    capabilities=PROVIDER_CAPABILITIES["openai_compat"],
    extra_body={"options": {"num_ctx": 32768}},
)
SCHEMA = pv.ToolSchema("read", "Read a file", {"type": "object", "properties": {}})

CONVERSATION = [
    Message("user", [TextBlock("look at a.txt"), ImageBlock("image/png", "AAAA")]),
    Message(
        "assistant",
        [
            ThinkingBlock(
                "hmm", "anthropic", {"type": "thinking", "thinking": "hmm", "signature": "s1"}
            ),
            ThinkingBlock("other", "openai_responses", {"id": "r1"}),
            TextBlock("Reading."),
            ToolUseBlock("t1", "read", {"path": "a.txt"}),
        ],
    ),
    Message(
        "user",
        [ToolResultBlock("t1", [TextBlock("contents"), ImageBlock("image/png", "BBBB")])],
    ),
]


def request(**kw) -> pv.ModelRequest:
    defaults = dict(model="m", system="SYS", messages=CONVERSATION, tools=[SCHEMA], max_tokens=1000)
    return pv.ModelRequest(**{**defaults, **kw})  # type: ignore[arg-type]


class Recorder:
    """Mock transport handler: records the request body and replies with a fixture."""

    def __init__(self, status: int = 200, body: bytes = b"", headers: dict | None = None):
        self.status, self.body, self.headers = status, body, headers or {}
        self.requests: list[dict] = []

    def __call__(self, req: httpx2.Request) -> httpx2.Response:
        self.requests.append(json.loads(req.content))
        headers = {"content-type": "text/event-stream", **self.headers}
        if self.status != 200:
            headers["content-type"] = "application/json"
        return httpx2.Response(self.status, headers=headers, content=self.body)


def client(handler: Recorder) -> httpx2.AsyncClient:
    return httpx2.AsyncClient(transport=httpx2.MockTransport(handler))


async def collect(provider: pv.Provider, req: pv.ModelRequest) -> list[pv.ProviderEvent]:
    return [e async for e in provider.stream(req)]


# --- Anthropic -------------------------------------------------------------------


def test_anthropic_params_cache_thinking_and_replay() -> None:
    profile = replace(SONNET, thinking=pv.ThinkingConfig(effort="medium"))
    params = anthropic_params(request(temperature=0.3, thinking=profile.thinking), profile)

    assert params["system"] == [
        {"type": "text", "text": "SYS", "cache_control": {"type": "ephemeral"}}
    ]
    assert params["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert params["output_config"] == {"effort": "medium"}
    assert "temperature" not in params
    assert params["tools"] == [
        {"name": "read", "description": "Read a file", "input_schema": SCHEMA.input_schema}
    ]
    user, assistant, results = params["messages"]
    assert user["content"][1]["source"] == {
        "type": "base64",
        "media_type": "image/png",
        "data": "AAAA",
    }
    # Own thinking replayed verbatim; another provider's dropped.
    assert assistant["content"][0] == {"type": "thinking", "thinking": "hmm", "signature": "s1"}
    assert [b["type"] for b in assistant["content"]] == ["thinking", "text", "tool_use"]
    tool_result = results["content"][-1]
    assert tool_result["cache_control"] == {"type": "ephemeral"}
    assert [b["type"] for b in tool_result["content"]] == ["text", "image"]


def test_anthropic_params_without_images_or_cache() -> None:
    caps = replace(SONNET.capabilities, images=False, prompt_cache=False)
    params = anthropic_params(request(), replace(SONNET, capabilities=caps))
    assert "cache_control" not in params["system"][0]
    assert "image omitted" in params["messages"][0]["content"][1]["text"]
    assert "image omitted" in params["messages"][2]["content"][0]["content"][1]["text"]
    assert "thinking" not in params


def test_anthropic_budget_tokens_for_older_models() -> None:
    params = anthropic_params(request(thinking=pv.ThinkingConfig(budget_tokens=4000)), SONNET)
    assert params["thinking"] == {"type": "enabled", "budget_tokens": 4000}
    assert "output_config" not in params


async def test_anthropic_stream_events_and_final_message() -> None:
    handler = Recorder(body=(FIXTURES / "anthropic_tool_use.sse").read_bytes())
    provider = AnthropicProvider(
        replace(SONNET, api_key_secret="k"),
        SecretStore({"LOOM_SECRET_K": "sk-test"}),
        http_client=client(handler),
    )
    events = await collect(provider, request(model="claude-sonnet-5-5"))

    assert [type(e) for e in events] == [
        pv.ThinkingDelta,
        pv.TextDelta,
        pv.TextDelta,
        pv.ToolUseStart,
        pv.ToolUseInputDelta,
        pv.ToolUseInputDelta,
        pv.ToolUseEnd,
        pv.MessageEnd,
    ]
    end = events[-1]
    assert isinstance(end, pv.MessageEnd) and end.stop_reason == "tool_use"
    assert end.usage.cache_read_tokens == 3000 and end.usage.output_tokens == 42
    thinking, text, tool_use = end.final.content
    assert isinstance(thinking, ThinkingBlock)
    assert thinking.opaque == {
        "type": "thinking",
        "thinking": "Need to read the file.",
        "signature": "sig-abc",
    }
    assert text == TextBlock("Reading it now.")
    assert tool_use == ToolUseBlock("toolu_1", "read", {"path": "a.txt"})
    assert handler.requests[0]["model"] == "claude-sonnet-5-5"
    assert handler.requests[0]["stream"] is True


@pytest.mark.parametrize(
    ("status", "headers", "kind", "retry_after"),
    [
        (429, {"retry-after": "7"}, "rate_limit", 7.0),
        (529, {}, "overloaded", None),
        (401, {}, "auth", None),
    ],
)
async def test_anthropic_error_mapping(status, headers, kind, retry_after) -> None:
    body = json.dumps({"type": "error", "error": {"type": "x", "message": "nope"}}).encode()
    handler = Recorder(status=status, body=body, headers=headers)
    provider = AnthropicProvider(
        replace(SONNET, api_key_secret="k"),
        SecretStore({"LOOM_SECRET_K": "sk"}),
        http_client=client(handler),
    )
    [event] = await collect(provider, request())
    assert isinstance(event, pv.ProviderError)
    assert (event.kind, event.retry_after) == (kind, retry_after)


async def test_anthropic_context_too_long() -> None:
    body = json.dumps(
        {
            "type": "error",
            "error": {
                "type": "invalid_request_error",
                "message": "prompt is too long: 1200000 tokens > 1000000 maximum",
            },
        }
    ).encode()
    provider = AnthropicProvider(
        replace(SONNET, api_key_secret="k"),
        SecretStore({"LOOM_SECRET_K": "sk"}),
        http_client=client(Recorder(status=400, body=body)),
    )
    [event] = await collect(provider, request())
    assert isinstance(event, pv.ProviderError) and event.kind == "context_too_long"


# --- OpenAI-compatible -----------------------------------------------------------------


def test_openai_message_conversion() -> None:
    params = openai_params(request(), LOCAL)
    msgs = params["messages"]
    assert msgs[0] == {"role": "system", "content": "SYS"}
    assert msgs[1]["content"] == "look at a.txt[image omitted: model does not accept images]"
    assert msgs[2] == {
        "role": "assistant",
        "content": "Reading.",
        "tool_calls": [
            {
                "id": "t1",
                "type": "function",
                "function": {"name": "read", "arguments": '{"path": "a.txt"}'},
            }
        ],
    }
    assert msgs[3] == {"role": "tool", "tool_call_id": "t1", "content": "contents"}
    assert msgs[4] == {"role": "user", "content": "[image omitted: model does not accept images]"}
    assert params["tools"][0]["function"]["name"] == "read"
    assert params["stream_options"] == {"include_usage": True}
    assert "parallel_tool_calls" not in params


def test_openai_images_system_role_and_parallel_flag() -> None:
    caps = ModelCapabilities(images=True, parallel_tool_calls=False, system_role="user")
    params = openai_params(request(), replace(LOCAL, capabilities=caps))
    msgs = params["messages"]
    assert msgs[0]["role"] == "user"
    assert msgs[0]["content"][0] == {"type": "text", "text": "SYS\n\n"}
    assert msgs[0]["content"][2]["image_url"]["url"] == "data:image/png;base64,AAAA"
    assert msgs[-1]["content"][0]["image_url"]["url"] == "data:image/png;base64,BBBB"
    assert params["parallel_tool_calls"] is False


async def test_openai_stream_accumulates_tool_calls() -> None:
    handler = Recorder(body=(FIXTURES / "openai_tool_calls.sse").read_bytes())
    provider = OpenAICompatProvider(LOCAL, SecretStore({}), http_client=client(handler))
    events = await collect(provider, request(model="qwen3:8b"))

    assert isinstance(events[0], pv.ThinkingDelta) and events[0].text == "thinking hard"
    assert pv.TextDelta("Reading.") in events
    assert pv.ToolUseStart("call_1", "read") in events
    end = events[-1]
    assert isinstance(end, pv.MessageEnd) and end.stop_reason == "tool_use"
    text, good, bad = end.final.content
    assert text == TextBlock("Reading.")
    assert good == ToolUseBlock("call_1", "read", {"path": "a.txt"})
    assert isinstance(bad, ToolUseBlock) and bad.input == {pv.INVALID_JSON_KEY: "{bad json"}
    assert (end.usage.input_tokens, end.usage.cache_read_tokens, end.usage.output_tokens) == (
        20,
        100,
        30,
    )
    body = handler.requests[0]
    assert body["options"] == {"num_ctx": 32768}  # extra_body merged into the request
    assert body["model"] == "qwen3:8b"


async def test_openai_connection_error_names_server() -> None:
    def refuse(req: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("connection refused", request=req)

    provider = OpenAICompatProvider(
        LOCAL,
        SecretStore({}),
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(refuse)),
    )
    [event] = await collect(provider, request())
    assert isinstance(event, pv.ProviderError) and event.kind == "other"
    assert "localhost:11434" in event.message


@pytest.mark.parametrize(
    ("secret", "hint"),
    [("anthropic_api_key", "LOOM_SECRET_ANTHROPIC_API_KEY"), (None, "api_key_secret")],
)
async def test_anthropic_without_credentials(secret, hint, tmp_path, monkeypatch) -> None:
    """No key from the store, the env or an SDK credentials profile: an auth error, not the
    SDK's TypeError (which used to crash the TUI)."""
    import os

    for name in [n for n in os.environ if n.startswith("ANTHROPIC_")]:
        monkeypatch.delenv(name)
    monkeypatch.setenv("HOME", str(tmp_path))  # no SDK credentials profile either
    handler = Recorder(body=(FIXTURES / "anthropic_tool_use.sse").read_bytes())
    provider = AnthropicProvider(
        replace(SONNET, api_key_secret=secret), SecretStore({}), http_client=client(handler)
    )
    [event] = await collect(provider, request())
    assert isinstance(event, pv.ProviderError) and event.kind == "auth"
    assert event.message.startswith("no Anthropic API key") and hint in event.message
    assert handler.requests == []


async def test_missing_credentials_end_the_turn_with_an_error(tmp_path, monkeypatch) -> None:
    import os

    from loom.app import build_agent
    from loom.config import LoomConfig
    from loom.core import events as ev

    for name in [n for n in os.environ if n.startswith("ANTHROPIC_")]:
        monkeypatch.delenv(name)
    monkeypatch.setenv("HOME", str(tmp_path))
    handler = Recorder(body=b"")
    provider = AnthropicProvider(SONNET, SecretStore({}), http_client=client(handler))
    (tmp_path / "ws").mkdir()
    bus = ev.EventBus()
    seen: list[ev.AgentEvent] = []

    async def record(event: ev.AgentEvent) -> None:
        seen.append(event)

    bus.subscribe(record)
    agent = build_agent(
        tmp_path / "ws", LoomConfig(), bus, provider=provider, home=tmp_path / "home"
    )
    await agent.run_turn("hi")  # returns instead of raising
    errors = [r for r in agent.log.read() if r["kind"] == "error"]
    assert len(errors) == 1 and "no Anthropic API key" in str(errors[0])
    assert any(isinstance(e, ev.ErrorEvent) for e in seen)
