"""M7: golden request conversion, capability shaping, Responses adapter, text tool calls,
TLS options, model switching, `loom models`."""

import json
import os
import ssl
from dataclasses import replace
from pathlib import Path

import httpx2
import pytest

from loom.app import build_agent
from loom.broker.secrets import SecretStore
from loom.config import LoomConfig
from loom.core.events import EventBus
from loom.core.messages import (
    ImageBlock,
    Message,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from loom.providers import base as pv
from loom.providers.anthropic import build_params as anthropic_params
from loom.providers.fake import FakeProvider
from loom.providers.openai_compat import OpenAICompatProvider
from loom.providers.openai_compat import build_params as compat_params
from loom.providers.openai_responses import OpenAIResponsesProvider
from loom.providers.openai_responses import build_params as responses_params
from loom.providers.probe import probe_model, suggestions
from loom.providers.profiles import (
    BUILTIN_PROFILES,
    PROVIDER_CAPABILITIES,
    ModelCapabilities,
    ModelProfile,
    profile_notices,
    resolve_profile,
)
from loom.providers.registry import create_provider
from loom.providers.text_tools import INVALID_NAME, TextToolCallProvider, _Parser, parse_call

FIXTURES = Path(__file__).parent / "fixtures"
GOLDEN = FIXTURES / "golden"
UPDATE = os.environ.get("LOOM_UPDATE_GOLDEN") == "1"

SONNET = replace(BUILTIN_PROFILES["sonnet"], thinking=pv.ThinkingConfig(effort="medium"))
COMPAT = ModelProfile(
    alias="compat",
    provider="openai_compat",
    model="gpt-test",
    capabilities=ModelCapabilities(images=True, thinking=True),
    thinking=pv.ThinkingConfig(effort="medium"),
)
RESPONSES = ModelProfile(
    alias="resp",
    provider="openai_responses",
    model="gpt-test",
    capabilities=ModelCapabilities(images=True, thinking=True),
    thinking=pv.ThinkingConfig(effort="medium"),
)
SCHEMA = pv.ToolSchema(
    "read", "Read a file", {"type": "object", "properties": {"path": {"type": "string"}}}
)

# The canonical conversation the golden files are built from: text, an image, thinking from
# two providers, a tool use, a tool result with an image, an error result, and a reply.
CONVERSATION = [
    Message("user", [TextBlock("look at a.txt and this"), ImageBlock("image/png", "AAAA")]),
    Message(
        "assistant",
        [
            ThinkingBlock(
                "hmm", "anthropic", {"type": "thinking", "thinking": "hmm", "signature": "sig1"}
            ),
            ThinkingBlock(
                "plan",
                "openai_responses",
                {
                    "type": "reasoning",
                    "id": "rs_1",
                    "summary": [{"type": "summary_text", "text": "plan"}],
                    "encrypted_content": "ENC",
                },
            ),
            TextBlock("Reading both."),
            ToolUseBlock("t1", "read", {"path": "a.txt"}),
            ToolUseBlock("t2", "read", {"path": "b.png"}),
        ],
    ),
    Message(
        "user",
        [
            ToolResultBlock("t1", "no such file", is_error=True),
            ToolResultBlock("t2", [TextBlock("b.png:"), ImageBlock("image/png", "BBBB")]),
        ],
    ),
    Message("assistant", [TextBlock("a.txt is missing; b.png is red.")]),
    Message("user", [TextBlock("thanks")]),
]

ADAPTERS = {
    "anthropic": (anthropic_params, SONNET),
    "openai_compat": (compat_params, COMPAT),
    "openai_responses": (responses_params, RESPONSES),
}


def request(profile: ModelProfile, messages: list[Message] = CONVERSATION) -> pv.ModelRequest:
    return pv.ModelRequest(
        model=profile.model,
        system="SYS",
        messages=messages,
        tools=[SCHEMA],
        max_tokens=1000,
        thinking=profile.thinking,
    )


def without(profile: ModelProfile, **caps) -> ModelProfile:
    return replace(profile, capabilities=replace(profile.capabilities, **caps))


# --- golden conversion -----------------------------------------------------------


@pytest.mark.parametrize("adapter", sorted(ADAPTERS))
def test_golden_request(adapter: str) -> None:
    build, profile = ADAPTERS[adapter]
    body = build(request(profile), profile)
    path = GOLDEN / f"{adapter}.json"
    if UPDATE:
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(body, indent=2) + "\n")
    assert body == json.loads(path.read_text())


@pytest.mark.parametrize("adapter", sorted(ADAPTERS))
def test_images_false_sends_no_image_content(adapter: str) -> None:
    build, profile = ADAPTERS[adapter]
    profile = without(profile, images=False)
    text = json.dumps(build(request(profile), profile))
    for marker in ("AAAA", "BBBB", "image_url", "input_image", '"image"', "base64"):
        assert marker not in text
    assert "image omitted" in text


@pytest.mark.parametrize("adapter", sorted(ADAPTERS))
def test_thinking_false_sends_no_reasoning_config(adapter: str) -> None:
    build, profile = ADAPTERS[adapter]
    profile = without(profile, thinking=False)
    body = build(request(profile), profile)
    for key in ("thinking", "output_config", "reasoning_effort", "reasoning", "include"):
        assert key not in body
    # Responses reasoning items need encrypted content plus the thinking capability.
    assert "rs_1" not in json.dumps(body)


def test_each_adapter_replays_only_its_own_thinking() -> None:
    anthropic = json.dumps(anthropic_params(request(SONNET), SONNET))
    compat = json.dumps(compat_params(request(COMPAT), COMPAT))
    responses = json.dumps(responses_params(request(RESPONSES), RESPONSES))
    assert "sig1" in anthropic and "ENC" not in anthropic
    assert "sig1" not in compat and "ENC" not in compat
    assert "ENC" in responses and "sig1" not in responses


def test_context_builder_sends_thinking_only_when_capable(tmp_path: Path) -> None:
    from loom.core.context import ContextBuilder

    no_thinking = without(SONNET, thinking=False)
    assert ContextBuilder(SONNET, tmp_path, []).build_request([], []).thinking is not None
    assert ContextBuilder(no_thinking, tmp_path, []).build_request([], []).thinking is None
    assert any("thinking" in n for n in profile_notices(no_thinking))


def test_system_role_variants() -> None:
    dev = without(RESPONSES, system_role="developer")
    body = responses_params(request(dev), dev)
    assert "instructions" not in body and body["input"][0] == {
        "role": "developer",
        "content": "SYS",
    }
    user = without(RESPONSES, system_role="user")
    body = responses_params(request(user), user)
    assert "instructions" not in body
    assert body["input"][0]["content"][0] == {"type": "input_text", "text": "SYS\n\n"}


# --- profiles --------------------------------------------------------------------


def test_builtin_profiles_and_provider_defaults() -> None:
    config = LoomConfig()
    assert resolve_profile("opus", config).model == "claude-opus-5-5"
    haiku = resolve_profile("haiku", config)
    assert haiku.model == "claude-haiku-5-5" and haiku.pricing is not None
    assert PROVIDER_CAPABILITIES["openai_responses"].images
    config = LoomConfig.model_validate(
        {"models": {"gpt": {"provider": "openai_responses", "model": "x"}}}
    )
    gpt = resolve_profile("gpt", config)
    assert gpt.capabilities.images and not gpt.capabilities.thinking


def test_create_provider_wraps_models_without_tools() -> None:
    profile = without(COMPAT, tools=False)
    assert isinstance(create_provider(profile, SecretStore({})), TextToolCallProvider)
    raw = create_provider(profile, SecretStore({}), text_tools=False)
    assert isinstance(raw, OpenAICompatProvider)
    assert isinstance(create_provider(RESPONSES, SecretStore({})), OpenAIResponsesProvider)


# --- Responses adapter -----------------------------------------------------------


class Recorder:
    def __init__(self, body: bytes, status: int = 200, content_type: str = "text/event-stream"):
        self.body, self.status, self.content_type = body, status, content_type
        self.requests: list[httpx2.Request] = []

    def __call__(self, req: httpx2.Request) -> httpx2.Response:
        self.requests.append(req)
        return httpx2.Response(
            self.status, headers={"content-type": self.content_type}, content=self.body
        )

    def body_of(self, i: int = 0) -> dict:
        return json.loads(self.requests[i].content)


def mock(handler) -> httpx2.AsyncClient:
    return httpx2.AsyncClient(transport=httpx2.MockTransport(handler))


async def collect(provider: pv.Provider, req: pv.ModelRequest) -> list[pv.ProviderEvent]:
    return [e async for e in provider.stream(req)]


async def test_responses_stream_and_stateless_replay() -> None:
    handler = Recorder((FIXTURES / "openai_responses_tool_call.sse").read_bytes())
    provider = OpenAIResponsesProvider(RESPONSES, SecretStore({}), http_client=mock(handler))
    first = [Message("user", [TextBlock("read a.txt")])]
    events = await collect(provider, request(RESPONSES, first))

    assert pv.ThinkingDelta("Plan: read the file.") in events
    assert pv.TextDelta("Reading.") in events
    assert pv.ToolUseStart("call_1", "read") in events
    assert pv.ToolUseInputDelta("call_1", '{"path": ') in events
    end = events[-1]
    assert isinstance(end, pv.MessageEnd) and end.stop_reason == "tool_use"
    thinking, text, call = end.final.content
    assert isinstance(thinking, ThinkingBlock) and thinking.text == "Plan: read the file."
    assert thinking.opaque["encrypted_content"] == "ENC123" and "status" not in thinking.opaque
    assert text == TextBlock("Reading.")
    assert call == ToolUseBlock("call_1", "read", {"path": "a.txt"})
    assert (end.usage.input_tokens, end.usage.cache_read_tokens) == (20, 100)

    body = handler.body_of()
    assert body["store"] is False and body["include"] == ["reasoning.encrypted_content"]
    assert body["reasoning"] == {"effort": "medium", "summary": "auto"}

    history = [*first, end.final, Message("user", [ToolResultBlock("call_1", "hello")])]
    await collect(provider, request(RESPONSES, history))
    replayed = handler.body_of(1)["input"]
    assert replayed[1]["type"] == "reasoning" and replayed[1]["encrypted_content"] == "ENC123"
    assert replayed[3]["type"] == "function_call" and replayed[3]["call_id"] == "call_1"
    assert replayed[4] == {"type": "function_call_output", "call_id": "call_1", "output": "hello"}


async def test_responses_error_event_maps_kind() -> None:
    event = {"type": "error", "code": "context_length_exceeded", "message": "too long"}
    body = f"event: error\ndata: {json.dumps(event)}\n\n".encode()
    provider = OpenAIResponsesProvider(RESPONSES, SecretStore({}), http_client=mock(Recorder(body)))
    [error] = await collect(provider, request(RESPONSES))
    assert error == pv.ProviderError("context_too_long", "too long")


async def test_responses_without_streaming() -> None:
    sse = (FIXTURES / "openai_responses_tool_call.sse").read_text()
    final = json.loads(sse.strip().split("\n\n")[-1].split("data: ", 1)[1])["response"]
    handler = Recorder(json.dumps(final).encode(), content_type="application/json")
    profile = without(RESPONSES, streaming=False)
    provider = OpenAIResponsesProvider(profile, SecretStore({}), http_client=mock(handler))
    events = await collect(provider, request(profile))
    assert handler.body_of()["stream"] is False
    assert (
        pv.TextDelta("Reading.") in events and pv.ToolUseEnd("call_1", {"path": "a.txt"}) in events
    )
    assert isinstance(events[-1], pv.MessageEnd)


async def test_openai_compat_without_streaming() -> None:
    reply = {
        "id": "c1",
        "object": "chat.completion",
        "created": 1,
        "model": "m",
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": "Reading.",
                    "tool_calls": [
                        {
                            "id": "call_9",
                            "type": "function",
                            "function": {"name": "read", "arguments": '{"path": "a"}'},
                        }
                    ],
                },
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    handler = Recorder(json.dumps(reply).encode(), content_type="application/json")
    profile = without(COMPAT, streaming=False)
    provider = OpenAICompatProvider(profile, SecretStore({}), http_client=mock(handler))
    events = await collect(provider, request(profile))
    body = handler.body_of()
    assert body["stream"] is False and "stream_options" not in body
    assert pv.TextDelta("Reading.") in events
    end = events[-1]
    assert isinstance(end, pv.MessageEnd) and end.stop_reason == "tool_use"
    assert end.final.tool_uses() == [ToolUseBlock("call_9", "read", {"path": "a"})]
    assert end.usage.output_tokens == 5


# --- headers, TLS ----------------------------------------------------------------


def test_tls_options(tmp_path: Path) -> None:
    assert pv.tls_verify(COMPAT) is True
    assert pv.tls_verify(replace(COMPAT, verify_tls=False)) is False
    with pytest.raises(pv.ProviderSetupError, match="ca_bundle"):
        pv.tls_verify(replace(COMPAT, ca_bundle=tmp_path / "missing.pem"))
    certifi = pytest.importorskip("certifi")
    ctx = pv.tls_verify(replace(COMPAT, ca_bundle=Path(certifi.where())))
    assert isinstance(ctx, ssl.SSLContext)


async def test_bad_ca_bundle_is_a_provider_error(tmp_path: Path) -> None:
    profile = replace(COMPAT, ca_bundle=tmp_path / "missing.pem")
    [event] = await collect(OpenAICompatProvider(profile, SecretStore({})), request(profile))
    assert isinstance(event, pv.ProviderError) and "ca_bundle" in event.message


async def test_secret_header_never_reaches_the_session_log(tmp_path: Path) -> None:
    secret = "gw-very-secret-value-123"
    seen: list[str | None] = []

    def server(req: httpx2.Request) -> httpx2.Response:
        seen.append(req.headers.get("x-gateway-key"))
        # A gateway that echoes the header back in its error message.
        error = {"error": {"message": f"bad gateway key {secret}", "type": "invalid_request"}}
        return httpx2.Response(400, json=error)

    profile = replace(COMPAT, headers={"X-Gateway-Key": "secret:gateway_key"})
    secrets = SecretStore({"LOOM_SECRET_GATEWAY_KEY": secret})
    provider = OpenAICompatProvider(profile, secrets, http_client=mock(server))
    ws = tmp_path / "ws"
    ws.mkdir()
    agent = build_agent(
        ws, LoomConfig(), EventBus(), provider=provider, secrets=secrets, home=tmp_path / "home"
    )
    await agent.run_turn("hi")

    assert seen == [secret]
    log = agent.log.path.read_text()
    assert "bad gateway key" in log and secret not in log


# --- text tool-call fallback -----------------------------------------------------


def parse(chunks: list[str]) -> tuple[str, list[str]]:
    parser = _Parser()
    out = "".join(parser.feed(c) for c in chunks) + parser.finish()
    return out, parser.calls


def test_parser_splits_tags_across_chunks() -> None:
    text, calls = parse(["Let me <to", 'ol_call>{"name": "read", ', '"input": {}}</tool_', "call>"])
    assert text == "Let me " and calls == ['{"name": "read", "input": {}}']


def test_parser_multiple_calls_and_invented_results() -> None:
    raw = '<tool_call>{"name":"a","input":{}}</tool_call> and <tool_call>{"name":"b","input":{}}'
    text, calls = parse([raw + '</tool_call><tool_result id="x">fake</tool_result>more'])
    assert text == " and " and len(calls) == 2


def test_parser_keeps_text_that_only_looks_like_a_tag() -> None:
    assert parse(["a < b and <tool"]) == ("a < b and <tool", [])
    assert parse(["<tool_call>{unterminated"]) == ("", ["{unterminated"])


def test_parse_call_variants() -> None:
    assert parse_call('```json\n{"name": "read", "input": {"path": "a"}}\n```').input == {
        "path": "a"
    }
    assert parse_call('{"name": "read", "arguments": {"path": "a"}}').input == {"path": "a"}
    bad = parse_call('{"name": "read", "input": {path: a}}')
    assert bad.name == "read" and pv.INVALID_JSON_KEY in bad.input
    assert parse_call("not json").name == INVALID_NAME


async def test_text_tool_calls_run_real_tools(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "a.txt").write_text("hello from a\n")
    fake = FakeProvider(
        [
            [
                TextBlock(
                    'Reading. <tool_call>{"name": "read", "input": {"path": "a.txt"}}</tool_call>'
                )
            ],
            [TextBlock('<tool_call>{"name": "read", "input": {path}}</tool_call>')],
            [TextBlock("It says hello.")],
        ]
    )
    agent = build_agent(
        ws, LoomConfig(), EventBus(), provider=TextToolCallProvider(fake), home=tmp_path / "home"
    )
    await agent.run_turn("what's in a.txt?")

    first, second, third = fake.requests
    assert first.tools == [] and "<tool_call>" in first.system and "## read" in first.system
    # History is canonical: real tool uses and results...
    call = agent.history[1].tool_uses()[0]
    assert (call.name, call.input) == ("read", {"path": "a.txt"})
    result = agent.history[2].content[0]
    assert isinstance(result, ToolResultBlock) and "hello from a" in str(result.content)
    # ...sent to the model as text.
    sent = second.messages[2].content[0]
    assert isinstance(sent, TextBlock)
    assert sent.text.startswith(f'<tool_result id="{call.id}">') and "hello from a" in sent.text
    assert "<tool_call>" in second.messages[1].text()
    # Malformed JSON gets a corrective error result.
    bad = agent.history[4].content[0]
    assert isinstance(bad, ToolResultBlock) and bad.is_error
    assert "not valid JSON" in str(bad.content)
    assert 'error="true"' in third.messages[4].content[0].text  # type: ignore[union-attr]
    assert agent.history[-1].text() == "It says hello."


# --- model switching -------------------------------------------------------------


async def test_switch_from_anthropic_to_openai_compat(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "a.txt").write_text("hi\n")
    thinking = ThinkingBlock(
        "hmm", "anthropic", {"type": "thinking", "thinking": "hmm", "signature": "SIG"}
    )
    fake = FakeProvider(
        [
            [thinking, TextBlock("Reading."), ToolUseBlock("toolu_1", "read", {"path": "a.txt"})],
            [thinking, TextBlock("It says hi.")],
        ]
    )
    agent = build_agent(ws, LoomConfig(), EventBus(), provider=fake, home=tmp_path / "home")
    await agent.run_turn("read a.txt")

    sse = (FIXTURES / "openai_tool_calls.sse").read_bytes()
    handler = Recorder(sse)
    local = replace(COMPAT, alias="local", capabilities=PROVIDER_CAPABILITIES["openai_compat"])
    agent.switch_model(
        local, OpenAICompatProvider(local, SecretStore({}), http_client=mock(handler))
    )
    await agent.run_turn("and again?")

    body = handler.body_of()
    text = json.dumps(body)
    assert "SIG" not in text and "thinking" not in text and "reasoning" not in text
    messages = body["messages"]
    assert [m["role"] for m in messages] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
        "user",
    ]
    assert messages[2]["tool_calls"][0]["id"] == "toolu_1"
    assert messages[3]["tool_call_id"] == "toolu_1"


# --- loom models -----------------------------------------------------------------


async def test_probe_and_suggestions() -> None:
    profile = replace(COMPAT, alias="local", capabilities=ModelCapabilities())
    scripts = iter(
        [
            [[TextBlock("pong")]],
            [[TextBlock("The sum is 5.")]],  # no native tool call
            [[TextBlock('<tool_call>{"name": "add", "input": {"a": 2, "b": 3}}</tool_call>')]],
            [pv.ProviderError("other", "Multimodal data provided, but model does not support")],
        ]
    )
    results = await probe_model(profile, lambda p: FakeProvider(next(scripts)))
    found = {r.name: r.ok for r in results}
    assert found == {
        "reply": True,
        "reasoning": False,
        "tools": False,
        "text tool calls": True,
        "images": False,
    }
    assert suggestions(profile, results) == ["[models.local.capabilities]", "tools = false"]


def test_models_list_cli(capsys: pytest.CaptureFixture[str]) -> None:
    from loom import cli

    assert cli.models_list(LoomConfig()) == 0
    out = capsys.readouterr().out
    assert "* sonnet" in out and "claude-opus-5-5" in out and "$2 / $10 per MTok" in out
