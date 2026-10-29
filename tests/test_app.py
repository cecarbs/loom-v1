from pathlib import Path

from loom.app import App
from loom.config import LoomConfig
from loom.core import events as ev
from loom.core.mentions import find_mentions
from loom.core.messages import ImageBlock, TextBlock, ToolUseBlock
from loom.providers.fake import FakeProvider, Script

PNG_1PX = bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000001000000010806000000"
    "1f15c4890000000d49444154789c6360000002000154a24f5d0000000049454e44ae426082"
)


def make_app(tmp_path: Path, script: list[Script]) -> tuple[App, FakeProvider, list]:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    provider = FakeProvider(script)
    app = App(workspace, LoomConfig(), provider_factory=lambda p: provider, home=tmp_path / "home")
    events: list = []

    async def record(e: ev.AgentEvent) -> None:
        events.append(e)

    app.bus.subscribe(record)
    return app, provider, events


def test_find_mentions() -> None:
    assert find_mentions("look at @src/a.py, and @b.txt.") == ["src/a.py", "b.txt"]
    assert find_mentions("@first then mail me@example.com") == ["first"]


async def test_mention_attaches_file_and_counts_as_read(tmp_path: Path) -> None:
    app, provider, events = make_app(
        tmp_path,
        [
            [ToolUseBlock("t1", "edit", {"path": "a.py", "old_string": "1", "new_string": "2"})],
            [TextBlock("done")],
        ],
    )
    (app.workspace / "a.py").write_text("x = 1\n")
    await app.submit("bump @a.py please, cc @alice")

    user = provider.requests[0].messages[0]
    assert user.content[0] == TextBlock("bump @a.py please, cc @alice")
    attached = user.content[1]
    assert isinstance(attached, TextBlock)
    assert attached.text.startswith('<file path="a.py">') and "x = 1" in attached.text
    assert len(user.content) == 2  # @alice isn't a file, so it stays plain text
    # The edit worked without a separate read.
    assert (app.workspace / "a.py").read_text() == "x = 2\n"
    assert any(isinstance(e, ev.Notice) and "Attached a.py" in e.text for e in events)


async def test_mention_respects_read_policy(tmp_path: Path) -> None:
    app, provider, events = make_app(tmp_path, [[TextBlock("ok")]])
    (app.workspace / ".env").write_text("TOKEN=secret")
    await app.submit("what is in @.env")
    user = provider.requests[0].messages[0]
    assert len(user.content) == 1 and "secret" not in str(user.content)
    assert any(isinstance(e, ev.Notice) and "not attached" in e.text for e in events)


async def test_mention_attaches_images(tmp_path: Path) -> None:
    app, provider, _ = make_app(tmp_path, [[TextBlock("a dot")]])
    (app.workspace / "dot.png").write_bytes(PNG_1PX)
    await app.submit("what is @dot.png")
    assert any(isinstance(b, ImageBlock) for b in provider.requests[0].messages[0].content)


async def test_large_mention_is_budgeted(tmp_path: Path) -> None:
    app, provider, _ = make_app(tmp_path, [[TextBlock("ok")]])
    (app.workspace / "big.txt").write_text("\n".join("y" * 60 for _ in range(1_500)))
    await app.submit("summarize @big.txt")
    attached = provider.requests[0].messages[0].content[1]
    assert isinstance(attached, TextBlock)
    assert "chars omitted" in attached.text and len(attached.text) < 12_500


async def test_switch_model_keeps_history(tmp_path: Path) -> None:
    config = LoomConfig.model_validate(
        {"models": {"local": {"provider": "openai_compat", "model": "qwen3:8b"}}}
    )
    workspace = tmp_path / "ws"
    workspace.mkdir()
    providers = {
        "sonnet": FakeProvider([[TextBlock("from sonnet")]]),
        "local": FakeProvider([[TextBlock("from local")]]),
    }
    app = App(workspace, config, provider_factory=lambda p: providers[p.alias], home=tmp_path / "h")
    await app.submit("one")
    assert app.agent.cost_usd is not None and app.agent.cost_usd > 0
    cost = app.agent.cost_usd

    app.switch_model("local")
    await app.submit("two")
    request = providers["local"].requests[0]
    assert request.model == "qwen3:8b"
    assert [m.text() for m in request.messages] == ["one", "from sonnet", "two"]
    assert app.agent.cost_usd == cost  # the unpriced local model adds nothing
    assert app.agent.tool_ctx.model_accepts_images is False
