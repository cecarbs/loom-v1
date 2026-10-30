"""M9 acceptance: the trace tree matches the session log for a scripted run, and its spans
export to a local OTLP collector (docs/07 "Observability")."""

import asyncio
import http.server
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import ClassVar

import pytest
from loom_tools_demo import package as demo_package

from loom.app import App
from loom.cli import main
from loom.config import LoomConfig
from loom.core import events as ev
from loom.core.messages import TextBlock, ToolUseBlock
from loom.providers.fake import FakeProvider
from loom.tracing.log import Span, build_trace, render_trace

TRIAGE = (
    "rows = demo.search_logs(level='ERROR')\n"
    "t = demo.create_ticket(title=f'{len(rows)} errors')\n"
    "print(len(rows))\n"
    "t['id']"
)


def script() -> list:
    return [
        [ToolUseBlock("b1", "bash", {"command": "echo hi"})],
        [ToolUseBlock("x1", "execute", {"code": TRIAGE})],
        [TextBlock("Opened a ticket.")],
        [ToolUseBlock("r1", "read", {"path": "missing.txt"})],
        [TextBlock("No such file.")],
    ]


def make_app(tmp_path: Path, **config) -> App:
    workspace = tmp_path / "ws"
    workspace.mkdir(exist_ok=True)
    cfg = LoomConfig.model_validate(
        {"approvals": {"mode": "ask", "overrides": {"execute": "allow"}}, **config}
    )
    provider = FakeProvider(script())
    app = App(
        workspace,
        cfg,
        provider_factory=lambda p: provider,
        home=tmp_path / "home",
        packages=[demo_package],
    )

    async def approve(event: ev.AgentEvent) -> None:
        if isinstance(event, ev.ApprovalRequested):
            app.bus.respond(event.request_id, ev.Decision("allow_once"))

    app.bus.subscribe(approve)
    return app


async def run(app: App) -> None:
    await app.start()
    try:
        await app.agent.run_turn("triage the errors")
        await app.agent.run_turn("read missing.txt")
    finally:
        await app.aclose()


def shape(span: Span) -> tuple:
    return (span.kind, span.name, [shape(c) for c in span.children if c.kind != "event"])


async def test_trace_tree_matches_the_session_log(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    await run(app)
    records = app.agent.log.read()
    trace = build_trace(records)

    model = "claude-sonnet-5-5"
    assert [shape(s) for s in trace.spans if s.kind != "event"] == [
        ("turn", "turn 1", [
            ("model", model, []),
            ("tool", "bash", []),
            ("model", model, []),
            ("tool", "execute", [
                ("kernel", "e1", [
                    ("broker", "demo.search_logs", []),
                    ("broker", "demo.create_ticket", []),
                ]),
            ]),
            ("model", model, []),
        ]),
        ("turn", "turn 2", [
            ("model", model, []),
            ("tool", "read", []),
            ("model", model, []),
        ]),
    ]  # fmt: skip

    # Every model request, tool call, kernel execution and broker call is in the tree
    # exactly once, with the record's timing.
    spans = [s for top in trace.spans for s in top.walk()]
    for kind, span_kind in (
        ("model_request", "model"),
        ("tool_call", "tool"),
        ("kernel_execute", "kernel"),
        ("broker_call", "broker"),
    ):
        logged = [r for r in records if r["kind"] == kind]
        traced = [s for s in spans if s.kind == span_kind]
        assert len(logged) == len(traced), kind
    for record, span in zip(
        (r for r in records if r["kind"] == "broker_call"),
        (s for s in spans if s.kind == "broker"),
        strict=True,
    ):
        assert span.name == record["tool"] and span.end == record["ts"]
        assert span.duration == pytest.approx(record["duration_ms"] / 1000, abs=1e-6)
    results = {r["id"]: r for r in records if r["kind"] == "tool_result"}
    for tool in (s for s in spans if s.kind == "tool"):
        assert tool.end == results[tool.attrs["id"]]["ts"]
        assert tool.status == ("error" if results[tool.attrs["id"]]["is_error"] else "ok")

    # Approvals are attached where they were asked (bash: to its call, before its
    # checkpoint; the ticket write: to its broker call).
    bash = next(s for s in spans if s.name == "bash")
    assert [c.name for c in bash.children] == ["approval", "checkpoint"]
    create = next(s for s in spans if s.name == "demo.create_ticket")
    assert [(c.name, c.attrs["by"]) for c in create.children] == [("approval", "user")]

    lines = render_trace(trace)
    text = "\n".join(lines)
    assert lines[0].startswith('turn 1  "triage the errors"')
    assert "├─ bash  echo hi" in text
    assert "│  ├─ approval exec: allow once (user)" in text
    assert "├─ execute e1" in text
    assert "│  ├─ broker demo.search_logs [read] allow " in text
    assert "│  ├─ broker demo.create_ticket [write] allow (user) " in text
    assert "│  └─ result " in text
    assert "├─ read  missing.txt  [error]" in text
    assert render_trace(trace, last=1)[0].startswith('turn 2  "read missing.txt"')
    with_code = "\n".join(render_trace(trace, code=True))
    assert "│  │ rows = demo.search_logs(level='ERROR')" in with_code


async def test_loom_trace_command(tmp_path: Path, monkeypatch, capsys) -> None:
    app = make_app(tmp_path)
    await run(app)
    monkeypatch.chdir(app.workspace)
    monkeypatch.setattr("loom.session.store.loom_home", lambda: tmp_path / "home")
    assert main(["trace", "--last", "1"]) == 0
    out = capsys.readouterr().out
    assert out.startswith(f"session {app.session_id}  sonnet")
    assert 'turn 2  "read missing.txt"' in out and "turn 1" not in out
    assert main(["trace", "nope"]) == 2


def test_cancelled_and_unfinished_turns() -> None:
    records = [
        {"ts": 1.0, "kind": "session_start", "model": "local"},
        {"ts": 2.0, "kind": "message", "message": {"role": "user", "content": [
            {"type": "text", "text": "first"}]}},
        {"ts": 3.0, "kind": "tool_call", "id": "t1", "name": "bash", "args": {"command": "x"}},
        {"ts": 4.0, "kind": "turn_cancelled"},
        {"ts": 5.0, "kind": "message", "message": {"role": "user", "content": [
            {"type": "text", "text": "second"}]}},
        {"ts": 6.0, "kind": "model_request", "model": "qwen3:8b", "latency_ms": 500,
         "usage": {"input_tokens": 1500, "output_tokens": 30}},
    ]  # fmt: skip
    trace = build_trace(records)
    first, second = trace.spans
    assert first.status == "cancelled" and first.children[0].status == "incomplete"
    assert second.status == "ok" and second.end == 6.0  # no turn_end: an older log
    assert "model  local  in 1.5k out 30" in render_trace(trace)[-1]


# --- OpenTelemetry -------------------------------------------------------------------


class _Collector(http.server.BaseHTTPRequestHandler):
    received: ClassVar[list[bytes]] = []

    def do_POST(self) -> None:
        body = self.rfile.read(int(self.headers["Content-Length"]))
        if self.path == "/v1/traces":
            type(self).received.append(body)
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, *args) -> None:
        pass


@pytest.fixture
def collector() -> Iterator[tuple[str, list[bytes]]]:
    pytest.importorskip("opentelemetry.sdk")
    _Collector.received = []
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Collector)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}/v1/traces", _Collector.received
    server.shutdown()


async def test_spans_export_to_a_local_collector(tmp_path: Path, collector) -> None:
    from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
        ExportTraceServiceRequest,
    )

    endpoint, received = collector
    app = make_app(tmp_path, tracing={"otlp_endpoint": endpoint})
    await run(app)

    spans = []
    for body in received:
        request = ExportTraceServiceRequest()
        request.ParseFromString(body)
        for resource in request.resource_spans:
            service = {a.key: a.value.string_value for a in resource.resource.attributes}
            assert service["service.name"] == "loom"
            for scope in resource.scope_spans:
                spans.extend(scope.spans)
    by_id = {s.span_id: s for s in spans}

    def attrs(span) -> dict:
        out = {}
        for a in span.attributes:
            value = a.value
            out[a.key] = value.string_value or value.int_value or value.double_value
        return out

    def parent(span) -> str:
        return by_id[span.parent_span_id].name if span.parent_span_id else ""

    names = sorted((s.name, parent(s)) for s in spans)
    assert names == sorted(
        [("agent.turn", "")] * 2
        + [("llm.call", "agent.turn")] * 5
        + [("tool.call", "agent.turn")] * 3
        + [("kernel.execute", "tool.call")]
        + [("broker.call", "kernel.execute")] * 2
    )
    turns = [s for s in spans if s.name == "agent.turn"]
    assert len({s.trace_id for s in turns}) == 2  # one trace per turn
    assert {attrs(s)["session.id"] for s in spans} == {app.session_id}

    kernel = next(s for s in spans if s.name == "kernel.execute")
    assert attrs(kernel)["openinference.span.kind"] == "CHAIN"
    assert attrs(kernel)["input.value"] == TRIAGE
    llm = next(s for s in spans if s.name == "llm.call")
    assert attrs(llm)["openinference.span.kind"] == "LLM"
    assert attrs(llm)["llm.model_name"] == "claude-sonnet-5-5"
    assert attrs(llm)["llm.token_count.prompt"] == 100
    assert attrs(llm)["llm.token_count.completion"] == 20
    broker = [s for s in spans if s.name == "broker.call"]
    assert [attrs(s)["tool.name"] for s in broker] == ["demo.search_logs", "demo.create_ticket"]
    approval = [e for e in broker[1].events if e.name == "approval"]
    assert len(approval) == 1
    read = next(s for s in spans if s.name == "tool.call" and attrs(s)["tool.name"] == "read")
    assert read.status.code == 2  # STATUS_CODE_ERROR
    for s in spans:  # children within their parents
        if s.parent_span_id:
            p = by_id[s.parent_span_id]
            assert p.start_time_unix_nano <= s.start_time_unix_nano
            assert s.end_time_unix_nano <= p.end_time_unix_nano


async def test_missing_collector_is_a_notice_not_a_crash(tmp_path: Path) -> None:
    pytest.importorskip("opentelemetry.sdk")
    app = make_app(tmp_path, tracing={"otlp_endpoint": "http://127.0.0.1:9/v1/traces"})
    notices: list[str] = []

    async def record(event: ev.AgentEvent) -> None:
        if isinstance(event, ev.Notice):
            notices.append(event.text)

    app.bus.subscribe(record)
    await run(app)  # the session itself is unaffected
    await asyncio.sleep(0.05)  # the notice is scheduled from the exporter's thread
    assert notices == ["Could not export traces to http://127.0.0.1:9/v1/traces."]
    assert [r["kind"] for r in app.agent.log.read()].count("turn_end") == 2
