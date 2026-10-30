"""OpenTelemetry export of the session trace (docs/07 "Level 2"; extra `loom[otel]`).

Spans are built from session-log records, the same way as `loom trace`, so the two always
agree and nothing reaches a span that wasn't redacted for the log. Each turn is exported
when its `turn_end` (or `turn_cancelled`) record is written: `agent.turn` → `llm.call`,
`tool.call` → `kernel.execute` → `broker.call`. Point records (approvals, network, kernel
start, errors) become span events. Attribute names follow OpenInference conventions.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Sequence
from typing import Any

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter, SpanExportResult
from opentelemetry.trace import Status, StatusCode

from loom import __version__
from loom.session.store import SessionLog
from loom.tracing.log import Span, build_trace

MAX_VALUE_CHARS = 8000
EXPORT_TIMEOUT_S = 5.0
TURN_ENDS = frozenset({"turn_end", "turn_cancelled"})
SPAN_NAMES = {
    "turn": "agent.turn",
    "model": "llm.call",
    "tool": "tool.call",
    "kernel": "kernel.execute",
    "broker": "broker.call",
    "compaction": "compaction",
}
SPAN_KINDS = {  # openinference.span.kind
    "turn": "AGENT",
    "model": "LLM",
    "tool": "TOOL",
    "kernel": "CHAIN",
    "broker": "TOOL",
    "compaction": "CHAIN",
}

ErrorCallback = Callable[[str], None]


class _Reporting(SpanExporter):
    """Reports the first failed export (e.g. no collector running) instead of the SDK's
    log warnings, which would print over the TUI."""

    def __init__(self, inner: SpanExporter, describe: str) -> None:
        self.inner = inner
        self.describe = describe
        self.on_error: ErrorCallback | None = None
        self._reported = False

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        result = self.inner.export(spans)
        if result is not SpanExportResult.SUCCESS and not self._reported:
            self._reported = True
            if self.on_error is not None:
                self.on_error(f"Could not export traces to {self.describe}.")
        return result

    def shutdown(self) -> None:
        self.inner.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return self.inner.force_flush(timeout_millis)


class OtelExporter:
    def __init__(self, endpoint: str, *, exporter: SpanExporter | None = None) -> None:
        for name in ("opentelemetry", "urllib3"):
            logging.getLogger(name).propagate = False  # see _Reporting
        inner = exporter or OTLPSpanExporter(endpoint=endpoint, timeout=EXPORT_TIMEOUT_S)
        self._exporter = _Reporting(inner, endpoint)
        resource = Resource.create(
            {
                "service.name": "loom",
                "service.version": __version__,
                "openinference.project.name": "loom",
            }
        )
        self.provider = TracerProvider(resource=resource)
        self.provider.add_span_processor(BatchSpanProcessor(self._exporter))
        self.tracer = self.provider.get_tracer("loom", __version__)
        self._log: SessionLog | None = None
        self._records: list[dict[str, Any]] = []
        self._exported = 0  # top-level spans of this session already sent
        self.session_id = ""

    @property
    def on_error(self) -> ErrorCallback | None:
        return self._exporter.on_error

    @on_error.setter
    def on_error(self, callback: ErrorCallback | None) -> None:
        self._exporter.on_error = callback

    def attach(self, log: SessionLog, session_id: str) -> None:
        """Follow a session's log (a new one after /clear or /resume)."""
        self.detach()
        self._log, self.session_id = log, session_id
        self._records, self._exported = [], 0
        log.listeners.append(self._on_record)

    def detach(self) -> None:
        """Export what's left (also an unfinished turn) and stop following."""
        if self._log is None:
            return
        self._export()
        self._log.listeners.remove(self._on_record)
        self._log = None

    def shutdown(self) -> None:
        self.detach()
        self.provider.shutdown()

    def _on_record(self, record: dict[str, Any]) -> None:
        self._records.append(record)
        if record.get("kind") in TURN_ENDS:
            self._export()

    def _export(self) -> None:
        """New top-level spans: at a turn end, everything up to that turn is finished.
        The log is rebuilt each time so turn numbers and model aliases stay right."""
        ready = build_trace(self._records).spans[self._exported :]
        for span in ready:
            if span.kind != "event":  # outside any turn (kernel exit at shutdown, …)
                self._emit(span, None)
        self._exported += len(ready)

    def _emit(self, span: Span, parent: trace.Span | None) -> None:
        context = trace.set_span_in_context(parent) if parent is not None else None
        otel = self.tracer.start_span(
            SPAN_NAMES.get(span.kind, span.kind),
            context=context,
            start_time=_ns(span.start),
            attributes=self._attributes(span),
        )
        for child in span.children:
            if child.kind == "event":
                otel.add_event(child.name, _event_attributes(child), timestamp=_ns(child.start))
            else:
                self._emit(child, otel)
        if span.status in ("error", "cancelled", "incomplete"):
            otel.set_status(Status(StatusCode.ERROR, span.status))
        otel.end(end_time=_ns(max(span.end, span.start)))

    def _attributes(self, span: Span) -> dict[str, Any]:
        a = span.attrs
        out: dict[str, Any] = {
            "openinference.span.kind": SPAN_KINDS.get(span.kind, "CHAIN"),
            "session.id": self.session_id,
        }
        match span.kind:
            case "turn":
                out |= {"input.value": a.get("prompt", ""), "loom.turn": a.get("turn", 0)}
                if a.get("cost_usd") is not None:
                    out["llm.cost.total"] = a["cost_usd"]
            case "model":
                usage = a.get("usage") or {}
                cached = usage.get("cache_read_tokens", 0)
                written = usage.get("cache_write_tokens", 0)
                prompt = usage.get("input_tokens", 0) + cached + written
                completion = usage.get("output_tokens", 0)
                out |= {
                    "llm.model_name": span.name,
                    "llm.provider": a.get("provider") or "",
                    "llm.invocation_parameters": json.dumps(a.get("settings") or {}),
                    "llm.token_count.prompt": prompt,
                    "llm.token_count.completion": completion,
                    "llm.token_count.total": prompt + completion,
                    "llm.token_count.prompt_details.cache_read": cached,
                    "llm.token_count.prompt_details.cache_write": written,
                    "output.value": _clip(a.get("output") or ""),
                    "loom.model_alias": a.get("alias") or "",
                    "loom.purpose": a.get("purpose") or "turn",
                    "loom.stop_reason": a.get("stop_reason") or "",
                }
                if a.get("cost_usd") is not None:
                    out["llm.cost.total"] = a["cost_usd"]
            case "tool":
                out |= {
                    "tool.name": span.name,
                    "input.value": _clip(json.dumps(a.get("args"), default=str)),
                    "input.mime_type": "application/json",
                    "output.value": _clip(a.get("result") or ""),
                    "loom.tool_call_id": a.get("id", ""),
                }
            case "kernel":
                out |= {
                    "input.value": _clip(a.get("code") or ""),  # docs/07: code on the span
                    "loom.exec_id": a.get("id") or "",
                    "loom.status": a.get("status") or "",
                    "loom.ename": a.get("ename") or "",
                }
            case "broker":
                decision = a.get("decision") or {}
                out |= {
                    "tool.name": span.name,
                    "input.value": _clip(json.dumps(a.get("args"), default=str)),
                    "input.mime_type": "application/json",
                    "loom.effect": a.get("effect") or "",
                    "loom.decision": decision.get("action", ""),
                    "loom.decision_by": decision.get("by", ""),
                    "loom.result_chars": a.get("result_chars", 0),
                }
                if "result" in a:  # only with [tracing] log_tool_results
                    out["output.value"] = _clip(json.dumps(a["result"], default=str))
            case "compaction":
                out |= {
                    "loom.tokens_before": a.get("tokens_before") or 0,
                    "loom.tokens_after": a.get("tokens_after") or 0,
                }
        return out


def _event_attributes(span: Span) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in span.attrs.items():
        if key in ("ts", "kind"):
            continue
        if isinstance(value, bool | int | float | str):
            out[key] = _clip(value) if isinstance(value, str) else value
        elif value is not None:
            out[key] = _clip(json.dumps(value, default=str))
    return out


def _clip(text: str) -> str:
    return text if len(text) <= MAX_VALUE_CHARS else text[:MAX_VALUE_CHARS] + "…"


def _ns(seconds: float) -> int:
    return int(seconds * 1_000_000_000)
