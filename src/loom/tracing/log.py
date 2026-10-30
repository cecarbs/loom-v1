"""The session log as a trace (docs/07 "Observability").

`build_trace` turns log records into a span tree: turn → model / tool calls; an `execute`
tool call → its kernel execution → the broker calls it made. `render_trace` prints the tree
for `loom trace` and `/trace`; the OTel exporter (`loom.tracing.otel`) sends the same spans.

Records are ordered as written. Spans end where their record is written and start
`duration_ms` / `latency_ms` earlier. Kernel executions run one at a time, so a broker call
belongs to the `execute` tool call that is open when it's logged.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

from loom.session.store import first_line

# Records that only describe the session (or are folded into other spans).
SKIPPED = frozenset({"session_start", "session_resume", "tool_result", "turn_end"})
DURATION_COLUMN = 60


@dataclass
class Span:
    kind: str  # turn | model | tool | kernel | broker | compaction | event
    name: str
    start: float  # epoch seconds
    end: float
    attrs: dict[str, Any] = field(default_factory=dict)
    children: list[Span] = field(default_factory=list)
    status: str = "ok"  # ok | error | cancelled | incomplete

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def walk(self) -> Iterator[Span]:
        yield self
        for child in self.children:
            yield from child.walk()


@dataclass
class Trace:
    session: dict[str, Any]  # the session_start record (empty if missing)
    spans: list[Span]  # turns, plus events and compactions outside any turn


class _Builder:
    def __init__(self) -> None:
        self.session: dict[str, Any] = {}
        self.top: list[Span] = []
        self.turn: Span | None = None
        self.turns = 0
        self.alias: str | None = None
        self.open_tools: dict[str, Span] = {}  # tool_call id → span, until its result
        # Children of the open execute call, until its kernel_execute record arrives.
        self.exec_children: dict[str, list[Span]] = {}
        self.approvals: dict[str, Span] = {}  # broker call id → approval, until the call
        self.last_ts = self.prev_ts = 0.0

    # --- helpers ---------------------------------------------------------------------

    def parent(self) -> list[Span]:
        return self.turn.children if self.turn is not None else self.top

    def open_execute(self) -> Span | None:
        for span in reversed(self.open_tools.values()):
            if span.name == "execute":
                return span
        return None

    def innermost(self) -> list[Span]:
        """Where events go: the newest open tool call (its kernel part for execute)."""
        if self.open_tools:
            span = next(reversed(self.open_tools.values()))
            return self.exec_children.get(span.attrs["id"], span.children)
        return self.parent()

    def event(self, record: dict[str, Any], target: list[Span] | None = None) -> None:
        ts = record["ts"]
        span = Span("event", record["kind"], ts, ts, {k: v for k, v in record.items()})
        (self.innermost() if target is None else target).append(span)

    def close_turn(self, ts: float, status: str | None = None) -> None:
        """`status` None: a turn without a `turn_end` record (logs before M9, a crash)
        counts as complete unless a tool call has no result."""
        if self.turn is None:
            return
        if status is None:
            status = "incomplete" if self.open_tools else "ok"
        for tool in list(self.open_tools.values()):
            self.finish_tool(tool, None)
        self.turn.end = max(ts, self.turn.end)
        self.turn.status = status
        self.turn = None

    def finish_tool(self, tool: Span, record: dict[str, Any] | None) -> None:
        call_id = tool.attrs["id"]
        self.open_tools.pop(call_id, None)
        tool.children.extend(self.exec_children.pop(call_id, []))
        if record is None:
            tool.end = self.last_ts
            tool.status = "incomplete"
            return
        tool.end = record["ts"]
        tool.status = "error" if record.get("is_error") else "ok"
        tool.attrs.update(
            result=record.get("result", ""),
            spill_path=record.get("spill_path"),
            duration_ms=record.get("duration_ms"),
        )

    # --- records ---------------------------------------------------------------------

    def add(self, record: dict[str, Any]) -> None:
        kind = record.get("kind")
        ts = record.get("ts", self.last_ts)
        record["ts"] = ts
        self.prev_ts, self.last_ts = self.last_ts, max(self.last_ts, ts)
        match kind:
            case "session_start" | "session_resume":
                self.session = self.session or record
                self.alias = record.get("model")
            case "model_switch":
                self.alias = record.get("new")
                self.event(record, self.parent())
            case "message":
                self.message(record)
            case "model_request":
                self.model(record)
            case "tool_call":
                tool = Span(
                    "tool", record.get("name", "?"), ts, ts,
                    {"id": record.get("id", ""), "args": record.get("args")},
                )  # fmt: skip
                self.parent().append(tool)
                self.open_tools[tool.attrs["id"]] = tool
                if tool.name == "execute":
                    self.exec_children[tool.attrs["id"]] = []
            case "tool_result":
                if (tool := self.open_tools.get(record.get("id", ""))) is not None:
                    self.finish_tool(tool, record)
            case "kernel_execute":
                self.kernel(record)
            case "broker_call":
                self.broker(record)
            case "approval":
                call_id = record.get("call_id", "")
                span = Span("event", "approval", ts, ts, dict(record))
                if (tool := self.open_tools.get(call_id)) is not None:
                    tool.children.append(span)
                elif self.open_execute() is not None:
                    self.approvals[call_id] = span  # attached to its broker call
                else:
                    self.parent().append(span)
            case "compaction":
                self.compaction(record)
            case "turn_end":
                self.close_turn(ts)
            case "turn_cancelled":
                self.event(record, self.parent())
                self.close_turn(ts, "cancelled")
            case _:
                self.event(record)

    def message(self, record: dict[str, Any]) -> None:
        msg = record.get("message") or {}
        content = msg.get("content") or []
        texts = [b.get("text", "") for b in content if b.get("type") == "text"]
        if msg.get("role") == "assistant":
            # The model span this message came from is the newest one in the turn.
            for span in reversed(self.parent()):
                if span.kind == "model":
                    span.attrs["output"] = "\n".join(texts)
                    span.attrs["tool_calls"] = [
                        b.get("name") for b in content if b.get("type") == "tool_use"
                    ]
                    break
            return
        is_prompt = (
            texts
            and not (msg.get("meta") or {}).get("compaction")
            and not any(b.get("type") == "tool_result" for b in content)
        )
        if not is_prompt:
            return
        self.close_turn(self.prev_ts)
        self.turns += 1
        self.turn = Span(
            "turn",
            f"turn {self.turns}",
            record["ts"],
            record["ts"],
            {"turn": self.turns, "prompt": texts[0], "cost_usd": None},
        )
        self.top.append(self.turn)

    def model(self, record: dict[str, Any]) -> None:
        ts = record["ts"]
        usage = record.get("usage") or {}
        span = Span(
            "model",
            record.get("model", "?"),
            ts - (record.get("latency_ms") or 0) / 1000,
            ts,
            {
                "alias": self.alias,
                "provider": self.session.get("provider"),
                "usage": usage,
                "cost_usd": record.get("cost_usd"),
                "stop_reason": record.get("stop_reason"),
                "purpose": record.get("purpose", "turn"),
                "settings": record.get("settings") or {},
            },
        )
        self.parent().append(span)
        if self.turn is not None and (cost := record.get("cost_usd")) is not None:
            self.turn.attrs["cost_usd"] = (self.turn.attrs["cost_usd"] or 0.0) + cost

    def kernel(self, record: dict[str, Any]) -> None:
        ts = record["ts"]
        tool = self.open_execute()
        children = self.exec_children.get(tool.attrs["id"], []) if tool else []
        start = ts - (record.get("duration_ms") or 0) / 1000
        if record.get("duration_ms") is None and children:
            start = children[0].start
        status = record.get("status", "ok")
        span = Span(
            "kernel",
            record.get("id", "?"),
            start,
            ts,
            {k: record.get(k) for k in ("id", "code", "status", "ename", "out_ref", "restarted")},
            children=list(children),
            status={"ok": "ok", "interrupted": "cancelled"}.get(status, "error"),
        )
        if tool is not None:
            self.exec_children[tool.attrs["id"]] = []
            tool.children.append(span)
        else:
            self.parent().append(span)

    def broker(self, record: dict[str, Any]) -> None:
        ts = record["ts"]
        span = Span(
            "broker",
            record.get("tool", "?"),
            ts - (record.get("duration_ms") or 0) / 1000,
            ts,
            dict(record),
            status="ok" if record.get("ok") else "error",
        )
        if (approval := self.approvals.pop(record.get("call_id", ""), None)) is not None:
            span.children.append(approval)
        tool = self.open_execute()
        if tool is not None:
            self.exec_children[tool.attrs["id"]].append(span)
        else:
            self.parent().append(span)

    def compaction(self, record: dict[str, Any]) -> None:
        """Its summary request (logged just before) becomes its child."""
        ts = record["ts"]
        parent = self.parent()
        children = []
        while parent and parent[-1].kind == "model" and parent[-1].attrs["purpose"] != "turn":
            children.insert(0, parent.pop())
        span = Span(
            "compaction",
            "compaction",
            children[0].start if children else ts,
            ts,
            {k: record.get(k) for k in ("tokens_before", "tokens_after", "first_kept")},
            children=children,
        )
        parent.append(span)

    def finish(self) -> Trace:
        self.close_turn(self.last_ts)
        return Trace(self.session, self.top)


def build_trace(records: Iterable[dict[str, Any]]) -> Trace:
    builder = _Builder()
    for record in records:
        builder.add(dict(record))
    return builder.finish()


# --- rendering -----------------------------------------------------------------------


def render_trace(trace: Trace, *, last: int | None = None, code: bool = False) -> list[str]:
    """Text lines for the tree; `last` keeps the last n turns, `code` adds kernel code."""
    spans = trace.spans
    if last is not None:
        turns = [s for s in spans if s.kind == "turn"]
        if len(turns) > last:
            cut = spans.index(turns[-last]) if last > 0 else len(spans)
            spans = spans[cut:]
    lines: list[str] = []
    for span in spans:
        lines.append(_line("", span))
        _children(lines, span, "", code)
    return lines


def _children(lines: list[str], span: Span, indent: str, code: bool) -> None:
    kids = _display_children(span)
    for i, child in enumerate(kids):
        last = i == len(kids) - 1
        branch, cont = ("└─ ", "   ") if last else ("├─ ", "│  ")
        if isinstance(child, str):
            lines.append(indent + branch + child)
            continue
        lines.append(_line(indent + branch, child))
        if code and child.kind == "tool" and child.name == "execute":
            kernel = _kernel(child)
            source = kernel.attrs.get("code") if kernel is not None else None
            for text in (source or "").rstrip().splitlines():
                lines.append(indent + cont + "│ " + text)
        _children(lines, child, indent + cont, code)


def _kernel(tool: Span) -> Span | None:
    return next((c for c in tool.children if c.kind == "kernel"), None)


def _display_children(span: Span) -> list[Span | str]:
    """An execute call shows as one node: its kernel execution's children, then a
    result line. Approvals of broker calls are shown on the broker line instead."""
    if span.kind == "broker":
        return []
    if span.kind == "tool" and span.name == "execute":
        out: list[Span | str] = []
        for child in span.children:
            out.extend(child.children if child.kind == "kernel" else [child])
        if span.status != "incomplete":
            out.append(_result_line(span))
        return out
    return list(span.children)


def _result_line(tool: Span) -> str:
    text = tool.attrs.get("result") or ""
    line = f"result {len(text)} chars"
    if tool.attrs.get("spill_path"):
        line += " (spilled)"
    return line


def _line(prefix: str, span: Span) -> str:
    label, extra = _label(span)
    head = prefix + label
    duration = "" if span.kind == "event" else _seconds(span.duration)
    if duration:
        pad = max(2, DURATION_COLUMN - len(head) - len(duration))
        head = head + " " * pad + duration
    return head + (f"  {extra}" if extra else "")


def _label(span: Span) -> tuple[str, str]:
    a = span.attrs
    match span.kind:
        case "turn":
            label = f'turn {a["turn"]}  "{first_line(a["prompt"], 40)}"'
            if span.status != "ok":
                label += f"  [{span.status}]"
            cost = a.get("cost_usd")
            return label, f"${cost:.2f}" if cost is not None else ""
        case "model":
            usage = a.get("usage") or {}
            cached = usage.get("cache_read_tokens", 0)
            prompt = usage.get("input_tokens", 0) + cached + usage.get("cache_write_tokens", 0)
            label = f"model  {a.get('alias') or span.name}  in {_k(prompt)}"
            if cached:
                label += f" (cache {_k(cached)})"
            label += f" out {_k(usage.get('output_tokens', 0))}"
            if a.get("purpose") != "turn":
                label += f"  [{a.get('purpose')}]"
            if a.get("stop_reason") not in (None, "end_turn", "tool_use"):
                label += f"  [{a['stop_reason']}]"
            return label, ""
        case "tool":
            if span.name == "execute":
                kernel = _kernel(span)
                label = f"execute {kernel.name}" if kernel else "execute"
                if kernel and kernel.attrs.get("status") != "ok":
                    label += f"  {kernel.attrs.get('ename') or kernel.attrs.get('status')}"
                elif span.status == "error":
                    label += "  error"
            else:
                label = f"{span.name}  {_tool_summary(span)}".rstrip()
                if span.status == "error":
                    label += "  [error]"
            if span.status == "incomplete":
                label += "  [no result]"
            return label, ""
        case "kernel":
            return f"kernel {span.name}  {a.get('status')}", ""
        case "broker":
            label = f"broker {span.name} [{a.get('effect') or '?'}] {_decision(a)}"
            if a.get("ok"):
                return label, f"{_k(a.get('result_chars', 0))} chars"
            error = a.get("error") or {}
            return label, f"{error.get('type', 'error')}"
        case "compaction":
            before, after = a.get("tokens_before"), a.get("tokens_after")
            if before is not None and after is not None:
                return f"compaction  {_k(before)} → {_k(after)} tokens", ""
            return "compaction", ""
        case _:
            return _event_label(span), ""


def _decision(record: dict[str, Any]) -> str:
    if blocked := record.get("blocked"):
        return f"blocked by {blocked.get('by')}"
    decision = record.get("decision")
    if not decision:
        return "not run"
    action = "allow" if str(decision.get("action", "")).startswith("allow") else "deny"
    by = decision.get("by")
    return action if by == "policy" else f"{action} ({by})"


def _event_label(span: Span) -> str:
    a = span.attrs
    text = first_line(str(a.get("text", "")), 60)
    match span.name:
        case "approval":
            action = str(a.get("action", "")).replace("_", " ")
            return f"approval {a.get('access')}: {action} ({a.get('by')})"
        case "network":
            target = f"{a.get('host')}:{a.get('port')}"
            outcome = "error" if a.get("error") else "allowed" if a.get("allowed") else "denied"
            return f"network {a.get('method', '')} {target} {outcome}"
        case "checkpoint":
            return f"checkpoint {str(a.get('sha', ''))[:7]}  {len(a.get('files') or [])} files"
        case "kernel_start":
            return f"kernel start ({a.get('backend')}, python {a.get('python')})"
        case "kernel_exit":
            return f"kernel exit {a.get('returncode')}"
        case "kernel_kill":
            return f"kernel killed: {a.get('reason')}"
        case "kernel_interrupt":
            return "kernel interrupt"
        case "model_switch":
            return f"model switch {a.get('old')} → {a.get('new')}"
        case "mode_change":
            return f"mode {a.get('mode')}"
        case "turn_cancelled":
            return "cancelled"
        case "error" | "notice" | "note":
            return f"{span.name}: {text}"
        case _:
            return span.name + (f": {text}" if text else "")


def _tool_summary(span: Span) -> str:
    args = span.attrs.get("args")
    if not isinstance(args, dict):
        return ""
    for key in ("command", "path", "query"):
        if isinstance(args.get(key), str):
            return first_line(args[key], 40)
    return first_line(json.dumps(args, ensure_ascii=False, default=str), 40) if args else ""


def _k(n: int | float | None) -> str:
    n = n or 0
    return f"{n / 1000:.1f}k" if n >= 1000 else str(int(n))


def _seconds(s: float) -> str:
    return f"{s * 1000:.0f}ms" if s < 1 else f"{s:.1f}s"
