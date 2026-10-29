"""ToolBroker: the one choke point for kernel tool calls (docs/04 "Broker call pipeline").

resolve → validate → (hooks, M8) → approve → execute with timeout → serialize + redact →
log → reply. Package code runs here, on the host, never in the sandbox.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import dataclasses
import datetime
import decimal
import difflib
import json
import logging
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import BaseModel, ValidationError

from loom.broker.approval import Access, Approvals
from loom.broker.registry import Effect, PackageError, ToolPackage, ToolSpec
from loom.broker.secrets import SecretStore
from loom.core.events import Decision
from loom.session.store import SessionLog

log = logging.getLogger(__name__)

PREVIEW_VALUE_CHARS = 300


class Clock(Protocol):
    """The calling execution's timeout clock; paused while the user is being asked."""

    def paused(self) -> contextlib.AbstractContextManager[None]: ...


class MissingSecret(Exception):
    pass


@dataclass(frozen=True)
class ToolReply:
    ok: bool
    value: Any = None
    error_type: str = ""
    message: str = ""

    @classmethod
    def error(cls, error_type: str, message: str) -> ToolReply:
        return cls(False, error_type=error_type, message=message)

    def message_for(self, call_id: str) -> dict[str, Any]:
        """The `tool_result` protocol message."""
        if self.ok:
            return {"type": "tool_result", "call_id": call_id, "ok": True, "value": self.value}
        return {
            "type": "tool_result",
            "call_id": call_id,
            "ok": False,
            "error": {"type": self.error_type, "message": self.message},
        }


class PackageContext:
    """What a tool function gets as `ctx`: its secrets, config, cached clients, a logger."""

    def __init__(
        self, package: ToolPackage, config: Mapping[str, Any], secrets: SecretStore
    ) -> None:
        self.package = package
        self.config: dict[str, Any] = dict(config)
        self.log = logging.getLogger(f"loom.packages.{package.name}")
        self._secrets = secrets
        self._clients: dict[str, Any] = {}
        self._lock = threading.Lock()  # sync tools run in worker threads

    def secret(self, name: str) -> str:
        declared = self.package.secrets.get(name)
        if declared is None:
            raise PackageError(f"{self.package.name} did not declare a secret named {name!r}")
        value = self._secrets.get(declared.name)
        if not value:
            raise MissingSecret(
                f"Secret {declared.name!r} is not set. The user can set it with "
                f"`loom secrets set {declared.name}`."
            )
        return value

    def client(self, name: str) -> Any:
        with self._lock:
            if name not in self._clients:
                factory = self.package.clients.get(name)
                if factory is None:
                    raise PackageError(f"{self.package.name} has no client named {name!r}")
                self._clients[name] = factory(self)
            return self._clients[name]


class ToolBroker:
    def __init__(
        self,
        packages: Sequence[ToolPackage],
        *,
        approvals: Approvals,
        secrets: SecretStore,
        session_log: SessionLog,
        package_config: Mapping[str, Mapping[str, Any]] | None = None,
        call_timeout_s: float = 120,
        log_results: bool = False,
    ) -> None:
        self.packages = {p.name: p for p in packages}
        self.specs = {s.qualname: s for p in packages for s in p.tools.values()}
        self.approvals = approvals
        self.secrets = secrets
        self.redactor = secrets.redactor
        self.log = session_log
        self.call_timeout_s = call_timeout_s
        self.log_results = log_results
        package_config = package_config or {}
        self._contexts = {
            p.name: PackageContext(p, package_config.get(p.name, {}), secrets) for p in packages
        }

    def catalog(self) -> list[dict[str, Any]]:
        return [p.catalog_entry() for p in self.packages.values()]

    # --- the pipeline -----------------------------------------------------------------

    async def call(
        self, tool: str, args: Any, *, call_id: str = "", clock: Clock | None = None
    ) -> ToolReply:
        start = time.monotonic()
        spec = self.specs.get(tool)
        decision: Decision | None = None
        try:
            if spec is None:
                reply = ToolReply.error("ToolNotFound", self._not_found(tool))
            elif (problem := self._validate(spec, args := from_jsonable(args))) is not None:
                reply = problem
            else:
                kwargs = self._kwargs(spec, args)
                # Extension `before_broker_call` hooks run here (M8).
                with clock.paused() if clock is not None else contextlib.nullcontext():
                    decision = await self.approvals.check(
                        tool, _access(spec, kwargs), call_id=call_id
                    )
                if not decision.allowed:
                    reply = ToolReply.error("PermissionDenied", self._denial(spec, decision))
                else:
                    reply = await self._run(spec, kwargs)
        except asyncio.CancelledError:
            self._log(tool, args, spec, decision, start, ToolReply.error("Cancelled", ""), call_id)
            raise
        self._log(tool, args, spec, decision, start, reply, call_id)
        return reply

    def _validate(self, spec: ToolSpec, args: Any) -> ToolReply | None:
        if not isinstance(args, dict):
            return ToolReply.error("InvalidArguments", f"{spec.qualname}: args must be an object")
        try:
            spec.args_model.model_validate(args)
        except ValidationError as e:
            problems = "; ".join(
                f"{'.'.join(str(p) for p in err['loc']) or '<args>'}: {err['msg']}"
                for err in e.errors()
            )
            return ToolReply.error("InvalidArguments", f"{spec.qualname}: {problems}")
        return None

    def _kwargs(self, spec: ToolSpec, args: dict[str, Any]) -> dict[str, Any]:
        model = spec.args_model.model_validate(args)  # coerced values (e.g. "5" → 5)
        return {name: getattr(model, name) for name in type(model).model_fields}

    async def _run(self, spec: ToolSpec, kwargs: dict[str, Any]) -> ToolReply:
        ctx = self._contexts[spec.package]
        try:
            if spec.is_async:
                coro = spec.func(ctx, **kwargs)
            else:
                coro = asyncio.to_thread(spec.func, ctx, **kwargs)
            value = await asyncio.wait_for(coro, self.call_timeout_s)
        except TimeoutError:
            return ToolReply.error(
                "ToolTimeout", f"{spec.qualname} timed out after {self.call_timeout_s:g}s"
            )
        except MissingSecret as e:
            return ToolReply.error("MissingSecret", str(e))
        except Exception as e:
            log.warning("tool %s raised", spec.qualname, exc_info=True)
            return ToolReply.error(
                "ToolError", self.redactor.text(f"{spec.qualname}: {type(e).__name__}: {e}")
            )
        try:
            return ToolReply(True, self.redactor.value(to_jsonable(value)))
        except TypeError as e:
            return ToolReply.error(
                "ToolError", f"{spec.qualname} returned a value loom can't send: {e}"
            )

    # --- helpers ----------------------------------------------------------------------

    def _not_found(self, tool: str) -> str:
        close = difflib.get_close_matches(tool, list(self.specs), n=3, cutoff=0.5)
        hint = f" Did you mean: {', '.join(close)}?" if close else ""
        return f"No tool named {tool!r}.{hint} Use search_tools to find tools."

    def _denial(self, spec: ToolSpec, decision: Decision) -> str:
        if decision.by == "user":
            base = f"User denied {spec.qualname}"
            return f"{base}: {decision.message}" if decision.message else base
        if decision.by == "override":
            return f"{spec.qualname} is denied by the user's configuration"
        return f"{spec.qualname} ({spec.effect}) is not allowed in {self.approvals.mode} mode"

    def _log(
        self,
        tool: str,
        args: Any,
        spec: ToolSpec | None,
        decision: Decision | None,
        start: float,
        reply: ToolReply,
        call_id: str,
    ) -> None:
        record: dict[str, Any] = {
            "tool": tool,
            "call_id": call_id,
            "args": self.redactor.value(args),
            "effect": spec.effect if spec else None,
            "decision": {"action": decision.action, "by": decision.by} if decision else None,
            "duration_ms": int((time.monotonic() - start) * 1000),
            "ok": reply.ok,
        }
        if reply.ok:
            record["result_chars"] = len(json.dumps(reply.value, default=str))
            if self.log_results:
                record["result"] = reply.value
        else:
            record["error"] = {"type": reply.error_type, "message": reply.message}
        self.log.append("broker_call", **record)

    # --- search_tools -----------------------------------------------------------------

    def search(self, query: str, limit: int = 10) -> str:
        if not self.packages:
            return "No tool packages are installed."
        terms = query.lower().split()
        if not terms:
            lines = [
                f"{p.name} — {p.description} ({len(p.tools)} tools)" for p in self.packages.values()
            ]
            return "Packages (search by keyword to see functions):\n" + "\n".join(lines)
        scored = []
        for spec in self.specs.values():
            score = _score(spec, terms)
            if score > 0:
                scored.append((-score, spec.qualname, spec))
        if not scored:
            names = ", ".join(self.packages)
            return f"No tools match {query!r}. Packages: {names}. Try search_tools('')."
        return "\n".join(_describe(spec) for _, _, spec in sorted(scored)[:limit])


def _access(spec: ToolSpec, kwargs: Mapping[str, Any]) -> Access:
    lines = []
    for name, value in kwargs.items():
        text = json.dumps(to_jsonable(value), ensure_ascii=False, default=str)
        if len(text) > PREVIEW_VALUE_CHARS:
            text = text[: PREVIEW_VALUE_CHARS - 1] + "…"
        lines.append(f"{name} = {text}")
    kind: Effect = spec.effect
    return Access(kind, "\n".join(lines), table="broker")


def _score(spec: ToolSpec, terms: list[str]) -> int:
    name, package, doc = spec.name.lower(), spec.package.lower(), spec.doc.lower()
    words = name.replace("_", " ").split()
    total = 0
    for term in terms:
        if term in name:
            total += 4
        elif term in package:
            total += 3
        elif term in doc:
            total += 1
        elif difflib.get_close_matches(term, words, n=1, cutoff=0.8):
            total += 1
    return total


def _describe(spec: ToolSpec) -> str:
    summary = spec.doc.split("\n\n")[0].replace("\n", " ").strip()
    return spec.display() + (f"\n    {summary}" if summary else "")


def from_jsonable(value: Any) -> Any:
    """Undo the kernel's encoding of arguments: {"__bytes__": b64} → bytes."""
    if isinstance(value, dict):
        if len(value) == 1 and isinstance(value.get("__bytes__"), str):
            return base64.b64decode(value["__bytes__"])
        return {k: from_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [from_jsonable(v) for v in value]
    return value


def to_jsonable(value: Any) -> Any:
    """Tool results must be JSON: datetimes → ISO strings, bytes → {"__bytes__": b64}."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, (bytes, bytearray)):
        return {"__bytes__": base64.b64encode(bytes(value)).decode("ascii")}
    if isinstance(value, (datetime.datetime, datetime.date, datetime.time)):
        return value.isoformat()
    if isinstance(value, decimal.Decimal):
        return float(value)
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return to_jsonable(dataclasses.asdict(value))
    raise TypeError(f"{type(value).__name__} is not JSON-serializable")
