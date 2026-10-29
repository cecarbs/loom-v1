"""Host-side types for the kernel's JSON-lines protocol (docs/04 "Protocol")."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

KernelState = Literal["off", "starting", "idle", "busy", "dead"]
ExecutionStatus = Literal["ok", "error", "timeout", "died"]


@dataclass(frozen=True)
class KernelError:
    ename: str
    evalue: str
    traceback: str


@dataclass(frozen=True)
class ExecutionResult:
    status: ExecutionStatus
    output: str  # captured stdout/stderr, in arrival order
    duration_ms: int
    value_repr: str | None = None
    error: KernelError | None = None
    out_ref: int | None = None  # the runner stored the full text as _out[out_ref]
    restarted: bool = False  # this execution ran in a new kernel; old variables are gone

    @classmethod
    def from_message(cls, msg: dict[str, Any], output: str, restarted: bool) -> ExecutionResult:
        err = msg.get("error")
        error = (
            KernelError(str(err.get("ename")), str(err.get("evalue")), str(err.get("traceback")))
            if isinstance(err, dict)
            else None
        )
        out_ref = msg.get("out_ref")
        return cls(
            status="ok" if msg.get("status") == "ok" else "error",
            output=output,
            duration_ms=int(msg.get("duration_ms", 0)),
            value_repr=msg.get("value_repr"),
            error=error,
            out_ref=out_ref if isinstance(out_ref, int) else None,
            restarted=restarted,
        )


@dataclass(frozen=True)
class VarInfo:
    name: str
    type: str
    summary: str


def encode(msg: dict[str, Any]) -> bytes:
    return (json.dumps(msg, ensure_ascii=False) + "\n").encode("utf-8")
