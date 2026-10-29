"""Approval policy (docs/03 "Approvals for built-in tools"; the broker table arrives in M4).

Built-in tools are sandbox-aware: workspace changes are undoable and the sandbox contains
everything else, so they don't prompt; with backend `none`, shell and kernel run with full
host access, so they ask. Every decision that isn't a plain table "allow" is logged.
"""

from __future__ import annotations

import asyncio
import itertools
from dataclasses import dataclass
from typing import Literal

from loom.config import ApprovalMode, ApprovalsConfig
from loom.core.events import ApprovalRequested, Decision, EventBus
from loom.session.store import SessionLog

Rule = Literal["allow", "ask", "deny"]
# What a built-in tool call does, as far as the policy is concerned.
AccessKind = Literal["read", "outside_read", "write", "exec"]


@dataclass(frozen=True)
class Access:
    kind: AccessKind
    preview: str = ""  # shown in the approval prompt
    reason: str = ""  # why it needs approval (e.g. "outside the workspace")
    scope: str = ""  # what "allow for this session" covers; defaults to the tool name


def builtin_rule(kind: AccessKind, mode: ApprovalMode, *, sandboxed: bool) -> Rule:
    match kind:
        case "read":
            return "allow"
        case "outside_read":
            return "allow" if mode == "yolo" else "ask"
        case "write":
            return "deny" if mode == "read-only" else "allow"
        case "exec":
            if sandboxed or mode in ("auto-write", "yolo"):
                return "allow"
            return "ask"


class Approvals:
    def __init__(
        self, cfg: ApprovalsConfig, *, sandboxed: bool, bus: EventBus, session_log: SessionLog
    ) -> None:
        self.mode: ApprovalMode = cfg.mode
        self.overrides = dict(cfg.overrides)
        self.sandboxed = sandboxed
        self.bus = bus
        self.log = session_log
        self._session_allowed: set[str] = set()
        self._ids = itertools.count(1)

    async def check(self, tool: str, access: Access, *, call_id: str = "") -> Decision:
        """Decide whether a built-in tool call may run, asking the user if needed."""
        rule = builtin_rule(access.kind, self.mode, sandboxed=self.sandboxed)
        if rule == "allow" and tool not in self.overrides:
            return Decision("allow_once", by="policy")  # the common case; not logged

        scope = access.scope or tool
        override = self.overrides.get(tool)
        if override is not None:
            rule, by = override, "override"
        else:
            by = "policy"
        if rule == "allow":
            decision = Decision("allow_once", by=by)
        elif rule == "deny":
            decision = Decision("deny", by=by)
        elif scope in self._session_allowed:
            decision = Decision("allow_once", by="session")
        else:
            decision = await self._ask(tool, access, call_id)
            if decision.action == "allow_session":
                self._session_allowed.add(scope)
        self.log.append(
            "approval",
            tool=tool,
            call_id=call_id,
            access=access.kind,
            reason=access.reason or None,
            mode=self.mode,
            action=decision.action,
            by=decision.by,
            message=decision.message,
        )
        return decision

    async def _ask(self, tool: str, access: Access, call_id: str) -> Decision:
        effect = access.kind.replace("_", " ")
        preview = access.preview + (f"\n({access.reason})" if access.reason else "")
        request = ApprovalRequested(f"a{next(self._ids)}", tool, preview, effect)
        try:
            return await self.bus.request_approval(request)
        except asyncio.CancelledError:
            self.log.append(
                "approval",
                tool=tool,
                call_id=call_id,
                access=access.kind,
                mode=self.mode,
                action="deny",
                by="cancel",
                message=None,
            )
            raise


def denial_text(tool: str, decision: Decision, mode: ApprovalMode) -> str:
    """The error result the model sees for a denied call."""
    if decision.by == "user":
        return f"The user denied this {tool} call" + (
            f": {decision.message}" if decision.message else "."
        )
    if decision.by == "override":
        return f"{tool} is denied by the user's configuration."
    return f"{tool} is not allowed in {mode} mode."
