"""Approval policy: one decision path for built-in tools (docs/03 "Approvals for built-in
tools") and kernel tool calls through the broker (docs/04 "Approval policy").

Built-in tools are sandbox-aware: workspace changes are undoable and the sandbox contains
everything else, so they don't prompt; with backend `none`, shell and kernel run with full
host access, so they ask. Broker tools are decided by their declared effect. Overrides and
"allow for this session" apply to both. Every decision that isn't a plain table "allow" is
logged.
"""

from __future__ import annotations

import asyncio
import copy
import itertools
from dataclasses import dataclass
from typing import Literal

from loom.config import ApprovalMode, ApprovalsConfig
from loom.core.events import ApprovalRequested, Decision, EventBus
from loom.session.store import SessionLog

Rule = Literal["allow", "ask", "deny"]
# What a call does, as far as the policy is concerned. Built-in tools use read,
# outside_read, write, exec; broker tools their declared effect (read, write, destructive).
AccessKind = Literal["read", "outside_read", "write", "exec", "destructive"]
Table = Literal["builtin", "broker"]


@dataclass(frozen=True)
class Access:
    kind: AccessKind
    preview: str = ""  # shown in the approval prompt
    reason: str = ""  # why it needs approval (e.g. "outside the workspace")
    scope: str = ""  # what "allow for this session" covers; defaults to the tool name
    table: Table = "builtin"


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
        case "destructive":  # not a built-in kind; treat like the broker would
            return broker_rule(kind, mode)


def broker_rule(effect: AccessKind, mode: ApprovalMode) -> Rule:
    if mode == "yolo" or effect == "read":
        return "allow"
    if mode == "read-only":
        return "deny"
    if effect == "write" and mode == "auto-write":
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
        self.agent_id: str | None = None
        self._session_allowed: set[str] = set()
        self._ids = itertools.count(1)

    def scoped(self, agent_id: str, session_log: SessionLog) -> Approvals:
        """The approvals of a sub-agent: same mode, overrides, bus, "allow for session" grants
        and request ids (they share the bus), but labelled and logged to its transcript. The
        mode is copied: it only changes between turns, and sub-agents don't outlive a turn."""
        scoped = copy.copy(self)
        scoped.agent_id = agent_id
        scoped.log = session_log
        return scoped

    async def check(self, tool: str, access: Access, *, call_id: str = "") -> Decision:
        """Decide whether a call may run, asking the user if needed."""
        if access.table == "broker":
            rule = broker_rule(access.kind, self.mode)
        else:
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
        request = ApprovalRequested(f"a{next(self._ids)}", tool, preview, effect, self.agent_id)
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
    """The error result the model sees for a denied built-in tool call."""
    if decision.by == "user":
        return f"The user denied this {tool} call" + (
            f": {decision.message}" if decision.message else "."
        )
    if decision.by == "override":
        return f"{tool} is denied by the user's configuration."
    return f"{tool} is not allowed in {mode} mode."
