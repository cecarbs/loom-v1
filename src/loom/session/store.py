"""Session directories and the append-only JSONL session log (docs/07)."""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import re
import secrets
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from loom.broker.redact import Redactor
from loom.core.events import Usage
from loom.core.messages import Message, TextBlock, ToolResultBlock, message_from_dict

logger = logging.getLogger(__name__)

SESSION_ID_RE = re.compile(r"^\d{8}-\d{4}-[0-9a-f]{4}$")


@dataclass(frozen=True)
class SessionPaths:
    id: str
    root: Path

    @property
    def log(self) -> Path:
        return self.root / "session.jsonl"

    @property
    def out_dir(self) -> Path:
        return self.root / "out"

    @property
    def sandbox_dir(self) -> Path:
        return self.root / "sandbox"

    @property
    def lock(self) -> Path:
        return self.root / "lock"

    @property
    def checkpoint_index(self) -> Path:
        return self.root / "checkpoint.index"


def loom_home() -> Path:
    return Path.home() / ".loom"


def project_hash(workspace: Path) -> str:
    return hashlib.sha256(str(workspace.resolve()).encode()).hexdigest()[:12]


def new_session_id(now: datetime | None = None) -> str:
    now = now or datetime.now()
    return f"{now:%Y%m%d-%H%M}-{secrets.token_hex(2)}"


def sessions_dir(workspace: Path, home: Path | None = None) -> Path:
    return (home or loom_home()) / "sessions" / project_hash(workspace)


def new_session(workspace: Path, home: Path | None = None) -> SessionPaths:
    session_id = new_session_id()
    paths = SessionPaths(session_id, sessions_dir(workspace, home) / session_id)
    for d in (paths.out_dir, paths.sandbox_dir / "home", paths.sandbox_dir / "tmp"):
        d.mkdir(parents=True, exist_ok=True)
    return paths


LogListener = Callable[[dict[str, Any]], None]


class SessionLog:
    """Every record passes through the redactor: secret values never reach the log.
    Listeners (the OTel exporter) get each record after it was written, redacted."""

    def __init__(self, path: Path, redactor: Redactor | None = None) -> None:
        self.path = path
        self.redactor = redactor
        self.listeners: list[LogListener] = []

    def append(self, kind: str, **data: Any) -> None:
        record = {"ts": time.time(), "kind": kind, **data}
        if self.redactor is not None:
            record = self.redactor.value(record)
        line = json.dumps(record, ensure_ascii=False, default=str)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        if not self.listeners:
            return
        written = json.loads(line)  # exactly what the log holds (default=str applied)
        for listener in self.listeners:
            try:
                listener(written)
            except Exception:  # an exporter bug must not break the session
                logger.warning("session log listener failed", exc_info=True)

    def read(self) -> list[dict[str, Any]]:
        with self.path.open(encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]


# --- listing and locking -------------------------------------------------------------


class SessionError(Exception):
    """A session can't be found or opened; the message is shown to the user."""


@dataclass(frozen=True)
class SessionSummary:
    id: str
    paths: SessionPaths
    modified: datetime
    model: str | None
    title: str  # first line of the first prompt, "" if none yet


def list_sessions(workspace: Path, home: Path | None = None) -> list[SessionSummary]:
    """This workspace's sessions, most recently active first."""
    root = sessions_dir(workspace, home)
    if not root.is_dir():
        return []
    out = []
    for d in root.iterdir():
        log = d / "session.jsonl"
        if not SESSION_ID_RE.match(d.name) or not log.is_file():
            continue
        model, title = _peek(log)
        modified = datetime.fromtimestamp(log.stat().st_mtime)
        out.append(SessionSummary(d.name, SessionPaths(d.name, d), modified, model, title))
    return sorted(out, key=lambda s: s.modified, reverse=True)


def _peek(log: Path) -> tuple[str | None, str]:
    """Model and title from the head of a log, without parsing all of it."""
    model = None
    with log.open(encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if record.get("kind") == "session_start":
                model = record.get("model")
            elif record.get("kind") == "message":
                msg = record.get("message", {})
                if msg.get("role") == "user":
                    texts = [b["text"] for b in msg.get("content", []) if b.get("type") == "text"]
                    if texts:
                        return model, first_line(texts[0])
    return model, ""


def first_line(text: str, limit: int = 80) -> str:
    line = text.strip().split("\n", 1)[0]
    return line if len(line) <= limit else line[: limit - 1] + "…"


def find_session(workspace: Path, ref: str | None, home: Path | None = None) -> SessionPaths:
    """`ref` is a session id or a unique prefix of one; None means the latest."""
    sessions = list_sessions(workspace, home)
    if not sessions:
        raise SessionError("no sessions in this workspace yet")
    if ref is None:
        return sessions[0].paths
    matches = [s for s in sessions if s.id == ref] or [s for s in sessions if s.id.startswith(ref)]
    if not matches:
        raise SessionError(f"no session {ref!r} in this workspace (see `loom sessions`)")
    if len(matches) > 1:
        ids = ", ".join(s.id for s in matches[:5])
        raise SessionError(f"session {ref!r} is ambiguous: {ids}")
    return matches[0].paths


def acquire_lock(paths: SessionPaths) -> None:
    """One loom process per session: two writers would interleave the log and fight over
    the sandbox container. A lock left by a dead process is taken over."""
    with contextlib.suppress(OSError, ValueError):
        pid = int(paths.lock.read_text().strip())
        if pid != os.getpid() and _alive(pid):
            raise SessionError(f"session {paths.id} is in use by another loom (pid {pid})")
    paths.lock.write_text(str(os.getpid()))


def release_lock(paths: SessionPaths) -> None:
    with contextlib.suppress(OSError, ValueError):
        if int(paths.lock.read_text().strip()) == os.getpid():
            paths.lock.unlink()


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


# --- resume --------------------------------------------------------------------------


@dataclass
class ResumedState:
    """What a session log says about the conversation, rebuilt for `loom -c`/`--resume`."""

    history: list[Message] = field(default_factory=list)
    turns: int = 0
    usage: Usage = field(default_factory=Usage)
    cost_usd: float | None = None
    last_prompt_tokens: int = 0
    model: str | None = None  # alias last used
    used_kernel: bool = False
    checkpoints: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)  # queued for the next prompt


def is_turn_start(msg: Message) -> bool:
    """A user prompt (not a tool-result message or a compaction summary)."""
    return (
        msg.role == "user"
        and not msg.meta.get("compaction")
        and any(isinstance(b, TextBlock) for b in msg.content)
        and not any(isinstance(b, ToolResultBlock) for b in msg.content)
    )


def rebuild(records: Iterator[dict[str, Any]] | list[dict[str, Any]]) -> ResumedState:
    """Replay `message` and `compaction` entries (docs/07). A compaction replaces the
    history before `first_kept` with its summary message, exactly as it did live."""
    state = ResumedState()
    for record in records:
        match record.get("kind"):
            case "session_start":
                state.model = record.get("model")
            case "model_switch":
                state.model = record.get("new")
            case "message":
                msg = message_from_dict(record["message"])
                state.history.append(msg)
                if is_turn_start(msg):
                    state.turns += 1
                    state.notes = []  # delivered with this prompt
            case "compaction":
                summary = message_from_dict(record["message"])
                state.history = [summary, *state.history[record["first_kept"] :]]
            case "model_request":
                usage = record.get("usage") or {}
                request = Usage(**{k: usage.get(k, 0) for k in Usage.__dataclass_fields__})
                state.usage = state.usage + request
                if record.get("purpose") != "compaction":
                    state.last_prompt_tokens = request.prompt_tokens
                if (cost := record.get("cost_usd")) is not None:
                    state.cost_usd = (state.cost_usd or 0.0) + cost
            case "kernel_start" | "kernel_execute":
                state.used_kernel = True
            case "checkpoint":
                state.checkpoints.append(record)
            case "note":
                state.notes.append(record["text"])
    return state
