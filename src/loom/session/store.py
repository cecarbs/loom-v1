"""Session directories and the append-only JSONL session log (docs/07)."""

from __future__ import annotations

import hashlib
import json
import secrets
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from loom.broker.redact import Redactor


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


def loom_home() -> Path:
    return Path.home() / ".loom"


def project_hash(workspace: Path) -> str:
    return hashlib.sha256(str(workspace.resolve()).encode()).hexdigest()[:12]


def new_session_id(now: datetime | None = None) -> str:
    now = now or datetime.now()
    return f"{now:%Y%m%d-%H%M}-{secrets.token_hex(2)}"


def new_session(workspace: Path, home: Path | None = None) -> SessionPaths:
    session_id = new_session_id()
    root = (home or loom_home()) / "sessions" / project_hash(workspace) / session_id
    paths = SessionPaths(session_id, root)
    for d in (paths.out_dir, paths.sandbox_dir / "home", paths.sandbox_dir / "tmp"):
        d.mkdir(parents=True, exist_ok=True)
    return paths


class SessionLog:
    """Every record passes through the redactor: secret values never reach the log."""

    def __init__(self, path: Path, redactor: Redactor | None = None) -> None:
        self.path = path
        self.redactor = redactor

    def append(self, kind: str, **data: Any) -> None:
        record = {"ts": time.time(), "kind": kind, **data}
        if self.redactor is not None:
            record = self.redactor.value(record)
        line = json.dumps(record, ensure_ascii=False, default=str)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def read(self) -> list[dict[str, Any]]:
        with self.path.open(encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
