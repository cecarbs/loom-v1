"""Path policy for the host-side file tools (docs/03 "Path policy")."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

# Shared with the seatbelt profile (M5): never readable without approval.
SENSITIVE_HOME_PATHS: tuple[str, ...] = (
    ".ssh",
    ".aws",
    ".config/gcloud",
    ".kube",
    ".docker",
    ".netrc",
    ".loom",
    "Library/Keychains",
    "Library/Application Support/Google/Chrome",
    "Library/Application Support/Firefox",
    "Library/Safari",
)


class PathPolicy:
    def __init__(
        self, workspace: Path, read_roots: Sequence[Path] = (), home: Path | None = None
    ) -> None:
        self.workspace = workspace.resolve()
        self.read_roots = [r.resolve() for r in read_roots]
        home = (home or Path.home()).resolve()
        self._sensitive = [home / p for p in SENSITIVE_HOME_PATHS]

    def resolve(self, raw: str) -> Path:
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = self.workspace / path
        return path.resolve()  # follows symlinks, so links can't escape the workspace

    def display(self, path: Path) -> str:
        return (
            str(path.relative_to(self.workspace))
            if path.is_relative_to(self.workspace)
            else str(path)
        )

    def read_approval_reason(self, path: Path) -> str | None:
        """Why reading `path` needs approval, or None if it's in the allowed roots."""
        if any(path.is_relative_to(root) for root in self.read_roots):
            return None
        if self._in_sensitive_dir(path) or _is_env_file(path):
            return "a sensitive path"
        if not path.is_relative_to(self.workspace):
            return "outside the workspace"
        return None

    def read_error(self, path: Path) -> str | None:
        """For readers that can't ask for approval (`@path` mentions)."""
        if reason := self.read_approval_reason(path):
            return f"{path} is {reason}; reading it needs approval (ask loom to read it)."
        return None

    def write_error(self, path: Path) -> str | None:
        if not path.is_relative_to(self.workspace):
            return f"{path} is outside the workspace; writing there is not allowed."
        if self._in_sensitive_dir(path):
            return f"{path} is a sensitive path; writing there is not allowed."
        return None

    def _in_sensitive_dir(self, path: Path) -> bool:
        return any(path.is_relative_to(s) for s in self._sensitive)


def _is_env_file(path: Path) -> bool:
    return path.name.startswith(".env")


@dataclass(frozen=True)
class _Stamp:
    mtime_ns: int
    sha256: str


class ReadSet:
    """Files the model has read this session, so write/edit can refuse stale or unseen files."""

    def __init__(self) -> None:
        self._stamps: dict[Path, _Stamp] = {}

    def record(self, path: Path, data: bytes) -> None:
        self._stamps[path] = _Stamp(path.stat().st_mtime_ns, hashlib.sha256(data).hexdigest())

    def check(self, path: Path) -> str | None:
        """None if the file was read and is unchanged since; otherwise an error message."""
        stamp = self._stamps.get(path)
        if stamp is None:
            return "Read the file first."
        if path.stat().st_mtime_ns == stamp.mtime_ns:
            return None
        if hashlib.sha256(path.read_bytes()).hexdigest() == stamp.sha256:
            return None
        return "The file changed since you last read it. Read it again first."
