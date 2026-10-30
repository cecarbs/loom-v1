"""Workspace trust (docs/08 "Extensions", docs/09 "Project config restrictions").

One grant per workspace, stored in `~/.loom/trust.json` (loom never rewrites the user's
TOML), unlocks the project's extensions and the restricted keys of its `.loom/config.toml`.
The grant records a fingerprint of both; if either changes (e.g. model-written code edits
`.loom/extensions/`), the workspace is untrusted until the user grants it again.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from loom.config import project_config_path, restricted_project_keys
from loom.extensions.loader import extension_files

SKIP_DIRS = frozenset({"__pycache__"})


def project_extensions_dir(workspace: Path) -> Path:
    return workspace / ".loom" / "extensions"


def trust_file(home: Path) -> Path:
    return home / "trust.json"


def fingerprint(workspace: Path) -> str:
    """Hash of the project config and every file under the project extensions dir."""
    digest = hashlib.sha256()
    config = project_config_path(workspace)
    digest.update(b"config\0" + (config.read_bytes() if config.is_file() else b"") + b"\0")
    root = project_extensions_dir(workspace)
    if root.is_dir():
        for path in sorted(root.rglob("*")):
            rel = path.relative_to(root)
            if not path.is_file() or SKIP_DIRS & set(rel.parts) or path.suffix == ".pyc":
                continue
            digest.update(str(rel).encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


@dataclass(frozen=True)
class TrustRequest:
    """What a trust grant would unlock in a workspace."""

    extensions: list[Path]
    config_keys: list[str]

    def __bool__(self) -> bool:
        return bool(self.extensions or self.config_keys)

    def describe(self, workspace: Path) -> list[str]:
        lines = []
        if self.extensions:
            names = ", ".join(p.name for p in self.extensions)
            lines.append(f"extensions in {project_extensions_dir(workspace)}: {names}")
        if self.config_keys:
            lines.append(
                f"settings in {project_config_path(workspace)}: {', '.join(self.config_keys)}"
            )
        return lines


def trust_request(workspace: Path, home: Path, user_config: Path | None = None) -> TrustRequest:
    """`home` is loom's home: when the workspace is $HOME, its `.loom/extensions` are the
    user's own (always loaded) and need no grant."""
    directory = project_extensions_dir(workspace)
    is_user_dir = directory.resolve() == (home / "extensions").resolve()
    return TrustRequest(
        [] if is_user_dir else extension_files(directory),
        restricted_project_keys(workspace, user_path=user_config),
    )


def _read(home: Path) -> dict[str, dict[str, str]]:
    try:
        data = json.loads(trust_file(home).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    workspaces = data.get("workspaces") if isinstance(data, dict) else None
    return workspaces if isinstance(workspaces, dict) else {}


def is_trusted(workspace: Path, home: Path) -> bool:
    record = _read(home).get(str(workspace.resolve()))
    return isinstance(record, dict) and record.get("fingerprint") == fingerprint(workspace)


def grant(workspace: Path, home: Path) -> None:
    workspaces = _read(home)
    workspaces[str(workspace.resolve())] = {
        "fingerprint": fingerprint(workspace),
        "granted_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    home.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=home, prefix=".trust.", suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump({"version": 1, "workspaces": workspaces}, f, indent=2)
        f.write("\n")
    os.replace(tmp, trust_file(home))
