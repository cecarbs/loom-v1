"""Shadow-git checkpoints and undo (docs/07 "Checkpoints and undo").

A git repo at `~/.loom/checkpoints/<project-hash>.git` with the workspace as its work tree,
independent of the user's own git. Each session uses its own index file (in the session
dir) and its own ref, `refs/loom/<session-id>`, so concurrent sessions don't interfere. The
repo lives outside every sandbox-visible path: model-written code can't rewrite history.

Snapshots respect the workspace's .gitignore plus DEFAULT_EXCLUDES; files over
MAX_FILE_BYTES are left out (and so are never restored or removed by undo).
"""

from __future__ import annotations

import asyncio
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from loom.session.store import SessionLog

MAX_FILE_BYTES = 5 * 1024 * 1024
DEFAULT_EXCLUDES = (".venv/", "venv/", "node_modules/", "__pycache__/", ".DS_Store")
GIT_TIMEOUT_S = 120.0
BEFORE_UNDO = "before /undo"

# The checkpoint commits need an identity, and the user's git config (hooks, fsmonitor,
# autocrlf, …) must not change what a snapshot contains.
_GIT_ENV = {
    "GIT_AUTHOR_NAME": "loom",
    "GIT_AUTHOR_EMAIL": "loom@localhost",
    "GIT_COMMITTER_NAME": "loom",
    "GIT_COMMITTER_EMAIL": "loom@localhost",
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_TERMINAL_PROMPT": "0",
}


class CheckpointError(Exception):
    pass


@dataclass(frozen=True)
class Checkpoint:
    sha: str
    turn: int
    prompt: str  # first line of the turn's prompt, or BEFORE_UNDO


@dataclass(frozen=True)
class Change:
    status: str  # "A" added, "M" modified, "D" deleted (relative to the older snapshot)
    path: str


@dataclass(frozen=True)
class UndoEntry:
    """A checkpoint as /undo lists it: restoring it reverts `changes`."""

    checkpoint: Checkpoint
    changes: list[Change]


@dataclass(frozen=True)
class UndoResult:
    restored: list[str]
    removed: list[str]


class Checkpoints:
    def __init__(
        self,
        workspace: Path,
        git_dir: Path,
        index_file: Path,
        session_id: str,
        session_log: SessionLog,
        history: list[Checkpoint] | None = None,
    ) -> None:
        self.workspace = workspace
        self.git_dir = git_dir
        self.index_file = index_file
        self.ref = f"refs/loom/{session_id}"
        self.log = session_log
        self.history: list[Checkpoint] = list(history or [])
        self.available = shutil.which("git") is not None
        self._pending: tuple[int, str] | None = None
        self._lock = asyncio.Lock()

    # --- recording -------------------------------------------------------------------

    def begin_turn(self, turn: int, prompt: str) -> None:
        """The next `ensure()` in this turn takes a snapshot; later ones don't."""
        self._pending = (turn, prompt)

    async def ensure(self) -> Checkpoint | None:
        """Snapshot before the turn's first workspace-mutating tool call."""
        async with self._lock:
            if self._pending is None or not self.available:
                return None
            turn, prompt = self._pending
            self._pending = None
            return await self._commit(turn, prompt)

    async def _commit(self, turn: int, prompt: str) -> Checkpoint:
        tree = await self.snapshot_tree()
        parent = self.history[-1].sha if self.history else None
        args = ["commit-tree", tree, "-m", f"turn {turn}: {prompt}"]
        if parent:
            args += ["-p", parent]
        sha = (await self._git(*args)).decode().strip()
        await self._git("update-ref", self.ref, sha)
        checkpoint = Checkpoint(sha, turn, prompt)
        self.history.append(checkpoint)
        changed = [c.path for c in await self.diff(parent, sha)] if parent else []
        self.log.append("checkpoint", sha=sha, turn=turn, prompt=prompt, files=changed)
        return checkpoint

    async def snapshot_tree(self) -> str:
        """Stage the workspace into this session's index and return the tree sha."""
        await self._init()
        listed = await self._git("ls-files", "-z", "--others", "--modified", "--exclude-standard")
        big = [p for p in _split(listed) if _too_big(self.workspace / p)]
        spec = b"\0".join([b".", *(b":(exclude,literal)" + _b(p) for p in big)]) + b"\0"
        await self._git("add", "-A", "--pathspec-from-file=-", "--pathspec-file-nul", input=spec)
        if big:  # tracked files that grew too big leave the snapshot
            spec = b"".join(b":(literal)" + _b(p) + b"\0" for p in big)
            await self._git(
                "rm", "--cached", "-q", "--ignore-unmatch", "--pathspec-from-file=-",
                "--pathspec-file-nul", input=spec,
            )  # fmt: skip
        return (await self._git("write-tree")).decode().strip()

    # --- undo ------------------------------------------------------------------------

    async def entries(self) -> list[UndoEntry]:
        """Checkpoints whose turn changed something, newest first."""
        if not self.history or not self.available:
            return []
        async with self._lock:
            now = await self.snapshot_tree()
        out = []
        for i, checkpoint in enumerate(self.history):
            after = self.history[i + 1].sha if i + 1 < len(self.history) else now
            if changes := await self.diff(checkpoint.sha, after):
                out.append(UndoEntry(checkpoint, changes))
        return out[::-1]

    async def restore(self, checkpoint: Checkpoint) -> UndoResult:
        """Make the workspace match `checkpoint`: restore changed and deleted files, remove
        files created since. The current state is checkpointed first, so an undo can be
        undone."""
        async with self._lock:
            before = await self._commit(self.history[-1].turn if self.history else 0, BEFORE_UNDO)
            changes = await self.diff(checkpoint.sha, before.sha)
            restore = [c.path for c in changes if c.status != "A"]
            remove = [c.path for c in changes if c.status == "A"]
            if restore:
                spec = b"".join(b":(literal)" + _b(p) + b"\0" for p in restore)
                await self._git(
                    "checkout", checkpoint.sha, "--pathspec-from-file=-", "--pathspec-file-nul",
                    input=spec,
                )  # fmt: skip
            removed = [p for p in remove if self._remove(p)]
            self.log.append(
                "undo", sha=checkpoint.sha, turn=checkpoint.turn, restored=restore, removed=removed
            )
            return UndoResult(restore, removed)

    def _remove(self, rel: str) -> bool:
        path = self.workspace / rel
        root = self.workspace.resolve()
        if not path.parent.resolve().is_relative_to(root):  # never follow a link outside
            return False
        try:
            path.unlink()
        except FileNotFoundError:
            return False
        parent = path.parent
        while parent != self.workspace and parent.resolve().is_relative_to(root):
            try:
                parent.rmdir()  # only if empty
            except OSError:
                break
            parent = parent.parent
        return True

    async def diff(self, old: str | None, new: str) -> list[Change]:
        if old is None:
            return []
        out = await self._git("diff-tree", "-r", "-z", "--no-renames", "--name-status", old, new)
        fields = _split(out)
        return [
            Change(status[0], path) for status, path in zip(fields[::2], fields[1::2], strict=True)
        ]

    # --- git -------------------------------------------------------------------------

    async def _init(self) -> None:
        if (self.git_dir / "HEAD").is_file():
            return
        self.git_dir.mkdir(parents=True, exist_ok=True)
        await self._git("init", "--quiet")
        await self._git("config", "core.bare", "false")
        exclude = self.git_dir / "info" / "exclude"
        exclude.parent.mkdir(parents=True, exist_ok=True)
        exclude.write_text("\n".join(DEFAULT_EXCLUDES) + "\n", encoding="utf-8")

    async def _git(self, *args: str, input: bytes | None = None) -> bytes:
        env = {
            **os.environ,
            **_GIT_ENV,
            "GIT_DIR": str(self.git_dir),
            "GIT_WORK_TREE": str(self.workspace),
            "GIT_INDEX_FILE": str(self.index_file),
        }
        proc = await asyncio.create_subprocess_exec(
            "git",
            *args,
            cwd=self.workspace,
            env=env,
            stdin=asyncio.subprocess.PIPE if input is not None else asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            out, err = await asyncio.wait_for(proc.communicate(input), GIT_TIMEOUT_S)
        except TimeoutError:
            proc.kill()
            await proc.wait()
            raise CheckpointError(f"git {args[0]} timed out after {GIT_TIMEOUT_S:g}s") from None
        if proc.returncode != 0:
            detail = err.decode(errors="replace").strip().splitlines()
            raise CheckpointError(f"git {args[0]} failed: {detail[-1] if detail else '?'}")
        return out


def _b(path: str) -> bytes:
    return path.encode(errors="surrogateescape")


def _split(out: bytes) -> list[str]:
    return [p.decode(errors="surrogateescape") for p in out.split(b"\0") if p]


def _too_big(path: Path) -> bool:
    try:
        return path.is_file() and not path.is_symlink() and path.stat().st_size > MAX_FILE_BYTES
    except OSError:
        return False
