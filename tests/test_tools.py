import os
from pathlib import Path

import pytest

from loom.config import BudgetConfig
from loom.core.events import EventBus
from loom.tools.base import ToolContext, ToolResult
from loom.tools.budget import OutputBudget
from loom.tools.builtin.edit import EditArgs, EditTool
from loom.tools.builtin.read import ReadArgs, ReadTool
from loom.tools.builtin.write import WriteArgs, WriteTool
from loom.tools.paths import PathPolicy, ReadSet

PNG_1PX = bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000001000000010806000000"
    "1f15c4890000000d49444154789c6360000002000154a24f5d0000000049454e44ae426082"
)


@pytest.fixture
def ws(tmp_path: Path) -> Path:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    return workspace


@pytest.fixture
def ctx(ws: Path, tmp_path: Path) -> ToolContext:
    home = tmp_path / "home"
    (home / ".loom" / "sessions" / "s" / "out").mkdir(parents=True)
    policy = PathPolicy(ws, read_roots=[home / ".loom" / "sessions" / "s" / "out"], home=home)
    return ToolContext(ws, policy, ReadSet(), EventBus())


async def read(ctx: ToolContext, path: str, **kw) -> ToolResult:
    return await ReadTool().run(ReadArgs(path=path, **kw), ctx)


async def write(ctx: ToolContext, path: str, content: str) -> ToolResult:
    return await WriteTool().run(WriteArgs(path=path, content=content), ctx)


async def edit(ctx: ToolContext, path: str, old: str, new: str, **kw) -> ToolResult:
    args = EditArgs(path=path, old_string=old, new_string=new, **kw)
    return await EditTool().run(args, ctx)


# --- read ---------------------------------------------------------------------


async def test_read_numbers_lines_and_pages(ctx: ToolContext, ws: Path) -> None:
    (ws / "f.txt").write_text("\n".join(f"line {i}" for i in range(1, 11)))
    result = await read(ctx, "f.txt", offset=3, limit=2)
    assert result.text.splitlines() == [
        "     3\tline 3",
        "     4\tline 4",
        "[… 6 more lines; continue with offset=5]",
    ]


async def test_read_clips_long_lines(ctx: ToolContext, ws: Path) -> None:
    (ws / "f.txt").write_text("x" * 2500)
    result = await read(ctx, "f.txt")
    assert "[… 500 chars cut]" in result.text


async def test_read_errors(ctx: ToolContext, ws: Path) -> None:
    (ws / "bin").write_bytes(b"\x00\x01\x02")
    assert (await read(ctx, "missing.txt")).is_error
    assert (await read(ctx, ".")).is_error
    assert "binary" in (await read(ctx, "bin")).text
    (ws / "empty").write_text("")
    assert "empty" in (await read(ctx, "empty")).text


async def test_read_image(ctx: ToolContext, ws: Path) -> None:
    (ws / "dot.png").write_bytes(PNG_1PX)
    result = await read(ctx, "dot.png")
    assert result.images and result.images[0].media_type == "image/png"

    ctx.model_accepts_images = False
    result = await read(ctx, "dot.png")
    assert not result.images and "does not accept images" in result.text


async def test_read_outside_workspace_and_sensitive_paths(
    ctx: ToolContext, ws: Path, tmp_path: Path
) -> None:
    (tmp_path / "outside.txt").write_text("secret")
    (ws / ".env").write_text("TOKEN=1")
    ssh = tmp_path / "home" / ".ssh"
    ssh.mkdir(parents=True)
    (ssh / "id_rsa").write_text("key")

    assert "outside the workspace" in (await read(ctx, "../outside.txt")).text
    assert "sensitive" in (await read(ctx, ".env")).text
    assert "sensitive" in (await read(ctx, str(ssh / "id_rsa"))).text


async def test_read_symlink_escape_is_blocked(ctx: ToolContext, ws: Path, tmp_path: Path) -> None:
    (tmp_path / "outside.txt").write_text("secret")
    os.symlink(tmp_path / "outside.txt", ws / "link.txt")
    result = await read(ctx, "link.txt")
    assert result.is_error and "outside the workspace" in result.text


async def test_read_spill_dir_allowed_despite_being_under_loom(
    ctx: ToolContext, tmp_path: Path
) -> None:
    spill = tmp_path / "home" / ".loom" / "sessions" / "s" / "out" / "0001.txt"
    spill.write_text("spilled output")
    result = await read(ctx, str(spill))
    assert not result.is_error and "spilled output" in result.text


# --- write ---------------------------------------------------------------------


async def test_write_creates_file_and_parents(ctx: ToolContext, ws: Path) -> None:
    result = await write(ctx, "a/b/new.py", "x = 1\n")
    assert not result.is_error
    assert (ws / "a/b/new.py").read_text() == "x = 1\n"
    assert "+x = 1" in result.meta["diff"]


async def test_write_overwrite_requires_read(ctx: ToolContext, ws: Path) -> None:
    (ws / "f.txt").write_text("old")
    result = await write(ctx, "f.txt", "new")
    assert result.is_error and "Read the file first" in result.text
    await read(ctx, "f.txt")
    assert not (await write(ctx, "f.txt", "new")).is_error
    assert (ws / "f.txt").read_text() == "new"


async def test_write_refuses_stale_file(ctx: ToolContext, ws: Path) -> None:
    f = ws / "f.txt"
    f.write_text("old")
    await read(ctx, "f.txt")
    f.write_text("changed elsewhere")
    os.utime(f, ns=(f.stat().st_atime_ns, f.stat().st_mtime_ns + 1_000_000))
    result = await write(ctx, "f.txt", "new")
    assert result.is_error and "changed since" in result.text


async def test_write_outside_workspace_denied(ctx: ToolContext, tmp_path: Path) -> None:
    result = await write(ctx, str(tmp_path / "x.txt"), "x")
    assert result.is_error and "outside the workspace" in result.text
    assert not (tmp_path / "x.txt").exists()


# --- edit ---------------------------------------------------------------------


async def test_edit_flow(ctx: ToolContext, ws: Path) -> None:
    f = ws / "f.py"
    f.write_text("a = 1\nb = 1\n")
    assert "Read the file first" in (await edit(ctx, "f.py", "a = 1", "a = 2")).text
    await read(ctx, "f.py")

    assert "0 matches" in (await edit(ctx, "f.py", "zzz", "y")).text
    assert "2 matches" in (await edit(ctx, "f.py", "= 1", "= 3")).text

    result = await edit(ctx, "f.py", "a = 1", "a = 2")
    assert not result.is_error and "-a = 1" in result.meta["diff"]
    # The read set was refreshed, so a second edit works without re-reading.
    assert not (await edit(ctx, "f.py", "= 1", "= 9", replace_all=True)).is_error
    assert f.read_text() == "a = 2\nb = 9\n"


async def test_edit_rejects_noop_and_empty(ctx: ToolContext, ws: Path) -> None:
    (ws / "f").write_text("abc")
    await read(ctx, "f")
    assert (await edit(ctx, "f", "", "x")).is_error
    assert (await edit(ctx, "f", "a", "a")).is_error


async def test_edit_preserves_crlf(ctx: ToolContext, ws: Path) -> None:
    f = ws / "f.txt"
    f.write_bytes(b"one\r\ntwo\r\n")
    await read(ctx, "f.txt")
    await edit(ctx, "f.txt", "two", "three")
    assert f.read_bytes() == b"one\r\nthree\r\n"


# --- budget ---------------------------------------------------------------------


def test_budget_truncates_and_spills(tmp_path: Path) -> None:
    budget = OutputBudget(BudgetConfig(), tmp_path / "out")
    text = "".join(chr(65 + i % 26) for i in range(200_000))
    shown, spill = budget.apply(text)
    assert spill is not None and spill == tmp_path / "out" / "0001.txt"
    assert spill.read_text() == text
    assert shown.startswith(text[:8_000]) and shown.endswith(text[-3_000:])
    assert "189,000 chars omitted" in shown
    assert len(shown) < 12_000


def test_budget_passes_small_results_and_numbers_spills(tmp_path: Path) -> None:
    budget = OutputBudget(BudgetConfig(), tmp_path / "out")
    assert budget.apply("small") == ("small", None)
    budget.apply("x" * 20_000)
    shown, spill = budget.apply("y" * 20_000, kernel_ref=4)
    assert spill is not None and spill.name == "0002.txt"
    assert "In kernel: _out[4]" in shown
