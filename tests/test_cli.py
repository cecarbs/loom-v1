import io
import subprocess
import sys
from pathlib import Path

import pytest

from loom import __version__
from loom.cli import main


@pytest.fixture
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Run with an empty HOME and workspace so real user config is never read."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    ws = tmp_path / "ws"
    ws.mkdir()
    monkeypatch.chdir(ws)
    return ws


def test_version_flag(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert capsys.readouterr().out.strip() == f"loom {__version__}"


def test_module_entry_point_version() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "loom", "--version"], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == f"loom {__version__}"


def test_interactive_exits_cleanly_on_eof(
    isolated: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    assert main([]) == 0
    assert "loom · sonnet" in capsys.readouterr().out
    assert list((isolated.parent / "home" / ".loom" / "sessions").glob("*/*/session.jsonl"))


def test_config_error_exits_2(isolated: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--model", "nope"]) == 2
    assert "unknown model alias 'nope'" in capsys.readouterr().err


def test_project_notices_printed(
    isolated: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    (isolated / ".loom").mkdir()
    (isolated / ".loom" / "config.toml").write_text('[approvals]\nmode = "yolo"\n')
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    main([])
    assert "approvals.mode (would loosen 'ask' to 'yolo')" in capsys.readouterr().err
