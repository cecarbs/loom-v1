"""Architecture invariant 1: the core never imports the TUI (directly or transitively)."""

import pkgutil
import subprocess
import sys

import loom.core

# loom.app is shared by every UI, so it is held to the same rule.
CHECK = """
import importlib, pkgutil, sys
import loom.core
names = ["loom.app"] + [
    m.name for m in pkgutil.walk_packages(loom.core.__path__, "loom.core.")
]
for name in names:
    importlib.import_module(name)
leaked = sorted(m for m in sys.modules if m == "loom.tui" or m.startswith("loom.tui."))
print(len(names), leaked)
assert not leaked, leaked
"""


def test_core_modules_exist() -> None:
    names = {m.name for m in pkgutil.walk_packages(loom.core.__path__, "loom.core.")}
    assert {"loom.core.agent", "loom.core.events", "loom.core.mentions"} <= names


def test_core_never_imports_tui() -> None:
    # A fresh interpreter, so modules imported by other tests can't hide a leak.
    result = subprocess.run(
        [sys.executable, "-c", CHECK], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr
