"""Load extensions: installed ones (entry-point group `loom.extensions`, each a `register`
function) and modules (`*.py` with a `register(api)` function) from extension dirs.

An extension that fails to import or register is skipped with a notice; nothing it registered
before failing is kept.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from collections.abc import Callable, Sequence
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any

from loom.extensions.api import ExtensionAPI, ExtensionEnv, Extensions

ENTRY_POINT_GROUP = "loom.extensions"


def extension_files(directory: Path) -> list[Path]:
    """`*.py` directly in `directory`; `_`-prefixed files are helpers, not extensions."""
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.glob("*.py") if not p.name.startswith("_"))


def load_extensions(
    files: Sequence[Path], env: ExtensionEnv | None = None, *, installed: bool = False
) -> tuple[Extensions, list[str]]:
    """`files` in order, after the installed extensions if `installed`."""
    extensions = Extensions()
    notices: list[str] = []
    if installed:
        for ep in sorted(entry_points(group=ENTRY_POINT_GROUP), key=lambda e: e.name):
            _add(extensions, notices, ep.name, ep.name, ep.load, env)
    for path in files:
        _add(extensions, notices, path.stem, path, lambda p=path: _import_register(p), env)
    return extensions, notices


def _add(
    extensions: Extensions,
    notices: list[str],
    name: str,
    source: object,
    load: Callable[[], Callable[[ExtensionAPI], Any]],
    env: ExtensionEnv | None,
) -> None:
    if name in extensions.names:
        notices.append(f"Skipped extension {source}: an extension named {name!r} is loaded")
        return
    api = ExtensionAPI(name, env)
    try:
        load()(api)
    except Exception as e:
        notices.append(f"Extension {source} failed to load: {type(e).__name__}: {e}")
        return
    extensions.add(api)


def _import_register(path: Path) -> Callable[[ExtensionAPI], Any]:
    module_name = f"loom_extension_{path.stem}_{hashlib.sha256(str(path).encode()).hexdigest()[:8]}"
    try:
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError("not a Python module")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module  # dataclasses etc. look the module up
        spec.loader.exec_module(module)
        register = getattr(module, "register", None)
        if not callable(register):
            raise AttributeError("no register(api) function")
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return register
