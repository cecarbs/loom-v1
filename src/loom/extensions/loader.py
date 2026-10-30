"""Load extension modules (`*.py` with a `register(api)` function) from extension dirs.

A module that fails to import or register is skipped with a notice; nothing it registered
before failing is kept.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from collections.abc import Sequence
from pathlib import Path

from loom.extensions.api import ExtensionAPI, Extensions


def extension_files(directory: Path) -> list[Path]:
    """`*.py` directly in `directory`; `_`-prefixed files are helpers, not extensions."""
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.glob("*.py") if not p.name.startswith("_"))


def load_extensions(files: Sequence[Path]) -> tuple[Extensions, list[str]]:
    extensions = Extensions()
    notices: list[str] = []
    for path in files:
        name = path.stem
        if name in extensions.names:
            notices.append(f"Skipped extension {path}: an extension named {name!r} is loaded")
            continue
        api = ExtensionAPI(name)
        module_name = f"loom_extension_{name}_{hashlib.sha256(str(path).encode()).hexdigest()[:8]}"
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
            register(api)
        except Exception as e:
            sys.modules.pop(module_name, None)
            notices.append(f"Extension {path} failed to load: {type(e).__name__}: {e}")
            continue
        extensions.add(api)
    return extensions, notices
